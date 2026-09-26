"""VMT lifecycle guard. Transport and worker scheduling remain separate.

The output and monotonic clock are injected. The independent supervisor serializes
calls and polls even when the producer stops. See docs/vmt-backend.md.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import uuid4


class StopPolicy(StrEnum):
    DISABLE_OWNED = "disable_owned"
    SAFE_POSE = "safe_pose"
    RESET_ALL = "reset_all"


class State(StrEnum):
    NEW = "new"
    WAITING = "waiting"
    ACTIVE = "active"
    TIMED_OUT = "timed_out"
    FAULTED = "faulted"
    CLOSED = "closed"


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class SafetyConfig:
    """Seconds; defaults are examples, not validated deployment settings."""

    heartbeat_timeout: float = 0.5
    target_timeout: float = 0.5
    retry_interval: float = 0.1
    stop_policy: StopPolicy = StopPolicy.DISABLE_OWNED
    exclusive_vmt: bool = False

    def __post_init__(self) -> None:
        for value in (self.heartbeat_timeout, self.target_timeout, self.retry_interval):
            if not _finite_number(value) or value <= 0:
                raise ValueError("timeouts and retry interval must be finite and positive")
        if not isinstance(self.stop_policy, StopPolicy):
            raise ValueError("stop_policy must be a StopPolicy")
        if type(self.exclusive_vmt) is not bool:
            raise ValueError("exclusive_vmt must be explicit boolean ownership")
        if self.stop_policy == StopPolicy.RESET_ALL and not self.exclusive_vmt:
            raise ValueError("Reset affects all VMT devices; exclusive ownership required")


@dataclass(frozen=True)
class Status:
    state: State
    reason: str | None = None
    errors: tuple[str, ...] = ()


class LifecycleError(RuntimeError):
    """The caller must explicitly arm or use a current lease."""


class OutputError(RuntimeError):
    """An output attempt failed; successful sends are still not acknowledgements."""


class VMTOutput[T](Protocol):
    """Bounded, serialized output port, implemented separately by DeviceOutput."""

    def validate_stop(self, policy: StopPolicy) -> None:
        """Pure validation of supported policy, owned devices and safe-pose calibration."""
        ...

    def validate_target(self, target: T) -> None:
        """Pure validation of a complete immutable target; no partial output."""
        ...

    def send_target(self, target: T) -> None: ...

    def neutralize(self) -> None:
        """Release every owned profile control, including untouched retained inputs."""
        ...

    def stop_pose(self, policy: StopPolicy) -> None:
        """Disable owned devices, use calibrated poses only, or explicitly reset all."""
        ...

    def close(self) -> None: ...


class VMTBackend[T]:
    """Lease and timeout guard; call poll from outside the producer loop."""

    def __init__(
        self,
        output: VMTOutput[T],
        config: SafetyConfig = SafetyConfig(),
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._output, self._config, self._clock = output, config, clock
        self._state = State.NEW
        self._reason: str | None = None
        self._errors: tuple[str, ...] = ()
        self._lease: str | None = None
        self._armed_at = 0.0
        self._heartbeat_at: float | None = None
        self._target_at: float | None = None
        self._sequences = {"heartbeat": -1, "target": -1}
        self._last_safety_at = float("-inf")

    @property
    def status(self) -> Status:
        """Snapshot of attempts/state, never proof of VMT or avatar state."""
        return Status(self._state, self._reason, self._errors)

    def _safety_attempt(self, now: float) -> tuple[str, ...]:
        self._last_safety_at = now
        errors = []
        # Pose stopping must still run if input release failed.
        for name, action in (
            ("neutralize", self._output.neutralize),
            ("stop_pose", lambda: self._output.stop_pose(self._config.stop_policy)),
        ):
            try:
                action()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return tuple(errors)

    def _halt(self, state: State, reason: str, now: float) -> None:
        self._state, self._reason, self._lease = state, reason, None
        self._errors = self._safety_attempt(now)
        if self._errors:
            self._state = State.FAULTED

    def arm(self) -> str:
        """Neutralize first, then issue a new lease; never replay an old target."""
        if self._state not in (State.NEW, State.TIMED_OUT, State.FAULTED):
            raise LifecycleError("arm requires a new or stopped backend")
        self._output.validate_stop(self._config.stop_policy)
        self._halt(State.WAITING, "startup", self._clock())
        if self._errors:
            raise OutputError("; ".join(self._errors))
        self._armed_at = self._clock()
        self._heartbeat_at = self._target_at = None
        self._sequences = {"heartbeat": -1, "target": -1}
        self._reason = None
        self._lease = uuid4().hex
        return self._lease

    def poll(self) -> Status:
        """Enforce deadlines and retry stopping; ordinary output errors are latched."""
        now = self._clock()
        if self._state in (State.WAITING, State.ACTIVE):
            for name, stamp, timeout in (
                ("heartbeat", self._heartbeat_at, self._config.heartbeat_timeout),
                ("target", self._target_at, self._config.target_timeout),
            ):
                start = self._armed_at if stamp is None else stamp
                if now - start >= timeout:
                    self._halt(State.TIMED_OUT, f"{name}_timeout", now)
                    break
        elif self._state in (State.TIMED_OUT, State.FAULTED):
            if now - self._last_safety_at >= self._config.retry_interval:
                self._errors = self._safety_attempt(now)
                if self._errors:
                    self._state = State.FAULTED
        return self.status

    def _check_lease(self, lease: str) -> float:
        # A packet arriving at/after expiry must not postpone an unpolled timeout.
        self.poll()
        if self._state not in (State.WAITING, State.ACTIVE) or lease != self._lease:
            raise LifecycleError("backend stopped or lease invalid; explicit arm required")
        return self._clock()

    def _check_message(self, stream: str, sequence: int, generated_at: float, now: float) -> None:
        timeout = (
            self._config.heartbeat_timeout if stream == "heartbeat" else self._config.target_timeout
        )
        previous = self._heartbeat_at if stream == "heartbeat" else self._target_at
        if type(sequence) is not int or sequence <= self._sequences[stream]:
            raise ValueError("sequence must strictly increase within its stream/lease")
        if not _finite_number(generated_at) or not 0 <= now - generated_at < timeout:
            raise ValueError("message must be fresh in the shared monotonic clock domain")
        if generated_at < self._armed_at:
            raise ValueError("message predates the current arm; fresh data required")
        if previous is not None and generated_at < previous:
            raise ValueError("message generation time went backwards")

    def heartbeat(self, lease: str, sequence: int, generated_at: float) -> None:
        now = self._check_lease(lease)
        self._check_message("heartbeat", sequence, generated_at, now)
        self._heartbeat_at = generated_at
        self._sequences["heartbeat"] = sequence

    def submit(self, lease: str, sequence: int, generated_at: float, target: T) -> None:
        now = self._check_lease(lease)
        self._check_message("target", sequence, generated_at, now)
        if self._heartbeat_at is None:
            raise LifecycleError("a fresh heartbeat is required before the first target")
        self._output.validate_target(target)
        # Validation is pure, but could consume time. Do not send an expired target.
        now = self._check_lease(lease)
        self._check_message("target", sequence, generated_at, now)
        try:
            self._output.send_target(target)
        except Exception as exc:
            self._halt(State.FAULTED, "output_error", self._clock())
            self._errors = (f"send_target: {type(exc).__name__}: {exc}",) + self._errors
            raise OutputError("; ".join(self._errors)) from exc
        self._target_at = generated_at
        self._sequences["target"] = sequence
        self._state = State.ACTIVE
        self.poll()

    def close(self) -> None:
        """Terminal best-effort release/stop/close; errors remain observable."""
        if self._state == State.CLOSED:
            return
        self._state, self._reason, self._lease = State.CLOSED, "shutdown", None
        try:
            self._errors = self._safety_attempt(self._clock())
        finally:
            try:
                self._output.close()
            except Exception as exc:
                self._errors += (f"close: {type(exc).__name__}: {exc}",)
        if self._errors:
            raise OutputError("; ".join(self._errors))
