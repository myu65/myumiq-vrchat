"""Independent output owner: bounded loopback UDP IPC, no unpickling of IPC data.

The worker keeps polling after a producer stall/crash. It cannot protect against
its own crash, OS suspend, runtime failure, or dropped UDP safety packets.
"""

import json
import multiprocessing as mp
import secrets
import socket
import time
from pathlib import Path
from threading import Lock

from ..actuation import ActuatorCompositor, HandInputCommand, LocomotionCommand, PoseTarget
from ..body import ActuationTarget
from .osc import DeviceOutput, LiveConfig, MockOutput
from .vmt import SafetyConfig, State, StopPolicy, VMTBackend


def _worker(config_json, safety, token, ready, stop, failed, log_path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.01)
    output = (
        DeviceOutput(LiveConfig.model_validate_json(config_json)) if config_json else MockOutput()
    )
    backend = VMTBackend(output, safety)
    compositor = ActuatorCompositor(
        pose_ttl_s=safety.target_timeout, input_ttl_s=min(0.15, safety.target_timeout)
    )
    next_frame, output_sequence = 0.0, 0
    previous, halted_at = None, None
    log = open(log_path, "x", encoding="utf-8", buffering=1)

    def record():
        nonlocal previous, halted_at
        status = backend.status
        if status.errors:
            failed.set()
        key = (status.state, status.reason, status.errors)
        if key != previous:
            log.write(
                json.dumps(
                    {
                        "timestamp": time.perf_counter(),
                        "state": status.state,
                        "reason": status.reason,
                        "errors": status.errors,
                    }
                )
                + "\n"
            )
            previous = key
        if status.state in (State.TIMED_OUT, State.FAULTED) and halted_at is None:
            failed.set()
            halted_at = time.perf_counter()

    try:
        lease = backend.arm()
        record()
        ready.send({"port": sock.getsockname()[1]})
        ready.close()
        while not stop.is_set():
            # A maximum of one datagram per tick prevents flood starvation of poll.
            backend.poll()
            record()
            if halted_at is not None and time.perf_counter() - halted_at >= 2:
                break  # Repeated neutral/disable attempts, then bounded orphan exit.
            now = time.perf_counter()
            if halted_at is None and now >= next_frame:
                target = compositor.compose(now)
                if target is not None:
                    # Replaying a pose is not evidence of a fresh producer. Keep
                    # its original timestamp so the existing watchdog still expires.
                    backend.submit(lease, output_sequence, compositor.stamps["pose"], target)
                    output_sequence += 1
                next_frame = now + 1 / 60
            sock.settimeout(max(0.0001, min(0.01, next_frame - time.perf_counter())))
            try:
                data, addr = sock.recvfrom(32769)
            except socket.timeout:
                continue
            if addr[0] != "127.0.0.1" or len(data) > 32768:
                continue
            if halted_at is not None:
                continue  # Remain stopped and retry safety output; do not accept a new lease.
            try:
                packet = json.loads(data)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(packet, dict) or packet.get("token") != token:
                continue
            try:
                if packet["kind"] == "heartbeat":
                    backend.heartbeat(lease, packet["sequence"], packet["timestamp"])
                elif packet["kind"] in ("frame", "trajectory"):
                    backend.heartbeat(lease, packet["sequence"], packet["timestamp"])
                    target = ActuationTarget.model_validate_json(json.dumps(packet["target"]))
                    if packet["kind"] == "trajectory":
                        from ..joint_trajectory import JointTrajectory

                        command = JointTrajectory.model_validate_json(
                            json.dumps(packet["trajectory"])
                        )
                        compositor.publish_trajectory(
                            target, command, packet["timestamp"], time.perf_counter()
                        )
                    else:
                        compositor.publish_frame(target, packet["timestamp"], time.perf_counter())
                elif packet["kind"] in ("pose", "locomotion", "hands"):
                    backend.heartbeat(lease, packet["sequence"], packet["timestamp"])
                    types = {
                        "pose": PoseTarget,
                        "locomotion": LocomotionCommand,
                        "hands": HandInputCommand,
                    }
                    command = types[packet["kind"]].model_validate_json(
                        json.dumps(packet["target"])
                    )
                    methods = {
                        "pose": compositor.publish_pose,
                        "locomotion": compositor.publish_locomotion,
                        "hands": compositor.publish_hands,
                    }
                    methods[packet["kind"]](command, packet["timestamp"], time.perf_counter())
                else:
                    raise ValueError("unknown supervisor message")
            except Exception as exc:
                failed.set()
                # heartbeat/submit can discover expiry between polling and IPC.
                # Preserve that cause before close replaces it with shutdown.
                record()
                # Authenticated malformed/stale traffic cannot silently continue motion.
                log.write(
                    json.dumps(
                        {
                            "timestamp": time.perf_counter(),
                            "rejected": type(exc).__name__,
                            "detail": str(exc),
                        }
                    )
                    + "\n"
                )
                backend.close()
                break
            record()
    except Exception as exc:
        failed.set()
        record()  # Preserve a watchdog expiry discovered during a servo send.
        log.write(json.dumps({"error": type(exc).__name__, "detail": str(exc)}) + "\n")
        raise
    finally:
        try:
            backend.close()
        finally:
            record()
            sock.close()
            ready.close()
            log.close()


class OutputSupervisor:
    def __init__(
        self, log_path: Path, config: LiveConfig | None = None, safety: SafetyConfig | None = None
    ):
        ctx = mp.get_context("spawn")
        self._stop = ctx.Event()
        self._failed = ctx.Event()
        self._receive, send = ctx.Pipe(duplex=False)
        self._token = secrets.token_hex(32)
        self._sequence = 0
        self._publish_guard = Lock()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._port = None
        self._process = ctx.Process(
            target=_worker,
            name="myumiq-output",
            args=(
                config.model_dump_json() if config else None,
                safety
                or SafetyConfig(
                    heartbeat_timeout=config.heartbeat_timeout if config else 0.5,
                    target_timeout=config.target_timeout if config else 0.5,
                    stop_policy=config.stop_policy if config else StopPolicy.DISABLE_OWNED,
                ),
                self._token,
                send,
                self._stop,
                self._failed,
                str(log_path),
            ),
            daemon=False,
        )

    def start(self):
        self._process.start()
        if not self._receive.poll(10):
            self.close()
            raise RuntimeError("output supervisor did not become ready")
        self._port = self._receive.recv()["port"]
        self._receive.close()
        return self

    @property
    def alive(self):
        return self._process.is_alive() and not self._failed.is_set()

    @property
    def failed(self):
        return self._failed.is_set()

    def _publish(self, kind, target=None, trajectory=None):
        with self._publish_guard:
            self._publish_locked(kind, target, trajectory)

    def _publish_locked(self, kind, target=None, trajectory=None):
        if not self.alive or self._port is None:
            raise RuntimeError("output supervisor is not running")
        packet = {
            "kind": kind,
            "token": self._token,
            "sequence": self._sequence,
            "timestamp": time.perf_counter(),
        }
        if target is not None:
            packet["target"] = target.model_dump(mode="json")
        if trajectory is not None:
            packet["trajectory"] = trajectory.model_dump(mode="json")
        self._sequence += 1
        data = json.dumps(packet, allow_nan=False, separators=(",", ":")).encode()
        if len(data) > 32768:
            raise ValueError("frame exceeds IPC packet size")
        self._socket.sendto(data, ("127.0.0.1", self._port))

    def publish(self, target: ActuationTarget):
        self._publish("frame", target)

    def publish_trajectory(self, target, trajectory):
        self._publish("trajectory", target, trajectory)

    def publish_pose(self, target: PoseTarget):
        """Refresh pose only; retained inputs keep their own expiry."""
        self._publish("pose", target)

    def publish_locomotion(self, command: LocomotionCommand):
        self._publish("locomotion", command)

    def publish_hands(self, command: HandInputCommand):
        self._publish("hands", command)

    def heartbeat(self):
        self._publish("heartbeat")

    def close(self):
        self._stop.set()
        if self._process.pid is not None:
            self._process.join(5)
            if self._process.is_alive():
                # Do not silently kill the only watchdog; report unresolved cleanup.
                raise RuntimeError("output worker did not stop; safety output is unconfirmed")
        self._socket.close()
        self._receive.close()
        if self._process.exitcode not in (None, 0):
            raise RuntimeError(f"output worker exited with code {self._process.exitcode}")
