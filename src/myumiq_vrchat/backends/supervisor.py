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

from ..actuation import (
    ActuatorCompositor,
    HandInputCommand,
    LocomotionCommand,
    PoseTarget,
    ServoEmission,
)
from ..body import ActuationTarget
from .osc import DeviceOutput, LiveConfig, MockOutput
from .vmt import LifecycleError, SafetyConfig, State, StopPolicy, VMTBackend


def trajectory_wire(command):
    """Bound JSON size without truncating the horizon (at most 0.5 nm error)."""

    def compact(value):
        if isinstance(value, float):
            return round(value, 9)
        if isinstance(value, list):
            return [compact(item) for item in value]
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()}
        return value

    return compact(command.model_dump(mode="json"))


def _worker(config_json, safety, token, ready, stop, failed, log_path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    output = (
        DeviceOutput(LiveConfig.model_validate_json(config_json)) if config_json else MockOutput()
    )
    backend = VMTBackend(output, safety)
    compositor = ActuatorCompositor(
        pose_ttl_s=safety.target_timeout, input_ttl_s=min(0.15, safety.target_timeout)
    )
    next_frame, output_sequence = 0.0, 0
    producer = None
    previous, halted_at = None, None
    timing_start, timing_frames, last_frame = time.perf_counter(), 0, None
    timing_max = {"frame_gap_s": 0.0, "compose_s": 0.0, "submit_s": 0.0, "receive_s": 0.0}
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
            if now - timing_start >= 1:
                log.write(
                    json.dumps(
                        {
                            "event": "servo_timing",
                            "timestamp": now,
                            "frames": timing_frames,
                            "window_s": now - timing_start,
                            "maximum": timing_max,
                        }
                    )
                    + "\n"
                )
                timing_start, timing_frames = now, 0
                timing_max = dict.fromkeys(timing_max, 0.0)
            if halted_at is None and now >= next_frame:
                target = compositor.compose(now)
                composed = time.perf_counter()
                timing_max["compose_s"] = max(timing_max["compose_s"], composed - now)
                if target is not None:
                    # Replaying a pose is not evidence of a fresh producer. Keep
                    # its original timestamp so the existing watchdog still expires.
                    try:
                        backend.submit(lease, output_sequence, compositor.stamps["pose"], target)
                    except (LifecycleError, ValueError):
                        # Expiry between poll and submission must still follow
                        # the normal neutral/disable retry lifecycle. The clock
                        # can cross expiry inside message freshness validation,
                        # after the backend's own lease check has passed.
                        backend.poll()
                        if backend.status.state != State.TIMED_OUT:
                            raise
                        record()
                    else:
                        timing_max["submit_s"] = max(
                            timing_max["submit_s"], time.perf_counter() - composed
                        )
                        if last_frame is not None:
                            timing_max["frame_gap_s"] = max(
                                timing_max["frame_gap_s"], now - last_frame
                            )
                        timing_frames += 1
                        last_frame = now
                        output_sequence += 1
                        if compositor.joint_servo is not None and producer is not None:
                            emission = ServoEmission(
                                epoch=compositor.joint_servo.command.epoch,
                                timestamp=time.perf_counter(),
                                pose=PoseTarget.from_target(target),
                            )
                            try:
                                sock.sendto(
                                    json.dumps(
                                        {
                                            "token": token,
                                            "emission": emission.model_dump(mode="json"),
                                        },
                                        separators=(",", ":"),
                                    ).encode(),
                                    producer,
                                )
                            except BlockingIOError:
                                pass  # Missing reports expire; never block the output owner.
                next_frame += 1 / 60
                if next_frame <= now:
                    next_frame = now + 1 / 60
            try:
                data, addr = sock.recvfrom(32769)
            except BlockingIOError:
                # Winsock receive timeouts can round to a much coarser clock
                # than the servo period. Keep IPC nonblocking; Python's timed
                # sleep lets this existing owner service its own deadline.
                time.sleep(max(0.0001, min(0.001, next_frame - time.perf_counter())))
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
            producer = addr
            try:
                receive_started = time.perf_counter()
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
                timing_max["receive_s"] = max(
                    timing_max["receive_s"], time.perf_counter() - receive_started
                )
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
        self._socket.bind(("127.0.0.1", 0))
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
            packet["trajectory"] = trajectory_wire(trajectory)
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

    def emissions(self):
        """Bounded reports of poses actually submitted by this owner, not device readback."""
        results = []
        with self._publish_guard:
            for _ in range(64):
                try:
                    data, addr = self._socket.recvfrom(32769)
                except BlockingIOError:
                    break
                if addr != ("127.0.0.1", self._port) or len(data) > 32768:
                    continue
                try:
                    packet = json.loads(data)
                    if packet.get("token") != self._token:
                        continue
                    report = ServoEmission.model_validate_json(json.dumps(packet["emission"]))
                    if 0 <= time.perf_counter() - report.timestamp < 0.3:
                        results.append(report)
                except (ValueError, KeyError, AttributeError):
                    continue
        return results

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
