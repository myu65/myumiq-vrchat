"""Official VMT v0.15 and VirtualHMD_OpenVR v0.1 protocol adapters.

No registration, display settings or RoomMatrix mutations. UDP sends are attempts,
not acknowledgements. Calibration belongs to an external, machine-local config.
"""

import math
import socket
import struct
from collections.abc import Callable, Iterable
from typing import Self

from pydantic import Field, model_validator
from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder

from ..body import EXTRA_PARTS, ActuationTarget, BodyPart, Controls, Frozen, Number, Pose, compose
from .vmt import StopPolicy


class TrackerBinding(Frozen):
    part: BodyPart
    index: int = Field(ge=0, le=57)
    # Transform from canonical anatomical frame to the virtual device mount.
    body_from_tracker: Pose = Pose(position=(0.0, 0.0, 0.0))


class LiveConfig(Frozen):
    vmt_port: int = Field(default=39570, ge=1, le=65535)
    hmd_port: int = Field(default=4242, ge=1, le=65535)
    left_index: int = Field(default=1, ge=0, le=57)
    right_index: int = Field(default=2, ge=0, le=57)
    trackers: tuple[TrackerBinding, ...] = Field(default=(), max_length=8)
    full_body_envelope: Number | None = Field(default=None, gt=0, le=2.5)
    # OpenVR-coordinate rigid transforms from canonical-stage-converted space.
    vmt_from_stage: Pose
    hmd_from_stage: Pose
    safe_target: ActuationTarget
    calibrated: bool
    console_validated: bool
    heartbeat_timeout: Number = Field(default=0.5, ge=0.1, le=5)
    target_timeout: Number = Field(default=0.5, ge=0.1, le=5)
    stop_policy: StopPolicy = StopPolicy.DISABLE_OWNED

    @model_validator(mode="after")
    def ready(self) -> Self:
        if self.stop_policy == StopPolicy.RESET_ALL:
            raise ValueError("live output never resets other VMT devices")
        if self.left_index == self.right_index or self.vmt_port == self.hmd_port:
            raise ValueError("devices and endpoint ports must be distinct")
        if self.trackers:
            if {t.part for t in self.trackers} != set(EXTRA_PARTS):
                raise ValueError("full-body output requires all eight extra anatomical trackers")
            if len({self.left_index, self.right_index, *(t.index for t in self.trackers)}) != 10:
                raise ValueError("owned tracker and controller indices must be unique")
            if not self.safe_target.is_full_body or self.full_body_envelope is None:
                raise ValueError(
                    "full-body output needs a complete safe pose and calibrated envelope"
                )
        elif self.full_body_envelope is not None or any(
            getattr(self.safe_target, part) is not None for part in EXTRA_PARTS
        ):
            raise ValueError("full-body settings require tracker ownership")
        if not self.calibrated or not self.console_validated:
            raise ValueError(
                "live output requires completed console and coordinate calibration gates"
            )
        if (
            self.safe_target.left.controls != Controls()
            or self.safe_target.right.controls != Controls()
        ):
            raise ValueError("safe target must release all controller inputs and open the fingers")
        return self


def to_openvr(pose: Pose) -> Pose:
    x, y, z = pose.position
    w, qx, qy, qz = pose.orientation
    return Pose(position=(-y, z, -x), orientation=(w, -qy, qz, -qx))


def from_openvr(pose: Pose) -> Pose:
    x, y, z = pose.position
    w, qx, qy, qz = pose.orientation
    return Pose(position=(-z, -x, y), orientation=(w, -qz, -qx, qy))


def vmt_pose(index: int, mode: int, pose: Pose, frame: Pose):
    p = compose(frame, to_openvr(pose))
    w, x, y, z = p.orientation
    return ("/VMT/Raw/Driver", [index, mode, 0.0, *p.position, x, y, z, w])


def vmt_controls(index: int, c: Controls):
    for address, values in (
        ("Button", c.buttons),
        ("Button/Touch", c.button_touches),
        ("Trigger", c.triggers),
        ("Trigger/Touch", c.trigger_touches),
        ("Trigger/Click", c.trigger_clicks),
        ("Joystick", c.sticks),
        ("Joystick/Touch", c.stick_touches),
        ("Joystick/Click", c.stick_clicks),
    ):
        for channel, value in enumerate(values):
            payload = (
                list(value)
                if isinstance(value, tuple)
                else [int(value) if isinstance(value, bool) else float(value)]
            )
            yield ("/VMT/Input/" + address, [index, channel, 0.0, *payload])
    # Scalar root/wrist and all five fingers, followed by mandatory Apply.
    for bone, curl in enumerate((0.0, *c.curls)):
        yield ("/VMT/Skeleton/Scalar", [index, bone, 1.0 - curl, 0, 0])
    yield ("/VMT/Skeleton/Apply", [index, 0.0])


def osc_packets(messages: Iterable[tuple[str, list]]) -> list[bytes]:
    """Small immediate bundles to avoid IP fragmentation; explicit OSC scalar types."""
    result, pending, size = [], [], 16

    def bundle(items):
        b = OscBundleBuilder(IMMEDIATELY)
        for item in items:
            b.add_content(item)
        return b.build().dgram

    for address, args in messages:
        builder = OscMessageBuilder(address=address)
        for arg in args:
            builder.add_arg(arg, "i" if type(arg) is int else "f")
        message = builder.build()
        if size + 4 + len(message.dgram) > 1200 and pending:
            result.append(bundle(pending))
            pending, size = [], 16
        pending.append(message)
        size += 4 + len(message.dgram)
    if pending:
        result.append(bundle(pending))
    return result


def hmd_packet(pose: Pose, frame: Pose) -> bytes:
    """Invert v0.1's Rz(Roll) Ry(-Yaw) Rx(Pitch), with centimetre XYZ remapping."""
    p = compose(frame, to_openvr(pose))
    w, x, y, z = p.orientation
    # ZYX decomposition. At the Euler singularity choose Rx=0 and equivalent Rz.
    sy = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    ry = math.asin(sy)
    if abs(sy) > 1 - 1e-10:
        rx = 0.0
        rz = math.atan2(2 * (w * z - x * y), 1 - 2 * (x * x + z * z))
    else:
        rx = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        rz = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    px, py, pz = p.position
    return struct.pack(
        "<6d", px * 100, pz * 100, py * 100, -math.degrees(ry), math.degrees(rx), math.degrees(rz)
    )


class DeviceOutput:
    def __init__(self, config: LiveConfig, send: Callable[[bytes, int], None] | None = None):
        self.config = config
        self._socket = None
        if send is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(False)

            def send(data, port):
                return self._socket.sendto(data, ("127.0.0.1", port))

        self._send = send

    def validate_stop(self, policy: StopPolicy) -> None:
        if policy not in (StopPolicy.DISABLE_OWNED, StopPolicy.SAFE_POSE):
            raise ValueError("this shared VMT adapter never issues global Reset")

    def validate_target(self, target: ActuationTarget) -> None:
        # Revalidate even objects constructed via Pydantic's unchecked helpers.
        checked = ActuationTarget.model_validate_json(target.model_dump_json())
        extra = {p for p in EXTRA_PARTS if getattr(checked, p) is not None}
        if extra != {t.part for t in self.config.trackers}:
            raise ValueError("body targets must match owned tracker capabilities exactly")
        for name, limit in (("head", 0.5), ("left", 0.9), ("right", 0.9)):
            pose, rest = getattr(checked, name), getattr(self.config.safe_target, name)
            if name != "head":
                pose, rest = pose.pose, rest.pose
            limit = self.config.full_body_envelope or limit
            if math.dist(pose.position, rest.position) > limit:
                raise ValueError(f"{name} exceeds calibrated local motion envelope")
        for binding in self.config.trackers:
            if (
                math.dist(
                    getattr(checked, binding.part).position,
                    getattr(self.config.safe_target, binding.part).position,
                )
                > self.config.full_body_envelope
            ):
                raise ValueError(f"{binding.part} exceeds full-body envelope")

    def _attempt_all(self, packets):
        errors = []
        for data, port in packets:
            try:
                self._send(data, port)
            except OSError as exc:
                errors.append(str(exc))
        if errors:
            raise OSError(f"{len(errors)} UDP send attempts failed: {errors[0]}")

    def _hands(self, target, mode):
        for index, hand, enabled in (
            (self.config.left_index, target.left, 5),
            (self.config.right_index, target.right, 6),
        ):
            yield vmt_pose(index, enabled if mode else 0, hand.pose, self.config.vmt_from_stage)
            yield from vmt_controls(index, hand.controls)
        for binding in self.config.trackers:
            pose = compose(getattr(target, binding.part), binding.body_from_tracker)
            yield vmt_pose(binding.index, 7 if mode else 0, pose, self.config.vmt_from_stage)

    def send_target(self, target: ActuationTarget) -> None:
        self.validate_target(target)
        packets = [(x, self.config.vmt_port) for x in osc_packets(self._hands(target, True))]
        packets.append((hmd_packet(target.head, self.config.hmd_from_stage), self.config.hmd_port))
        self._attempt_all(packets)

    def neutralize(self) -> None:
        messages = [
            m
            for i in (self.config.left_index, self.config.right_index)
            for m in vmt_controls(i, Controls())
        ]
        self._attempt_all((x, self.config.vmt_port) for x in osc_packets(messages))

    def stop_pose(self, policy: StopPolicy) -> None:
        self.validate_stop(policy)
        # Also reset the separate HMD endpoint, even if a VMT send fails.
        packets = [
            (x, self.config.vmt_port)
            for x in osc_packets(
                self._hands(self.config.safe_target, policy == StopPolicy.SAFE_POSE)
            )
        ]
        packets.append(
            (
                hmd_packet(self.config.safe_target.head, self.config.hmd_from_stage),
                self.config.hmd_port,
            )
        )
        self._attempt_all(packets)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()


class MockOutput:
    def validate_stop(self, policy):
        pass

    def validate_target(self, target):
        ActuationTarget.model_validate_json(target.model_dump_json())

    def send_target(self, target):
        pass

    def neutralize(self):
        pass

    def stop_pose(self, policy):
        pass

    def close(self):
        pass
