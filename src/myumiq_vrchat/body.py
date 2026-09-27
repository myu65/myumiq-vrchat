"""Immutable canonical state and targets. RH metres: forward X, left Y, up Z."""

import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

Number = Annotated[float, Field(allow_inf_nan=False)]
Unit = Annotated[Number, Field(ge=0, le=1)]
SignedUnit = Annotated[Number, Field(ge=-1, le=1)]
Vec3 = tuple[Number, Number, Number]
Quat = tuple[Number, Number, Number, Number]  # wxyz, active rotation
Stick = tuple[SignedUnit, SignedUnit]
BodyPart = Literal[
    "head",
    "chest",
    "pelvis",
    "left_hand",
    "right_hand",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
]
EXTRA_PARTS = (
    "chest",
    "pelvis",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, validate_default=True)


class Pose(Frozen):
    position: Vec3
    orientation: Quat = (1.0, 0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def normalized(self) -> Self:
        if abs(sum(v * v for v in self.orientation) - 1) > 1e-5:
            raise ValueError("orientation must be a normalized wxyz quaternion")
        return self


class Controls(Frozen):
    """Complete VMT profile inventory. Curl 0=open, 1=fist; never sparse deltas."""

    buttons: tuple[bool, ...] = Field(default=(False,) * 18, min_length=18, max_length=18)
    button_touches: tuple[bool, ...] = Field(default=(False,) * 18, min_length=18, max_length=18)
    triggers: tuple[Unit, ...] = Field(default=(0.0,) * 9, min_length=9, max_length=9)
    trigger_touches: tuple[bool, ...] = Field(default=(False,) * 9, min_length=9, max_length=9)
    trigger_clicks: tuple[bool, ...] = Field(default=(False,) * 9, min_length=9, max_length=9)
    sticks: tuple[Stick, ...] = Field(default=((0.0, 0.0),) * 4, min_length=4, max_length=4)
    stick_touches: tuple[bool, ...] = Field(default=(False,) * 4, min_length=4, max_length=4)
    stick_clicks: tuple[bool, ...] = Field(default=(False,) * 4, min_length=4, max_length=4)
    curls: tuple[Unit, Unit, Unit, Unit, Unit] = (0.0,) * 5


class HandTarget(Frozen):
    pose: Pose
    controls: Controls = Controls()


class BodyTarget(Frozen):
    head: Pose
    left: HandTarget
    right: HandTarget
    chest: Pose | None = None
    pelvis: Pose | None = None
    left_elbow: Pose | None = None
    right_elbow: Pose | None = None
    left_knee: Pose | None = None
    right_knee: Pose | None = None
    left_foot: Pose | None = None
    right_foot: Pose | None = None

    def pose_for(self, part: BodyPart) -> Pose | None:
        if part in ("left_hand", "right_hand"):
            return getattr(self, "left" if part == "left_hand" else "right").pose
        return getattr(self, part)

    @property
    def is_full_body(self) -> bool:
        return all(getattr(self, part) is not None for part in EXTRA_PARTS)


# Public compatibility name for existing replay and three-point callers.
ActuationTarget = BodyTarget


def rest_target() -> ActuationTarget:
    return ActuationTarget(
        head=Pose(position=(0.0, 0.0, 1.6)),
        left=HandTarget(pose=Pose(position=(0.2, 0.25, 1.15))),
        right=HandTarget(pose=Pose(position=(0.2, -0.25, 1.15))),
    )


class PoseSignal(Frozen):
    pose: Pose | None = None
    valid: bool = False
    connected: bool = False
    tracking_result: int | None = None
    confidence: Unit = 0.0
    source: Literal["unavailable", "simulated", "openvr_raw"] = "unavailable"
    timestamp: Number


class BodyState(Frozen):
    head: PoseSignal
    left: PoseSignal
    right: PoseSignal
    chest: PoseSignal = PoseSignal(timestamp=0.0)
    pelvis: PoseSignal = PoseSignal(timestamp=0.0)
    left_elbow: PoseSignal = PoseSignal(timestamp=0.0)
    right_elbow: PoseSignal = PoseSignal(timestamp=0.0)
    left_knee: PoseSignal = PoseSignal(timestamp=0.0)
    right_knee: PoseSignal = PoseSignal(timestamp=0.0)
    left_foot: PoseSignal = PoseSignal(timestamp=0.0)
    right_foot: PoseSignal = PoseSignal(timestamp=0.0)
    # No avatar/root/contact ground truth is inferred from device poses.

    def signal_for(self, part: BodyPart) -> PoseSignal:
        return getattr(self, {"left_hand": "left", "right_hand": "right"}.get(part, part))


class BodyTask(Frozen):
    id: str = Field(min_length=1, max_length=80)
    kind: Literal["posture", "locomotion", "gaze", "reach", "gesture", "hold"]
    effectors: tuple[BodyPart, ...] = Field(min_length=1, max_length=11)
    target: str | None = Field(default=None, max_length=80)
    posture: Literal["standing", "crouching", "sitting_floor", "lying", "standing_up"] | None = None
    priority: int = Field(default=0, ge=0, le=100)


class BodyConstraint(Frozen):
    part: BodyPart
    kind: Literal["fixed_pose", "contact"]
    pose: Pose | None = None
    surface: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def arguments(self) -> Self:
        if self.kind == "fixed_pose" and (self.pose is None or self.surface is not None):
            raise ValueError("fixed pose requires pose only")
        if self.kind == "contact" and (not self.surface or self.pose is not None):
            raise ValueError("contact requires a surface reference only")
        return self


class BodyCondition(Frozen):
    """End-state requirements; missing components are free, never zero targets.

    All offsets use tracking axes. ``current`` binds to the observed part at
    acceptance; ``target`` adds a calibrated object's metric position. A supplied
    target-frame orientation is absolute, because WorldObject has no orientation.
    """

    part: BodyPart
    frame: Literal["tracking", "current", "target"] = "tracking"
    target: str | None = Field(default=None, min_length=1, max_length=80)
    position: tuple[Number | None, Number | None, Number | None] | None = None
    orientation: Quat | None = None
    position_tolerance: Number = Field(default=0.03, ge=0.005, le=0.12)
    angular_tolerance: Number = Field(default=0.15, ge=0.02, le=0.35)

    @model_validator(mode="after")
    def arguments(self):
        if (self.target is not None) != (self.frame == "target"):
            raise ValueError("target frame requires exactly one target identity")
        if self.position is not None and (
            all(v is None for v in self.position)
            or any(v is not None and abs(v) > 3 for v in self.position)
        ):
            raise ValueError("condition positions require bounded selected coordinates")
        if self.position is None and self.orientation is None:
            raise ValueError("empty body condition")
        if self.frame == "target" and self.position is None:
            raise ValueError("metric target binding requires a position condition")
        if self.orientation is not None:
            Pose(position=(0.0, 0.0, 0.0), orientation=self.orientation)
        return self


class BodyGoal(Frozen):
    tasks: tuple[BodyTask, ...] = Field(default=(), max_length=11)
    constraints: tuple[BodyConstraint, ...] = Field(default=(), max_length=22)
    conditions: tuple[BodyCondition, ...] = Field(default=(), max_length=11)
    duration_s: Number = Field(gt=0, le=60)

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        result = handler(self)
        if not self.conditions:
            result.pop("conditions", None)  # Preserve existing replay identities.
        return result

    @model_validator(mode="after")
    def ownership(self) -> Self:
        if self.conditions:
            if self.tasks or self.constraints:
                raise ValueError("conditions and legacy task ownership cannot be mixed")
            if len({c.part for c in self.conditions}) != len(self.conditions):
                raise ValueError("combine conditions for a body part into one entry")
        elif not self.tasks:
            raise ValueError("body goal requires tasks or conditions")
        ids, owned = set(), set()
        for task in self.tasks:
            if task.id in ids or len(set(task.effectors)) != len(task.effectors):
                raise ValueError("duplicate task or effector")
            if owned.intersection(task.effectors):
                raise ValueError("concurrent tasks conflict over body effectors")
            ids.add(task.id)
            owned.update(task.effectors)
        fixed = {c.part for c in self.constraints if c.kind == "fixed_pose"}
        for task in self.tasks:
            if task.kind != "hold" and fixed.intersection(task.effectors):
                raise ValueError("moving task conflicts with a fixed body constraint")
        if len({(c.part, c.kind) for c in self.constraints}) != len(self.constraints):
            raise ValueError("duplicate body constraint")
        return self


class WorldObject(Frozen):
    name: str = Field(min_length=1, max_length=80)
    position: Vec3
    source: Literal["fixture", "manual", "vision"]
    kind: Literal["player", "object", "unknown"] = "unknown"
    confidence: Unit = 1.0
    last_seen: Number | None = None
    velocity: Vec3 = (0.0, 0.0, 0.0)
    # Normalized image coordinates: right/down positive, image centre (0, 0).
    # Kept separate from the detector's approximate metric position.
    image_position: tuple[SignedUnit, SignedUnit] | None = None


class WorldState(Frozen):
    objects: tuple[WorldObject, ...] = ()
    timestamp: Number | None = None

    def locate(self, name: str) -> Vec3:
        matches = [x.position for x in self.objects if x.name == name]
        if len(matches) != 1:
            raise ValueError(f"target must identify exactly one known object: {name}")
        return matches[0]


def simulated_body(target: ActuationTarget, timestamp: float) -> BodyState:
    def signal(pose):
        return PoseSignal(
            pose=pose,
            valid=True,
            connected=True,
            confidence=1.0,
            source="simulated",
            timestamp=timestamp,
        )

    return BodyState(
        head=signal(target.head),
        left=signal(target.left.pose),
        right=signal(target.right.pose),
        **{
            part: signal(getattr(target, part))
            if getattr(target, part) is not None
            else PoseSignal(timestamp=timestamp)
            for part in EXTRA_PARTS
        },
    )


def qmul(a: Quat, b: Quat) -> Quat:
    w, x, y, z = a
    v, i, j, k = b
    return (
        w * v - x * i - y * j - z * k,
        w * i + x * v + y * k - z * j,
        w * j - x * k + y * v + z * i,
        w * k + x * j - y * i + z * v,
    )


def rotate(q: Quat, v: Vec3) -> Vec3:
    return qmul(qmul(q, (0.0, *v)), (q[0], -q[1], -q[2], -q[3]))[1:]


def compose(frame: Pose, pose: Pose) -> Pose:
    v = rotate(frame.orientation, pose.position)
    return Pose(
        position=tuple(a + b for a, b in zip(frame.position, v)),
        orientation=qmul(frame.orientation, pose.orientation),
    )


def inverse(frame: Pose) -> Pose:
    q = (frame.orientation[0], *(-v for v in frame.orientation[1:]))
    return Pose(position=rotate(q, tuple(-v for v in frame.position)), orientation=q)


def q_from_matrix(m) -> Quat:
    """Convert a proper 3x3 rotation, including 180-degree rotations."""
    trace = sum(m[i][i] for i in range(3))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = (s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    else:
        i = max(range(3), key=lambda i: m[i][i])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + m[i][i] - m[j][j] - m[k][k]) * 2
        xyz = [0.0, 0.0, 0.0]
        xyz[i], xyz[j], xyz[k] = s / 4, (m[j][i] + m[i][j]) / s, (m[k][i] + m[i][k]) / s
        q = ((m[k][j] - m[j][k]) / s, *xyz)
    norm = math.sqrt(sum(v * v for v in q))
    return tuple(v / norm for v in q)
