"""Small periodic imitation policy. Feedback bounds motion; it is not physics RL.

Learned coefficients map phase to eleven poses. BodyState supplies the starting
pose and the per-step feedback guard. A single whole-body task is supported;
multi-task arbitration and contact-aware physics belong to a later policy.
"""

import math
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from .body import (
    BodyGoal,
    BodyState,
    BodyTarget,
    Frozen,
    HandTarget,
    Number,
    Pose,
)
from .motion import QUATERNIUS, interpolate

PARTS = tuple(QUATERNIUS)


def vector(target: BodyTarget) -> np.ndarray:
    if not target.is_full_body:
        raise ValueError("whole-body policy requires all eleven poses")
    return np.array(
        [(*target.pose_for(p).position, *target.pose_for(p).orientation) for p in PARTS],
        dtype=np.float64,
    )


def target_from_vector(values: np.ndarray) -> BodyTarget:
    values = np.asarray(values).reshape(11, 7)
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite policy output")
    poses = {}
    for part, row in zip(PARTS, values):
        norm = np.linalg.norm(row[3:])
        if norm < 0.25:
            raise ValueError("degenerate learned rotation")
        poses[part] = Pose(position=tuple(row[:3]), orientation=tuple(row[3:] / norm))
    return BodyTarget(
        head=poses.pop("head"),
        left=HandTarget(pose=poses.pop("left_hand")),
        right=HandTarget(pose=poses.pop("right_hand")),
        **poses,
    )


def features(phase, harmonics):
    phase = np.asarray(phase).reshape(-1, 1)
    angles = phase * (2 * math.pi * np.arange(1, harmonics + 1))
    return np.concatenate((np.ones_like(phase), np.sin(angles), np.cos(angles)), axis=1)


class PeriodicImitation(Frozen):
    format_version: int = Field(default=1, ge=1, le=1)
    clip: str = Field(min_length=1, max_length=80)
    duration_s: Number = Field(gt=0, le=60)
    harmonics: int = Field(ge=1, le=20)
    coefficients: tuple[tuple[Number, ...], ...]

    @model_validator(mode="after")
    def dimensions(self):
        if len(self.coefficients) != 1 + 2 * self.harmonics or any(
            len(row) != 77 for row in self.coefficients
        ):
            raise ValueError("invalid imitation checkpoint dimensions")
        return self

    @classmethod
    def fit(cls, frames, duration_s, clip, harmonics=6, ridge=1e-5):
        if not frames or ridge <= 0 or not math.isfinite(ridge):
            raise ValueError("finite regularization and training samples required")
        targets = np.stack([vector(t) for _, t in frames])
        # Quaternion sign continuity before regression (q and -q are equivalent).
        for i in range(1, len(targets)):
            flips = np.sum(targets[i - 1, :, 3:] * targets[i, :, 3:], axis=1) < 0
            targets[i, flips, 3:] *= -1
        x = features([t / duration_s for t, _ in frames], harmonics)
        weights = np.linalg.solve(
            x.T @ x + ridge * np.eye(x.shape[1]), x.T @ targets.reshape(len(targets), -1)
        )
        return cls(
            clip=clip,
            duration_s=float(duration_s),
            harmonics=harmonics,
            coefficients=tuple(tuple(float(v) for v in row) for row in weights),
        )

    def sample(self, phase: float) -> BodyTarget:
        if not math.isfinite(phase):
            raise ValueError("phase must be finite")
        weights = np.asarray(self.coefficients)
        if weights.shape != (1 + 2 * self.harmonics, 77):
            raise ValueError("invalid imitation checkpoint dimensions")
        return target_from_vector(features([phase % 1], self.harmonics) @ weights)

    def save(self, path: Path):
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def state_target(state: BodyState) -> BodyTarget:
    signals = [state.signal_for(part) for part in PARTS]
    if any(not s.valid or not s.connected or s.pose is None for s in signals):
        raise ValueError("whole-body policy requires complete valid body feedback")
    return target_from_vector(np.array([(*s.pose.position, *s.pose.orientation) for s in signals]))


def bounded_step(
    current: BodyTarget, desired: BodyTarget, dt: float, speed=0.6, angular_speed=2.0
) -> BodyTarget:
    if not 0 < dt <= 0.1 or not 0 < speed <= 2 or not 0 < angular_speed <= 6:
        raise ValueError("invalid whole-body timestep or speed limit")
    a, b = vector(current), vector(desired)
    distance = np.linalg.norm(a[:, :3] - b[:, :3], axis=1).max()
    dots = np.abs(np.sum(a[:, 3:] * b[:, 3:], axis=1))
    angle = (2 * np.arccos(np.clip(dots, 0, 1))).max()
    fraction = min(1.0, speed * dt / max(distance, 1e-10), angular_speed * dt / max(angle, 1e-10))
    value = np.empty_like(a)
    value[:, :3] = a[:, :3] + fraction * (b[:, :3] - a[:, :3])
    value[:, 3:] = np.stack([interpolate(x, y, fraction, True) for x, y in zip(a[:, 3:], b[:, 3:])])
    return target_from_vector(value)


class WholeBodyPolicy:
    def __init__(self, model: PeriodicImitation, playback_rate=0.25):
        if not 0 < playback_rate <= 1:
            raise ValueError("playback rate must be in (0, 1]")
        self.model = model
        self.playback_rate = playback_rate
        self.elapsed = 0.0

    def step(self, body: BodyState, goal: BodyGoal, dt: float) -> BodyTarget:
        if (
            len(goal.tasks) != 1
            or goal.conditions
            or goal.constraints
            or set(goal.tasks[0].effectors) != set(PARTS)
            or goal.tasks[0].kind != "locomotion"
            or goal.tasks[0].target != self.model.clip
        ):
            raise ValueError("this policy supports only its single unconstrained whole-body clip")
        if self.elapsed + dt > goal.duration_s + 1e-9:
            raise ValueError("body goal expired")
        current = state_target(body)
        desired = self.model.sample(self.elapsed * self.playback_rate / self.model.duration_s)
        result = bounded_step(current, desired, dt)
        self.elapsed += dt
        return result


def floor_sitting_reward(body: BodyState, contacts: dict[str, bool] | None = None) -> dict:
    """Reference-free geometry component for a future simulator SAC environment.

    Contacts must come from the simulator or another explicit observer. Missing
    contacts prevent success; tracker proximity never invents contact evidence.
    This helper does not provide dynamics, a trainer, or a success claim in VRChat.
    """
    pose = state_target(body)
    pelvis, lk, rk = (pose.pose_for(p).position for p in ("pelvis", "left_knee", "right_knee"))
    feet = [pose.pose_for(p).position for p in ("left_foot", "right_foot")]
    error = abs(pelvis[2] - 0.15) + sum(abs(f[2] - 0.08) for f in feet)
    error += sum(max(0.0, pelvis[2] + 0.2 - k[2]) for k in (lk, rk))
    # Bent knees forward and feet near the pelvis (体育座り geometry).
    error += sum(max(0.0, 0.15 - (k[0] - pelvis[0])) for k in (lk, rk))
    error += sum(max(0.0, math.dist(f, pelvis) - 0.6) for f in feet)
    supported = contacts is not None and all(
        contacts.get(p, False) for p in ("pelvis", "left_foot", "right_foot")
    )
    return {
        "geometry_reward": math.exp(-5 * error),
        "contacts_observed": contacts is not None,
        "success": error < 0.15 and supported,
    }
