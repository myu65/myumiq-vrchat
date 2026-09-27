"""Learned finite motions and feedback-paced references for one whole-body actor."""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .body import Frozen, Number, qmul, rotate
from .tracker_policy import pose_error
from .whole_body import PeriodicImitation, target_from_vector, vector


def basis(phase, count):
    x = np.asarray(phase).reshape(-1, 1)
    return np.exp(-0.5 * ((x - np.linspace(0, 1, count)) * (count - 1)) ** 2)


class FiniteImitation(Frozen):
    format_version: Literal[2] = 2
    clip: str = Field(min_length=1, max_length=80)
    duration_s: Number = Field(gt=0, le=60)
    coefficients: tuple[tuple[Number, ...], ...]

    @model_validator(mode="after")
    def dimensions(self):
        if not 4 <= len(self.coefficients) <= 128 or any(
            len(row) != 77 for row in self.coefficients
        ):
            raise ValueError("invalid finite imitation checkpoint dimensions")
        return self

    @classmethod
    def fit(cls, frames, duration_s, clip, count=48, ridge=1e-6):
        if not frames or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("finite regularization and samples required")
        targets = np.stack([vector(t) for _, t in frames])
        for i in range(1, len(targets)):
            flips = np.sum(targets[i - 1, :, 3:] * targets[i, :, 3:], axis=1) < 0
            targets[i, flips, 3:] *= -1
        x = basis([t / duration_s for t, _ in frames], count)
        weights = np.linalg.solve(
            x.T @ x + ridge * np.eye(count), x.T @ targets.reshape(len(targets), -1)
        )
        return cls(
            clip=clip,
            duration_s=duration_s,
            coefficients=tuple(tuple(float(v) for v in row) for row in weights),
        )

    def sample(self, phase):
        if not math.isfinite(phase):
            raise ValueError("phase must be finite")
        return target_from_vector(
            basis([np.clip(phase, 0, 1)], len(self.coefficients)) @ np.asarray(self.coefficients)
        )

    def save(self, path):
        Path(path).write_text(self.model_dump_json(indent=2), "utf-8")


def load_motion(content):
    data = json.loads(content)
    if data.get("format_version") == 2:
        return FiniteImitation.model_validate_json(content)
    return PeriodicImitation.model_validate_json(content)


class MotionReference(Frozen):
    policy: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    playback_rate: Number = Field(default=0.25, gt=0, le=1)
    execution_timeout_s: Number | None = Field(default=None, ge=1, le=20)
    hand: Literal["left", "right"] | None = None
    description: str = Field(default="", max_length=80)

    def load(self, floor):
        from .cli import outside_repo

        content = outside_repo(self.policy).read_bytes()
        if hashlib.sha256(content).hexdigest() != self.sha256:
            raise ValueError("motion checkpoint changed since evaluation")
        model = load_motion(content)
        validate_motion(model, floor)
        return model


def validate_motion(model, floor):
    for phase in np.linspace(0, 1, 241):
        points = vector(model.sample(float(phase)))
        if (
            np.abs(points[:, :2]).max() > 1.5
            or points[:, 2].max() > 2.3
            or points[:, 2].min() < -0.2
            or points[:, 2].min() < floor
        ):
            raise ValueError("learned motion exceeds its workspace or declared floor")


def heading(pose):
    forward = rotate(pose.orientation, (1.0, 0.0, 0.0))
    if math.hypot(*forward[:2]) < 0.1:
        # Bending or lying down can point the pelvis forward axis vertically.
        # Its orthogonal lateral axis still defines a horizontal reference; this
        # is a task-frame convention, not an upright-pose requirement.
        lateral = rotate(pose.orientation, (0.0, 1.0, 0.0))
        return math.atan2(-lateral[0], lateral[1])
    return math.atan2(forward[1], forward[0])


class MotionPlayback:
    """References only: no motor commands, pose resets or controller locomotion."""

    def __init__(self, model, current, rate, *, origin_xy=None):
        self.model, self.rate = model, rate
        origin = model.sample(0)
        angle = heading(current.pelvis) - heading(origin.pelvis)
        self.rotation = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
        pivot = rotate(self.rotation, origin.pelvis.position)
        xy = current.pelvis.position[:2] if origin_xy is None else origin_xy
        if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            raise ValueError("motion origin requires finite tracking XY")
        self.offset = (xy[0] - pivot[0], xy[1] - pivot[1], 0.0)
        self.phase = 0.0
        self.covered = set()
        self.completed = False
        self.last_observation = None
        self.first_confirmed_pose = None
        self.movement = 0.0

    def target(self, phase=None):
        points = vector(self.model.sample(self.phase if phase is None else phase))
        for row in points:
            row[:3] = np.asarray(rotate(self.rotation, tuple(row[:3]))) + self.offset
            row[3:] = qmul(self.rotation, tuple(row[3:]))
        return target_from_vector(points)

    def observe(self, current, timestamp, dt, *, speed_scale=1.0):
        if not math.isfinite(speed_scale) or not 0 <= speed_scale <= 1:
            raise ValueError("gait speed scale must be within 0..1")
        # Called only after the controller has confirmed the last issued action.
        if self.last_observation is not None and timestamp <= self.last_observation:
            return
        elapsed = dt if self.last_observation is None else timestamp - self.last_observation
        self.last_observation = timestamp
        error = pose_error(current, self.target())
        if (
            np.linalg.norm(error[:, :3], axis=1).max() > 0.12
            or np.linalg.norm(error[:, 3:], axis=1).max() > 0.35
        ):
            return
        self.covered.add(min(15, int(min(self.phase, 1.0) * 16)))
        points = vector(current)[:, :3]
        if self.first_confirmed_pose is None:
            self.first_confirmed_pose = points.copy()
        self.movement = max(
            self.movement, float(np.linalg.norm(points - self.first_confirmed_pose, axis=1).max())
        )
        if self.phase >= 1.0:
            self.completed = len(self.covered) == 16 and self.movement > 0.015
        # A finite trajectory stays at its learned end; periodic samples wrap.
        increment = min(0.1, max(0.0, elapsed)) * self.rate * speed_scale / self.model.duration_s
        increment = min(increment, 1 / 32)
        self.phase = (
            self.phase + increment
            if isinstance(self.model, PeriodicImitation)
            else min(1.0, self.phase + increment)
        )

    def evidence(self):
        return {
            "clip": self.model.clip,
            "phase": self.phase,
            "phase_bins_observed": len(self.covered),
            "phase_bins_required": 16,
            "sequence_observed": self.completed,
            "observed_motion_m": self.movement,
            "endpoint_required": not isinstance(self.model, PeriodicImitation),
            "reference_kind": "periodic" if isinstance(self.model, PeriodicImitation) else "finite",
            "world_displacement_observed": False,
        }
