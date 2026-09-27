"""Bounded joint trajectories for the existing servo; no inference or device I/O."""

import math

import numpy as np
from pydantic import Field, model_validator

from .actuation import PoseTarget
from .articulated_body import ArticulatedRig, JointState
from .body import BodyTarget, Frozen, HandTarget, Number, Quat, Vec3
from .motion_buffer import MotionBuffer, MotionKnot, blend_state
from .whole_body import vector


class JointSample(Frozen):
    time: Number
    root: Vec3
    rotations: tuple[Quat, ...] = Field(min_length=2, max_length=64)

    def state(self):
        return JointState(np.asarray(self.root), np.asarray(self.rotations))

    @classmethod
    def from_knot(cls, knot):
        return cls(
            time=knot.time,
            root=tuple(knot.state.root),
            rotations=tuple(tuple(q) for q in knot.state.rotations),
        )


class JointTrajectory(Frozen):
    epoch: int = Field(ge=0)
    rig: ArticulatedRig
    origin: JointSample
    measured: PoseTarget
    knots: tuple[JointSample, ...] = Field(min_length=1, max_length=12)
    filter_s: Number = Field(ge=0, le=0.2)
    floor: Number

    @model_validator(mode="after")
    def bounded(self):
        if self.knots[-1].time - self.knots[0].time > 0.51 or any(
            b.time <= a.time for a, b in zip(self.knots, self.knots[1:])
        ):
            raise ValueError("unordered or excessive servo horizon")
        for sample in (self.origin,) + self.knots:
            state = sample.state()
            if state.rotations.shape != (len(self.rig.names), 4) or not np.allclose(
                np.linalg.norm(state.rotations, axis=1), 1, atol=1e-6
            ):
                raise ValueError("invalid joint quaternion state")
            self.rig.validate_limits(state)
        if any(value is None for value in self.measured.model_dump().values()):
            raise ValueError("joint trajectory requires all eleven poses")
        return self


class JointServo:
    def __init__(self, command, now):
        self.command = command
        pose = command.measured
        measured = BodyTarget(
            **{k: v for k, v in pose if k not in ("left", "right")},
            left=HandTarget(pose=pose.left),
            right=HandTarget(pose=pose.right),
        )
        state = command.origin.state()
        self.buffer = MotionBuffer(command.rig, state, measured)
        self.stages, self.at = [state] * 3, now
        self.update(command)

    def update(self, command):
        if (
            command.epoch != self.command.epoch
            or command.rig != self.command.rig
            or command.origin != self.command.origin
            or command.measured != self.command.measured
            or command.filter_s != self.command.filter_s
            or command.floor != self.command.floor
        ):
            raise ValueError("servo epoch changed its immutable trajectory context")
        self.buffer.knots = [MotionKnot(k.time, k.state(), np.zeros(66)) for k in command.knots]

    def sample(self, now):
        state = self.buffer.sample(now).state
        if self.command.filter_s:
            alpha = -math.expm1(-min(0.05, max(0.0, now - self.at)) / self.command.filter_s)
            for i, old in enumerate(self.stages):
                state = blend_state(self.command.rig, old, state, alpha)
                self.stages[i] = state
        self.at = now
        pose = self.buffer.pose(state)
        if vector(pose)[:, 2].min() < self.command.floor:
            raise ValueError("servo trajectory crossed the declared tracking floor")
        return PoseTarget.from_target(pose)
