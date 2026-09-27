"""Finite joint-space trajectories. Interpolation never extrapolates motion."""

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from .articulated_body import JointState, inverse
from .body import qmul
from .joint_limits import quaternion, rotation_vector
from .motion import interpolate
from .whole_body import target_from_vector, vector


def blend_state(rig, a, b, fraction):
    # Apply the same convex rotation chart in one array operation. Per-joint
    # numpy calls here multiply the servo cost across all three filter stages.
    rotations = quaternion(
        (1 - fraction) * rotation_vector(a.rotations) + fraction * rotation_vector(b.rotations)
    )
    for i, (qa, qb) in enumerate(zip(a.rotations, b.rotations)):
        if not rig.joint_limits or rig.joint_limits[i] is None:
            rotations[i] = interpolate(qa, qb, fraction, quaternion=True)
    state = JointState((1 - fraction) * a.root + fraction * b.root, rotations)
    rig.validate_limits(state)
    return state


@dataclass(frozen=True)
class MotionKnot:
    time: float
    state: JointState
    rates: np.ndarray


class MotionBuffer:
    def __init__(self, rig, state, measured):
        self.rig = rig
        fitted, actual = vector(rig.forward(state)), vector(measured)
        self.position_residual = actual[:, :3] - fitted[:, :3]
        self.rotation_residual = [
            qmul(inverse(tuple(a)), tuple(b)) for a, b in zip(fitted[:, 3:], actual[:, 3:])
        ]
        self.knots = []

    def pose(self, state):
        values = vector(self.rig.forward(state))
        values[:, :3] += self.position_residual
        values[:, 3:] = [
            qmul(tuple(q), residual) for q, residual in zip(values[:, 3:], self.rotation_residual)
        ]
        return target_from_vector(values)

    def sample(self, now):
        if not self.knots:
            raise ValueError("empty motion buffer")
        i = bisect_right([k.time for k in self.knots], now)
        if i == 0:
            return self.knots[0]
        if i == len(self.knots):
            return self.knots[-1]
        a, b = self.knots[i - 1 : i + 1]
        t = (now - a.time) / (b.time - a.time)
        return MotionKnot(
            now, blend_state(self.rig, a.state, b.state, t), (1 - t) * a.rates + t * b.rates
        )

    def extend(self, knots):
        if len(knots) < 2 or any(b.time <= a.time for a, b in zip(knots, knots[1:])):
            raise ValueError("ordered motion horizon required")
        # The caller sampled the same buffer at this future splice time. Retain
        # the existing prefix and join at that shared joint state and velocity.
        self.knots = [k for k in self.knots if k.time < knots[0].time] + list(knots)
        self.knots = self.knots[-32:]
