"""Whole-body pose-rate action adapter; no gravity or automatic neutral pose."""

import math

import numpy as np

from .body import BodyTarget, qmul
from .whole_body import PARTS, target_from_vector, vector

CONTRACT = "myumiq-tracker-rates-v1"
ACTION_SHAPE = (len(PARTS), 6)


def integrate_tracker_action(current: BodyTarget, action, dt: float) -> BodyTarget:
    """Apply one coordinated normalized action, in tracking-space axes.

    Each row contains xyz linear rate then xyz angular rate. Controller inputs
    are intentionally not integrated here: they are separate device commands,
    and their effect on world displacement requires an external observation.
    """
    values = np.asarray(action, dtype=np.float64)
    if values.shape != ACTION_SHAPE or not np.isfinite(values).all() or np.any(np.abs(values) > 1):
        raise ValueError("invalid coordinated tracker action")
    if not math.isfinite(dt) or not 0 < dt <= 0.1:
        raise ValueError("invalid tracker action timestep")
    poses = vector(current)
    for index, rate in enumerate(values):
        linear = rate[:3] * 0.6
        angular = rate[3:] * 2.0
        linear /= max(1.0, float(np.linalg.norm(linear)) / 0.6)
        angular /= max(1.0, float(np.linalg.norm(angular)) / 2.0)
        poses[index, :3] += linear * dt
        angle = float(np.linalg.norm(angular)) * dt
        if angle:
            axis = angular / np.linalg.norm(angular)
            rotation = (math.cos(angle / 2), *(axis * math.sin(angle / 2)))
            poses[index, 3:] = qmul(rotation, tuple(poses[index, 3:]))
    # Neutral input controls are emitted by default, never inherited as a lease.
    return target_from_vector(poses)
