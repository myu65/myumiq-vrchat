"""Measured trajectory diagnostics; never a contact or human-naturalness oracle."""

import numpy as np


def trajectory_quality(poses, dt):
    """Uniformly sampled positions in metres, including the final held frame."""
    points = np.asarray(poses, dtype=float)
    if (
        points.ndim != 3
        or points.shape[1:] != (11, 3)
        or not len(points)
        or not np.isfinite(points).all()
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("finite uniformly sampled eleven-tracker positions required")
    result = {"scope": "tracker_kinematic_proxy_not_contact_or_avatar_naturalness"}
    derivative = points
    for name in ("speed_m_s", "acceleration_m_s2", "jerk_m_s3"):
        derivative = np.diff(derivative, axis=0) / dt
        norms = np.linalg.norm(derivative, axis=-1)
        result["maximum_" + name] = float(norms.max()) if norms.size else 0.0
        result["rms_" + name] = float(np.sqrt(np.mean(norms**2))) if norms.size else 0.0
    result["maximum_foot_displacement_m"] = float(
        np.linalg.norm(points[:, 9:, :] - points[0:1, 9:, :], axis=-1).max()
    )
    result["minimum_tracker_height_m"] = float(points[..., 2].min())
    return result
