"""Configurable local joint envelopes; no gravity, neutral-pose controller or UI axes."""

import numpy as np
from pydantic import model_validator

from .body import Frozen, Number

CONTRACT = "parent_rest_rotation_vector_box_ball_v1"


def rotation_vector(q):
    q = np.asarray(q)
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    length = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    return q[..., 1:] * (2 * np.arctan2(length, q[..., :1]) / np.maximum(length, 1e-12))


def quaternion(v):
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.concatenate((np.cos(angle / 2), v * 0.5 * np.sinc(angle / (2 * np.pi))), axis=-1)


class JointLimit(Frozen):
    lower: tuple[Number, Number, Number]
    upper: tuple[Number, Number, Number]
    max_angle: Number

    @model_validator(mode="after")
    def bounds(self):
        if not 0 < self.max_angle < np.pi - 0.001 or any(
            not -np.pi < lo <= 0 <= hi < np.pi or lo >= hi for lo, hi in zip(self.lower, self.upper)
        ):
            raise ValueError(
                "joint envelope must contain rest and use an unambiguous rotation chart"
            )
        return self


def arrays(limits):
    # Root is intentionally unconstrained, including pitch and roll.
    lower = np.array([x.lower if x else (-np.inf,) * 3 for x in limits])
    upper = np.array([x.upper if x else (np.inf,) * 3 for x in limits])
    radius = np.array([x.max_angle if x else np.inf for x in limits])[:, None]
    return lower, upper, radius


def project_vectors(v, lower, upper, radius):
    clipped = np.clip(v, lower, upper)
    return clipped * np.minimum(
        1.0, radius / np.maximum(np.linalg.norm(clipped, axis=-1, keepdims=True), 1e-12)
    )


def project(q, limits):
    if not limits:
        return q
    v = rotation_vector(q)
    constrained = project_vectors(v, *arrays(limits))
    changed = np.any(np.abs(constrained - v) > 1e-12, axis=-1, keepdims=True)
    # Preserve quaternion sign and exact holds when no restriction is active.
    return np.where(changed, quaternion(constrained), q)


def violation(q, limits):
    """Largest chart-coordinate/radius excess in radians, not an anatomical score."""
    if not limits:
        return np.zeros(np.asarray(q).shape[:-1])
    lower, upper, radius = arrays(limits)
    v = rotation_vector(q)
    box = np.maximum(np.maximum(lower - v, v - upper), 0.0).max(axis=-1)
    return np.maximum(box, np.maximum(np.linalg.norm(v, axis=-1) - radius[:, 0], 0.0))


def from_reference(rotations, margin=0.1):
    """Estimate motion support from caller-selected reference samples, never held-out data."""
    q = np.asarray(rotations)
    if (
        q.ndim != 3
        or q.shape[0] < 2
        or q.shape[-1] != 4
        or not np.isfinite(q).all()
        or not np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-6)
        or not np.isfinite(margin)
        or not 0 < margin <= 0.35
    ):
        raise ValueError("invalid joint-envelope reference samples or margin")
    v = rotation_vector(q)
    lower = np.minimum(v.min(axis=0), 0.0) - margin
    upper = np.maximum(v.max(axis=0), 0.0) + margin
    radius = np.linalg.norm(v, axis=-1).max(axis=0) + margin
    return (None,) + tuple(
        JointLimit(lower=tuple(lo), upper=tuple(hi), max_angle=float(r))
        for lo, hi, r in zip(lower[1:], upper[1:], radius[1:])
    )
