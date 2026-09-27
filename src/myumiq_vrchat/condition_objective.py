"""Original end-state requirements for candidate-only body practice."""

from dataclasses import dataclass

import numpy as np
import torch

from .articulated_dynamics import pose_error
from .experience_refinement import ReferenceStart
from .whole_body import PARTS


@dataclass(frozen=True)
class ConditionStart(ReferenceStart):
    desired: np.ndarray
    condition_scale: np.ndarray


def condition_scale(resolved):
    scale = np.zeros((11, 6))
    for item in resolved.goal.conditions:
        index = PARTS.index(item.part)
        scale[index, :3] = resolved.position_mask[index] / item.position_tolerance
        scale[index, 3:] = float(resolved.orientation_mask[index]) / item.angular_tolerance
    return scale


def condition_distance(pose, desired, scale):
    """Worst original condition ratio; masks do not constrain unrequested axes."""
    errors = pose_error(pose, desired) * scale
    return errors.reshape(len(pose), 22, 3).norm(dim=-1).amax(1)


class ConditionObjective:
    """Soft objectives only; hard admission and live limits remain independent."""

    def __init__(self, cases, *, weight=0.1, smoothness=0.04):
        if not 0 < weight <= 1 or not 0 <= smoothness <= 1:
            raise ValueError("invalid condition practice weights")
        self.weight, self.smoothness = weight, smoothness
        self.desired = torch.tensor(
            np.stack([getattr(c, "desired", c.goal) for c in cases]), dtype=torch.float32
        )
        self.scale = torch.tensor(
            np.stack([getattr(c, "condition_scale", np.zeros((11, 6))) for c in cases]),
            dtype=torch.float32,
        )

    def __call__(self, indices, before, after, rates, previous, dt):
        desired, scale = self.desired[indices], self.scale[indices]
        old = condition_distance(before, desired, scale)
        new = condition_distance(after, desired, scale)
        # Continue towards the inner tolerance instead of rewarding only a loose
        # dense-pose endpoint. Smoothness cannot substitute for task achievement.
        error = torch.relu(new - 0.5)
        return self.weight * (10 * (old - new) - error) - self.smoothness * (
            (rates - previous) * (0.05 / dt)
        ).square().mean(1)
