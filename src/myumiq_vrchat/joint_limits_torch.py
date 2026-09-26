"""Differentiable implementation of the versioned local joint envelope."""

import numpy as np
import torch
from torch import nn

from .joint_limits import arrays


def rotation_vector(q):
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    norm = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    exact = 2 * torch.atan2(norm, q[..., :1]) / norm.clamp(min=1e-12)
    factor = torch.where(norm > 1e-7, exact, 2 / q[..., :1].clamp(min=0.5))
    return q[..., 1:] * factor


class JointEnvelope(nn.Module):
    def __init__(self, limits):
        super().__init__()
        self.enabled = bool(limits)
        if self.enabled:
            for name, value in zip(("lower", "upper", "radius"), arrays(limits)):
                # Finite unconstrained-root bounds avoid 0 * inf in backpropagation.
                self.register_buffer(
                    name,
                    torch.tensor(
                        np.nan_to_num(value, posinf=10.0, neginf=-10.0), dtype=torch.float64
                    ),
                )

    def forward(self, q):
        if not self.enabled:
            return q
        v = rotation_vector(q)
        clipped = torch.maximum(torch.minimum(v, self.upper), self.lower)
        norm = torch.linalg.vector_norm(clipped, dim=-1, keepdim=True)
        projected = clipped * (self.radius / norm.clamp(min=1e-12)).clamp(max=1.0)
        angle = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
        result = torch.cat(
            (torch.cos(angle / 2), projected * 0.5 * torch.sinc(angle / (2 * torch.pi))), dim=-1
        )
        changed = (projected - v).abs().amax(dim=-1, keepdim=True) > 1e-12
        return torch.where(changed, result, q)

    def violation(self, q):
        if not self.enabled:
            return torch.zeros_like(q[..., 0])
        v = rotation_vector(q)
        box = torch.maximum(self.lower - v, v - self.upper).clamp(min=0).amax(-1)
        radial = (torch.linalg.vector_norm(v, dim=-1) - self.radius[:, 0]).clamp(min=0)
        return torch.maximum(box, radial)
