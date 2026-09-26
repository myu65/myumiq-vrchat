"""Learned whole-body rate synergies; zero action holds any current pose."""

import hashlib
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .body import Frozen, Number


class RateBasis(Frozen):
    format_version: Literal[1] = 1
    action_size: int = Field(default=66, ge=3, le=192)
    rows: tuple[tuple[Number, ...], ...] = Field(min_length=1, max_length=66)
    variance_fraction: Number = Field(ge=0, le=1)

    @model_validator(mode="after")
    def dimensions(self):
        if self.action_size % 3 or any(len(row) != self.action_size for row in self.rows):
            raise ValueError("rate basis must map to complete xyz rate groups")
        matrix = np.asarray(self.rows)
        if not np.allclose(matrix @ matrix.T, np.eye(len(matrix)), atol=1e-5):
            raise ValueError("rate basis rows must be orthonormal")
        return self

    @property
    def identity(self):
        return hashlib.sha256(np.asarray(self.rows, dtype="<f8").tobytes()).hexdigest()

    @classmethod
    def fit(cls, rates, components):
        samples = np.asarray(rates, dtype=np.float64)
        if (
            samples.ndim != 2
            or not 3 <= samples.shape[1] <= 192
            or samples.shape[1] % 3
            or not np.isfinite(samples).all()
            or np.abs(samples).max() > 1
            or not 1 <= components <= min(samples.shape)
        ):
            raise ValueError("invalid rate examples or component count")
        _, singular, axes = np.linalg.svd(samples, full_matrices=False)
        total = float(np.sum(singular**2))
        if total < 1e-12:
            raise ValueError("motion basis requires nonzero examples")
        return cls(
            action_size=samples.shape[1],
            rows=tuple(tuple(float(x) for x in row) for row in axes[:components]),
            variance_fraction=min(1.0, float(np.sum(singular[:components] ** 2)) / total),
        )

    def decode(self, latent):
        values = np.asarray(latent, dtype=np.float64)
        if (
            values.shape != (len(self.rows),)
            or not np.isfinite(values).all()
            or np.abs(values).max() > 1
        ):
            raise ValueError("invalid latent whole-body action")
        rates = values @ np.asarray(self.rows)
        rows = rates.reshape(-1, 3)
        # A shared scale retains coupling instead of clipping each tracker separately.
        scale = max(
            1.0,
            float(np.linalg.norm(rows, axis=1).max()),
        )
        return np.clip(rates / scale, -1.0, 1.0)


def torch_decoder(basis):
    import torch

    class Decode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("basis", torch.tensor(basis.rows, dtype=torch.float32))

        def forward(self, latent):
            rates = latent @ self.basis
            rows = rates.reshape(-1, basis.action_size // 3, 3)
            scale = torch.linalg.vector_norm(rows, dim=2).amax(dim=1).clamp(min=1.0)
            return (rates / scale[:, None]).clamp(-1.0, 1.0)

    return Decode()
