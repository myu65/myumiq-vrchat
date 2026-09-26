"""Persisted learned motor residual shared by training and inference."""

from pydantic import Field

from .body import Frozen, Number, Pose


class ResidualPolicy(Frozen):
    schema_version: int = 1
    skill: str = Field(pattern="^(WAVE|REACH)$")
    hand: str = Field(pattern="^(left|right)$")
    correction: tuple[Number, Number, Number]
    training_samples: int = Field(gt=0)
    source: str = "openvr_raw"

    def apply(self, pose: Pose) -> Pose:
        return pose.model_copy(
            update={"position": tuple(x + d for x, d in zip(pose.position, self.correction))}
        )
