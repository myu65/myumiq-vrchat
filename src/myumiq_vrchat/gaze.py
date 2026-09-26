"""Visual gaze feedback and a fitted inverse model of the real camera response.

This is a small learned motor baseline, not SAC. A calibration fit does not prove
task improvement: evaluate the resulting policy with fresh VRChat observations.
"""

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .body import Frozen, Number, Pose, WorldObject, WorldState


def yaw_pitch(pose: Pose) -> tuple[float, float]:
    w, x, y, z = pose.orientation
    return (
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
    )


def image_target(world: WorldState, name: str, now: float) -> WorldObject:
    matches = [item for item in world.objects if item.name == name]
    if len(matches) != 1:
        raise ValueError("visual gaze requires a unique observed target")
    item = matches[0]
    if (
        item.source != "vision"
        or item.image_position is None
        or item.last_seen is None
        or not 0 <= now - item.last_seen <= 0.75
        or item.confidence < 0.5
    ):
        age = None if item.last_seen is None else round(now - item.last_seen, 3)
        raise ValueError(
            "visual gaze requires fresh, confident image feedback "
            f"(source={item.source}, image={item.image_position is not None}, "
            f"age_s={age}, confidence={item.confidence:.3f})"
        )
    return item


def gaze_targets(world: WorldState) -> tuple[WorldObject, ...]:
    """Targets usable at this snapshot; execution still rechecks fresh feedback."""
    result = []
    for obj in world.objects:
        if obj.source == "vision":
            try:
                image_target(world, obj.name, world.timestamp)
            except ValueError:
                continue
        result.append(obj)
    return tuple(result)


class VisualGazePolicy(Frozen):
    schema_version: Literal[1] = 1
    skill: Literal["LOOK_AT"] = "LOOK_AT"
    source: Literal["vrchat_image_openvr"] = "vrchat_image_openvr"
    # Maps image displacement to the head-angle displacement that caused it.
    inverse_jacobian: tuple[tuple[Number, Number], tuple[Number, Number]]
    training_samples: int = Field(ge=12)
    gain: Number = Field(default=0.6, gt=0, le=1)

    @model_validator(mode="after")
    def bounded_matrix(self):
        a, b = self.inverse_jacobian
        if max(abs(v) for row in self.inverse_jacobian for v in row) > 4:
            raise ValueError("gaze inverse response is too large")
        if abs(a[0] * b[1] - a[1] * b[0]) < 0.001:
            raise ValueError("gaze inverse response is singular")
        return self

    def desired_angles(self, head: Pose, error: tuple[float, float]) -> tuple[float, float]:
        angles = yaw_pitch(head)
        delta = [
            max(-0.12, min(0.12, -self.gain * sum(a * b for a, b in zip(row, error))))
            for row in self.inverse_jacobian
        ]
        return (
            max(-0.8, min(0.8, angles[0] + delta[0])),
            max(-0.5, min(0.5, angles[1] + delta[1])),
        )


def train_visual_gaze(replay: Path, output: Path) -> dict:
    """Fit finite settled transitions, with a chronological holdout and provenance gates."""
    import numpy as np

    from .replay import Transition

    motions, displacements = [], []
    for line in replay.read_text(encoding="utf-8").splitlines():
        row = Transition.model_validate_json(line)
        if row.outcome != "visual_feedback" or row.decision.goal.skill != "LOOK_AT":
            continue
        before, after = row.observation, row.next_observation
        # The image must cover the whole settled action, not one motor tick with
        # a stale visual sample. Calibration collection is deliberately low-rate.
        if not 0.2 <= after.timestamp - before.timestamp <= 5:
            continue
        name = row.decision.goal.target
        old = image_target(before.world, name, before.timestamp)
        new = image_target(after.world, name, after.timestamp)
        if new.last_seen <= old.last_seen:
            raise ValueError("calibration reused a visual frame")
        heads = before.body.head, after.body.head
        if any(not h.valid or h.source != "openvr_raw" or h.pose is None for h in heads):
            raise ValueError("calibration requires actual OpenVR head observations")
        angles = [yaw_pitch(h.pose) for h in heads]
        motion = tuple(b - a for a, b in zip(*angles))
        if not 0.01 <= math.hypot(*motion) <= 0.25:
            continue
        motions.append(motion)
        displacements.append(tuple(b - a for a, b in zip(old.image_position, new.image_position)))
    if len(motions) < 20:
        raise ValueError("at least 20 settled real visual transitions are required")
    split = int(0.8 * len(motions))
    x, y = np.asarray(motions), np.asarray(displacements)
    if np.linalg.matrix_rank(x[:split]) != 2 or np.linalg.cond(x[:split]) > 20:
        raise ValueError("calibration must excite both head axes")
    response, _, _, _ = np.linalg.lstsq(x[:split], y[:split], rcond=None)
    if np.linalg.cond(response) > 20:
        raise ValueError("visual response is singular or poorly conditioned")
    predicted = x[split:] @ response
    error = float(np.sqrt(np.mean((predicted - y[split:]) ** 2)))
    signal = float(np.sqrt(np.mean(y[split:] ** 2)))
    if signal < 0.01 or error > max(0.015, signal * 0.25):
        raise ValueError("held-out visual response does not support this inverse model")
    inverse = np.linalg.inv(response.T)
    policy = VisualGazePolicy(
        inverse_jacobian=tuple(tuple(float(v) for v in r) for r in inverse),
        training_samples=split,
    )
    report = {
        "training_samples": split,
        "holdout_samples": len(motions) - split,
        "holdout_image_rmse": error,
        "holdout_image_signal_rms": signal,
        "inverse_jacobian": policy.inverse_jacobian,
        "live_improvement_verified": False,
        "method": "least-squares inverse visual dynamics; not SAC",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(policy.model_dump_json(indent=2), encoding="utf-8")
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
