"""Small replay-trained residual policy for the first live motor-learning gate."""

import json
import math
import statistics
from pathlib import Path

from .calibration import ResidualPolicy
from .replay import Transition


def _active_residuals(path: Path, skill: str, hand: str) -> list[tuple[float, float, float]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            transition = Transition.model_validate_json(line)
            goal, command = transition.decision.goal, transition.command
            if (
                goal.skill != skill
                or goal.hand != hand
                or not (0.5 < command.elapsed_s < goal.duration_s - 0.5)
            ):
                continue
            signal = getattr(transition.next_observation.body, hand)
            if not signal.valid or signal.source != "openvr_raw" or signal.pose is None:
                continue
            requested = getattr(transition.action, hand).pose.position
            rows.append(tuple(signal.pose.position[i] - requested[i] for i in range(3)))
    return rows


def train_residual_policy(
    replay: Path, output: Path, *, skill: str = "WAVE", hand: str = "right"
) -> dict:
    """Chronological split; fit on real feedback and publish only on holdout improvement."""
    rows = _active_residuals(replay, skill, hand)
    if len(rows) < 50:
        raise ValueError("at least 50 active real-device transitions are required")
    split = int(len(rows) * 0.8)
    train, holdout = rows[:split], rows[split:]
    median_residual = tuple(statistics.median(row[i] for row in train) for i in range(3))
    correction = tuple(-x for x in median_residual)

    def mean_error(offset):
        return sum(math.dist(row, offset) for row in holdout) / len(holdout)

    before = mean_error((0.0, 0.0, 0.0))
    after = mean_error(median_residual)
    relative = (before - after) / before if before else 0.0
    # Never promote sub-micrometre readback noise as a body improvement.
    improved = before >= 1e-6 and relative >= 0.01
    report = {
        "skill": skill,
        "hand": hand,
        "training_samples": len(train),
        "holdout_samples": len(holdout),
        "mean_error_before_m": before,
        "mean_error_after_m": after,
        "relative_improvement": relative,
        "promoted": improved,
        "limitation": "OpenVR device tracking only; avatar/task success was not observed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = ResidualPolicy(
        skill=skill,
        hand=hand,
        correction=correction,
        training_samples=len(train),
    )
    candidate = output if improved else output.with_suffix(".candidate.json")
    candidate.write_text(policy.model_dump_json(indent=2), encoding="utf-8")
    report["candidate_path"] = str(candidate)
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
