"""Bounded learning jobs around existing trainers; no LLM-selected commands/assets."""

import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .body import Frozen, Number
from .capabilities import motion_capability
from .replay_learning_job import ReplayRefinementSettings


class MotionLesson(Frozen):
    clip: str = Field(min_length=1, max_length=80)
    kind: Literal["periodic", "finite"] = "periodic"
    playback_rate: Number = Field(default=0.25, gt=0, le=1)
    hand: Literal["left", "right"] | None = None
    description: str = Field(default="", max_length=80)
    source: Path | None = None
    license: Path | None = None
    source_url: str | None = None
    retarget: Path | None = None


class LearningSettings(Frozen):
    replay_refinement: ReplayRefinementSettings | None = None
    acquire_configured_motions: bool = False
    motion_source: Path | None = None
    motion_license: Path | None = None
    source_url: str = ""
    clip: str = "Walk_Loop"
    timeout_s: int = Field(default=60, ge=5, le=300)
    motions: dict[str, MotionLesson] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def supported_lessons(self):
        if not all(motion_capability(name) for name in self.motions):
            raise ValueError("unsupported motion learning capability")
        for name, lesson in self.motions.items():
            if (name == "WAVE") != (lesson.hand is not None):
                raise ValueError("only a WAVE lesson must declare its demonstrated hand")
        return self

    def lesson(self, capability):
        if not motion_capability(capability):
            return None
        return self.motions.get(capability) or (
            MotionLesson(clip=self.clip) if capability == "WALK_IN_PLACE" else None
        )

    def assets(self, lesson):
        return (
            lesson.source or self.motion_source,
            lesson.license or self.motion_license,
            lesson.source_url or self.source_url,
        )


def make_task(capability, goal_id, registry, settings):
    item = registry.get(capability)
    blockers = [p for p in item.prerequisites if not registry.get(p).available]
    adapter = {
        "WALK_IN_PLACE": "scripts/train-whole-body.py",
        "SIT": "whole_body.floor_sitting_reward (task environment required)",
        "REACH": "scripts/refine_reach_replay.py",
    }.get(capability)
    lesson = settings.lesson(capability)
    if lesson and item.method == "imitation":
        adapter = "myumiq_vrchat.motion_training"
        if (capability == "WAVE") != (lesson.hand is not None):
            blockers.append("motion_demonstration_hand_required")
        source, license, url = settings.assets(lesson)
        if not source or not license or not url:
            blockers.append("configured_CC0_motion_required")
        elif (
            not source.is_file()
            or not license.is_file()
            or lesson.retarget
            and not lesson.retarget.is_file()
        ):
            blockers.append("motion_asset_missing")
    elif item.method == "reference_free_rl":
        blockers.append("configured_task_environment_and_observed_success_required")
    elif item.method == "vrchat_replay":
        blockers.append("validated_REACH_task_replay_and_baseline_policy_required")
    elif item.method == "imitation":
        blockers.append("capability_specific_teacher_and_motor_adapter_required")
    else:
        blockers.append("capability_definition_and_evaluator_required")
    return {
        "id": uuid.uuid4().hex,
        "goal_id": goal_id,
        "capability": capability,
        "method": item.method,
        "adapter": adapter,
        "prerequisites": list(item.prerequisites),
        "blockers": blockers,
        "status": "blocked" if blockers else "queued",
        "scope": "candidate_training_not_general_mastery",
    }


def train_task(task, settings, output, cancel=None):
    lesson = settings.lesson(task["capability"])
    if lesson is None or task["method"] != "imitation":
        raise ValueError("no automatic trainer is enabled for this capability")
    repo = Path(__file__).resolve().parents[2]
    output = output.resolve()
    if output.is_relative_to(repo):
        raise ValueError("training output must be outside repository")
    source, license, url = settings.assets(lesson)
    command = [
        sys.executable,
        "-m",
        "myumiq_vrchat.motion_training",
        str(source),
        "--license",
        str(license),
        "--source-url",
        url,
        "--clip",
        lesson.clip,
        "--kind",
        lesson.kind,
        "--out",
        str(output),
    ]
    if lesson.retarget:
        command += ["--retarget", str(lesson.retarget)]
    # Fixed trainer, argument list and operator-owned assets. No shell/code from goals.
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as process:
        deadline = time.monotonic() + settings.timeout_s
        while True:
            if (cancel is not None and cancel.is_set()) or time.monotonic() > deadline:
                process.kill()
                process.communicate()
                raise RuntimeError("motion training cancelled or timed out")
            try:
                _, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError("motion trainer failed: " + stderr[-300:])
    report = json.loads((output / "report.json").read_text("utf-8"))
    if not (
        report["position_mean_cm"] < report["baseline_position_mean_cm"]
        and report["position_max_cm"] < 5
        and report["rotation_mean_deg"] < 5
    ):
        raise ValueError("imitation candidate failed the held-out interpolation gate")
    from .motion_prior import load_motion

    model = load_motion((output / "policy.json").read_text("utf-8"))
    # Finite normalized full-body targets and a conservative workspace before activation.
    for i in range(32):
        target = model.sample(i / 32)
        for part in (
            "head",
            "chest",
            "pelvis",
            "left_hand",
            "right_hand",
            "left_foot",
            "right_foot",
        ):
            p = target.pose_for(part).position
            if abs(p[0]) > 1.5 or abs(p[1]) > 1.5 or not -0.2 <= p[2] <= 2.3:
                raise ValueError("learned trajectory outside supported workspace")
    return model, {
        "policy": str(output / "policy.json"),
        "report": str(output / "report.json"),
        "sha256": hashlib.sha256((output / "policy.json").read_bytes()).hexdigest(),
        "playback_rate": lesson.playback_rate,
        "hand": lesson.hand,
        "description": lesson.description,
        "position_mean_cm": report["position_mean_cm"],
        "baseline_position_mean_cm": report["baseline_position_mean_cm"],
        "evaluation_scope": report["evaluation_scope"],
    }


def restore_candidate(result, learning_root):
    """Load only this executive's evaluated checkpoint, not a persisted availability claim."""
    from .motion_prior import load_motion

    path = Path(result["policy"]).resolve()
    if not path.is_relative_to(learning_root.resolve()):
        raise ValueError("learned policy outside owned learning directory")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != result["sha256"]:
        raise ValueError("learned checkpoint changed since evaluation")
    return load_motion(content)
