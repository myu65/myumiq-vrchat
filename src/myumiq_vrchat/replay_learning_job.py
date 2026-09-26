"""Configured candidate job for the existing executive's single learning slot."""

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from pydantic import Field, model_validator

from .body import Frozen, Number


class ReplayRefinementSettings(Frozen):
    prior: Path
    train_replay: Path
    validation_replay: Path
    reference_corpus: Path
    updates: int = Field(default=50, ge=1, le=1000)
    learning_rate: Number = Field(default=1e-5, gt=0, le=0.001)
    floor_weight: Number = Field(default=50.0, gt=0, le=1000)
    reference_rehearsal: bool = False
    backtracking_steps: int = Field(default=0, ge=0, le=3)
    timeout_s: int = Field(default=300, ge=30, le=1800)

    @model_validator(mode="after")
    def external_inputs(self):
        repo = Path(__file__).resolve().parents[2]
        for path in (self.prior, self.train_replay, self.validation_replay, self.reference_corpus):
            if path.resolve().is_relative_to(repo):
                raise ValueError("replay learning assets must be outside repository")
        if self.train_replay.resolve() == self.validation_replay.resolve():
            raise ValueError("independent validation replay required")
        return self


def make_replay_task(settings):
    # Preserve the identity of jobs saved before the optional penalty setting.
    omitted = {"floor_weight"} if settings.floor_weight == 50.0 else set()
    if not settings.reference_rehearsal:
        omitted.add("reference_rehearsal")
    if not settings.backtracking_steps:
        omitted.add("backtracking_steps")
    key = hashlib.sha256(settings.model_dump_json(exclude=omitted).encode()).hexdigest()
    return dict(
        id=uuid.uuid4().hex,
        goal_id=None,
        capability="ARTICULATED_REFINEMENT",
        kind="replay_refinement",
        configuration_id=key,
        status="queued",
        blockers=[],
        scope="candidate_only_observed_starts_model_refinement",
    )


def train_replay_task(settings, output, cancel):
    """Fixed packaged process with CPU/deadline limits; never mutate live inference."""
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("candidate output must be outside repository")
    command = [
        sys.executable,
        "-m",
        "myumiq_vrchat.experience_training",
        "--prior",
        str(settings.prior),
        "--train-replay",
        str(settings.train_replay),
        "--heldout-replay",
        str(settings.validation_replay),
        "--reference-corpus",
        str(settings.reference_corpus),
        "--output",
        str(output),
        "--updates",
        str(settings.updates),
        "--learning-rate",
        str(settings.learning_rate),
        "--floor-weight",
        str(settings.floor_weight),
        "--backtracking-steps",
        str(settings.backtracking_steps),
    ]
    if settings.reference_rehearsal:
        command.append("--reference-rehearsal")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1"}
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as process:
        deadline = time.monotonic() + settings.timeout_s
        while True:
            if cancel.is_set() or time.monotonic() >= deadline:
                process.kill()
                process.communicate()
                raise RuntimeError("replay refinement cancelled or timed out")
            try:
                _, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError("replay refinement failed: " + stderr[-300:])
    result_path = output / "result.json"
    if not result_path.is_file() or result_path.stat().st_size > 2_000_000:
        raise ValueError("missing or oversized candidate evaluation")
    result = json.loads(result_path.read_text("utf-8"))
    from .articulated_actor import ArticulatedActor

    actor = ArticulatedActor(output / "candidate-actor.pt")
    if (
        actor.manifest["sha256"] != result["candidate_sha256"]
        or result["automatic_live_sync"]
        or result["promoted"]
        or result["avatar_verified"]
    ):
        raise ValueError("candidate evidence or isolation contract differs")
    return None, dict(
        policy=str(output / "candidate-actor.pt"),
        report=str(result_path),
        sha256=actor.manifest["sha256"],
        eligible_for_controlled_trial=result["eligible_for_controlled_trial"],
        before=result["before"],
        after=result["after"],
        scope=result["scope"],
        live_promoted=False,
    )
