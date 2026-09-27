import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from myumiq_vrchat.body import BodyCondition, BodyGoal
from myumiq_vrchat.condition_validation import ConditionCase
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.practice_feedback import (
    append_training,
    digest,
    live_admissible,
    sampled_quality,
)
from myumiq_vrchat.practice_loop import PracticeConfig, PracticeLoop


def case(name="practice", split="train", offset=-0.1):
    return ConditionCase(
        id=name,
        start="STAND",
        split=split,
        goal=BodyGoal(
            conditions=(
                BodyCondition(part="head", frame="current", position=(None, None, offset)),
            ),
            duration_s=2.0,
        ),
    )


def measurement():
    return {
        "probes": [
            dict(
                id="practice",
                completed=True,
                conditions={"success": True},
                duration_s=2.0,
                start_pose=posture_target("standing").model_dump(mode="json"),
                floor_failure=False,
                motion_quality=dict(
                    maximum_foot_displacement_m=0.01,
                    maximum_acceleration_m_s2=1.0,
                    maximum_jerk_m_s3=10.0,
                ),
            )
        ]
    }


def test_irregular_device_time_is_resampled_and_gaps_do_not_score_zero():
    times = np.array([0.0, 0.03, 0.06, 0.12, 0.2, 0.25])
    poses = np.ones((len(times), 11, 3))
    poses[:, 3, 0] += times * 0.2
    result = sampled_quality(times, poses)
    assert result["maximum_speed_m_s"] == pytest.approx(0.2)
    assert result["maximum_acceleration_m_s2"] < 1e-10
    for bad in ([0.0, 0.03, 0.03, 0.12, 0.2, 0.25], [0.0, 0.03, 0.06, 0.12, 0.2, 1.0]):
        with pytest.raises(ValueError, match="continuous"):
            sampled_quality(bad, poses)


def test_feedback_keeps_frozen_cases_and_rejects_heldout_training():
    original = [case(), case("test", "heldout", -0.15)]
    start = original[0].model_copy(
        update={"id": "observed", "start": None, "start_pose": posture_target("standing")}
    )
    grown = append_training(original, [start, start])
    assert grown == [*original, start]
    assert grown[1].model_dump_json() == original[1].model_dump_json()
    with pytest.raises(ValueError, match="held-out"):
        append_training(original, [start.model_copy(update={"split": "heldout"})])


@pytest.mark.parametrize(
    "defect", ["incomplete", "missing", "slide", "jerk", "floor", "slow", "start"]
)
def test_live_admission_requires_complete_continuous_nonregressing_evidence(defect):
    old, new = measurement(), measurement()
    assert live_admissible(old, new)
    row = new["probes"][0]
    if defect == "incomplete":
        row["conditions"]["success"] = False
    if defect == "missing":
        row["motion_quality"] = None
    if defect == "slide":
        row["motion_quality"]["maximum_foot_displacement_m"] = 0.1
    if defect == "jerk":
        row["motion_quality"]["maximum_jerk_m_s3"] = 20.0
    if defect == "floor":
        row["floor_failure"] = True
    if defect == "slow":
        row["duration_s"] = 5.0
    if defect == "start":
        row["start_pose"]["head"]["position"][0] += 0.04
    assert not live_admissible(old, new)


def inputs(tmp_path):
    prior = tmp_path / "prior"
    prior.mkdir()
    actor = prior / "candidate-actor.pt"
    actor.write_bytes(b"original")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps(
            dict(
                actor=str(actor),
                actor_sha256=digest(actor),
                rig_sha256="a" * 64,
                reference_floor=0.0,
                condition_goals=True,
                goals={"STAND": posture_target("standing").model_dump(mode="json")},
            )
        ),
        "utf-8",
    )
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([c.model_dump(mode="json") for c in (case(), case("test", "heldout", -0.15))]),
        "utf-8",
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "result.json").write_text("{}")
    return PracticeConfig(
        tasks=tasks,
        cases=cases,
        prior=prior,
        reference_corpus=corpus,
        adapter=("host",),
        probe_ids=("practice", "test"),
        rounds=2,
        updates=2,
    )


def test_offline_rejected_candidates_never_go_live_and_feedback_reaches_next_round(tmp_path):
    config = inputs(tmp_path)
    original_hash = digest(config.prior / "candidate-actor.pt")
    case_counts = []

    def runner(argv, timeout, log):
        assert "condition_training" in argv[2]

        def path(flag):
            return Path(argv[argv.index(flag) + 1])

        out = path("--out")
        out.mkdir()
        case_counts.append(len(json.loads(path("--cases").read_text())))
        (out / "candidate-actor.pt").write_bytes(b"rejected")
        (out / "result.json").write_text(
            json.dumps(
                dict(
                    initial_actor_sha256=original_hash,
                    baseline_actor_sha256=original_hash,
                    cases_sha256=digest(path("--cases")),
                    updates=2,
                    eligible_for_controlled_trial=False,
                )
            )
        )

    loop = PracticeLoop(config, tmp_path / "batch", runner)
    calls = []

    def trial(folder, tasks, probes):
        assert json.loads(tasks.read_text())["actor_sha256"] == original_hash
        calls.append(folder)
        start = posture_target("standing")
        start = start.model_copy(
            update={
                "head": start.head.model_copy(
                    update={"position": (0.0, 0.0, 1.4 + 0.01 * len(calls))}
                )
            }
        )
        feedback = probes[0].model_copy(
            update={"id": f"live-{len(calls)}", "start": None, "start_pose": start}
        )
        return measurement(), [feedback]

    loop.trial = trial
    result = loop.run()
    assert result["completed"] and len(calls) == 3
    assert case_counts == [3, 4]
    assert result["selected_actor_sha256"] == original_hash
    assert not any(r["adopted_in_batch"] for r in result["rounds"])
    assert json.loads((loop.out / "frozen-cases.json").read_text()) == json.loads(
        config.cases.read_text()
    )


def test_live_timeout_always_calls_cleanup_and_failed_restoration_stops_batch(tmp_path):
    config = inputs(tmp_path)
    calls = []

    def runner(argv, timeout, log):
        op = argv[argv.index("--operation") + 1]
        calls.append(op)
        if op == "run":
            raise subprocess.TimeoutExpired(argv, timeout)
        path = Path(argv[argv.index("--report") + 1])
        path.write_text(json.dumps(dict(restored=False, errors=["profile restore failed"])))

    loop = PracticeLoop(config, tmp_path / "batch", runner)
    with pytest.raises(RuntimeError, match="restoration"):
        loop.run()
    assert calls == ["run", "cleanup"]


def test_interrupted_directory_is_never_silently_replayed(tmp_path):
    config = inputs(tmp_path)
    PracticeLoop(config, tmp_path / "batch")
    with pytest.raises(FileExistsError):
        PracticeLoop(config, tmp_path / "batch")


def test_admitted_candidates_use_latest_prior_but_keep_original_baseline(tmp_path):
    config = inputs(tmp_path)
    original_actor = config.prior / "candidate-actor.pt"
    original_tasks = config.tasks.read_bytes()
    selected = []

    def runner(argv, timeout, log):
        def path(flag):
            return Path(argv[argv.index(flag) + 1])

        assert path("--baseline") == original_actor
        assert path("--prior") == (config.prior if not selected else selected[-1])
        out = path("--out")
        out.mkdir()
        (out / "candidate-actor.pt").write_bytes(f"candidate-{len(selected)}".encode())
        tasks = json.loads(path("--tasks").read_text())
        (out / "result.json").write_text(
            json.dumps(
                dict(
                    initial_actor_sha256=tasks["actor_sha256"],
                    baseline_actor_sha256=digest(original_actor),
                    cases_sha256=digest(path("--cases")),
                    updates=2,
                    eligible_for_controlled_trial=True,
                )
            )
        )
        selected.append(out)

    loop = PracticeLoop(config, tmp_path / "batch", runner)
    trial_actors = []

    def trial(folder, tasks, probes):
        trial_actors.append(json.loads(tasks.read_text())["actor_sha256"])
        return measurement(), []

    loop.trial = trial
    result = loop.run()
    assert all(r["adopted_in_batch"] for r in result["rounds"])
    assert len(set(trial_actors)) == 3
    assert config.tasks.read_bytes() == original_tasks
    assert result["selected_prior"] == str(selected[-1])
