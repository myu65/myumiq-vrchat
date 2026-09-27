import json

import pytest
from test_practice_loop import case

from myumiq_vrchat.body import BodyState, PoseSignal, WorldState
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.practice_feedback import collect
from myumiq_vrchat.replay import Observation, WholeBodyTransition
from myumiq_vrchat.whole_body import PARTS


def fixture(tmp_path, defect=None):
    pose = posture_target("standing")
    probe = case(offset=0.0)
    rows = []
    actor_sha = "a" * 64

    def observation(t):
        fields = {}
        for part in PARTS:
            name = {"left_hand": "left", "right_hand": "right"}.get(part, part)
            fields[name] = PoseSignal(
                pose=pose.pose_for(part),
                valid=True,
                connected=True,
                source="simulated" if defect == "simulated" else "openvr_raw",
                timestamp=t - 1 if defect == "stale" else t,
            )
        return Observation(timestamp=t, world=WorldState(), body=BodyState(**fields))

    for i in range(5):
        rows.append(
            WholeBodyTransition(
                observation=observation(i * 0.05),
                next_observation=observation((i + 1) * 0.05),
                body_goal=case(offset=-0.1).goal if defect == "goal" else probe.goal,
                action=pose,
                policy_id="autonomous",
                outcome="device_feedback",
                environment="vrchat",
                intent_metadata=dict(
                    id=7,
                    source="operator_body_goal",
                    motor_policy="articulated:"
                    + ("b" * 16 if defect == "actor" else actor_sha[:16]),
                    execution=dict(phase="completed" if i == 4 else "running"),
                ),
            )
        )
    (tmp_path / "experience.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in rows), "utf-8"
    )
    report = dict(
        session=str(tmp_path),
        controlled_home=True,
        error=None,
        actor_sha256=actor_sha,
        probes=[dict(case_id=probe.id, generation=7)],
    )
    return report, probe, actor_sha


def test_device_replay_supplies_only_train_starts_and_original_goal(tmp_path):
    report, probe, actor = fixture(tmp_path)
    result, starts = collect(report, [probe], actor, 0.0)
    assert result["probes"][0]["completed"]
    assert result["probes"][0]["conditions"]["success"]
    assert starts[0].goal == probe.goal and starts[0].start is None
    assert starts[0].start_pose == posture_target("standing")
    _, heldout_starts = collect(report, [probe.model_copy(update={"split": "heldout"})], actor, 0.0)
    assert not heldout_starts
    json.dumps(result)


@pytest.mark.parametrize("defect", ["simulated", "stale", "goal", "actor"])
def test_nonmatching_or_unobserved_feedback_is_rejected(tmp_path, defect):
    report, probe, actor = fixture(tmp_path, defect)
    with pytest.raises(ValueError):
        collect(report, [probe], actor, 0.0)
