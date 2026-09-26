import math

import pytest

from myumiq_vrchat.body import BodyGoal, BodyTask, simulated_body

np = pytest.importorskip("numpy")

from myumiq_vrchat.whole_body import (  # noqa: E402 (optional motion dependency)
    PARTS,
    PeriodicImitation,
    WholeBodyPolicy,
    bounded_step,
    floor_sitting_reward,
    target_from_vector,
    vector,
)


def fixture(t):
    a = np.zeros((11, 7))
    a[:, 3] = 1
    a[:, 2] = 1 + 0.1 * math.sin(2 * math.pi * t)
    return target_from_vector(a)


def test_imitation_generalizes_to_unseen_times_and_roundtrips(tmp_path):
    frames = [(i / 20, fixture(i / 20)) for i in range(20)]
    model = PeriodicImitation.fit(frames, 1, "fixture", harmonics=2)
    path = tmp_path / "policy.json"
    model.save(path)
    restored = PeriodicImitation.model_validate_json(path.read_text())
    for t in (0.025, 0.175, 0.425, 1.625):
        assert np.max(np.abs(vector(restored.sample(t)) - vector(fixture(t)))) < 1e-5


def test_velocity_and_angular_limits_apply_to_every_part():
    a = vector(fixture(0))
    b = a.copy()
    b[:, 0] += 1
    b[:, 3:] = [0, 1, 0, 0]
    out = vector(bounded_step(target_from_vector(a), target_from_vector(b), 0.01))
    assert np.max(np.linalg.norm(out[:, :3] - a[:, :3], axis=1)) <= 0.006 + 1e-9
    assert np.max(2 * np.arccos(np.clip(out[:, 3], -1, 1))) <= 0.02 + 1e-9


def test_policy_rejects_missing_feedback_and_unsupported_task():
    from myumiq_vrchat.body import rest_target

    model = PeriodicImitation.fit([(i / 20, fixture(i / 20)) for i in range(20)], 1, "walk")
    goal = BodyGoal(
        tasks=(BodyTask(id="walk", kind="locomotion", target="walk", effectors=PARTS),),
        duration_s=1,
    )
    policy = WholeBodyPolicy(model)
    with pytest.raises(ValueError, match="feedback"):
        policy.step(simulated_body(rest_target(), 0.0), goal, 0.01)
    other = goal.model_copy(update={"constraints": (object(),)})
    with pytest.raises(ValueError, match="single"):
        policy.step(simulated_body(fixture(0), 0.0), other, 0.01)


def test_reference_free_reward_does_not_invent_contacts():
    result = floor_sitting_reward(simulated_body(fixture(0), 0.0))
    assert not result["contacts_observed"] and not result["success"]


def test_fk_postures_preserve_limb_lengths_and_floor_sit_reward():
    from myumiq_vrchat.postures import posture_target

    for name in ("standing", "crouching", "sitting_floor", "lying"):
        target = posture_target(name)
        assert target.is_full_body
        for side in ("left", "right"):
            assert math.dist(
                target.pose_for(side + "_knee").position, target.pose_for(side + "_foot").position
            ) == pytest.approx(0.4)
            assert math.dist(
                target.pose_for(side + "_elbow").position, target.pose_for(side + "_hand").position
            ) == pytest.approx(0.25)
    body = simulated_body(posture_target("sitting_floor"), 0.0)
    assert floor_sitting_reward(body)["geometry_reward"] == pytest.approx(1.0)
    assert not floor_sitting_reward(body)["success"]
    assert floor_sitting_reward(body, {p: True for p in ("pelvis", "left_foot", "right_foot")})[
        "success"
    ]


def test_full_body_replay_preserves_goal_and_observation_source(tmp_path):
    from myumiq_vrchat.body import WorldState
    from myumiq_vrchat.replay import Observation, ReplayBuffer, WholeBodyTransition, summarize

    target = fixture(0)
    goal = BodyGoal(
        tasks=(BodyTask(id="walk", kind="locomotion", target="walk", effectors=PARTS),),
        duration_s=1,
    )
    before = Observation(timestamp=0.0, body=simulated_body(target, 0.0), world=WorldState())
    record = WholeBodyTransition(
        observation=before,
        next_observation=before,
        body_goal=goal,
        action=target,
        policy_id="test",
        outcome="simulated",
        environment="mock",
    )
    replay = ReplayBuffer(4)
    replay.add(record.model_dump_json())
    replay.save_state(tmp_path / "experience")
    restored = ReplayBuffer(4)
    restored.load_state(tmp_path / "experience")
    assert restored.get_data() == [record]
    assert summarize(tmp_path / "experience.jsonl")["head_sources"] == ["simulated"]
    assert restored.get_data()[0].learning is None

    from myumiq_vrchat.replay import TrackerLearningStep

    learning = TrackerLearningStep(
        rates=(0.0,) * 66,
        dt=0.05,
        policy_observation=(0.0, 1.0),
        next_policy_observation=(0.0, 1.0),
        observation_contract="fixture-v1",
        reward_components={"progress": 0.2},
        reward_scope="tracker_geometry",
        evidence_ids=("fixture-measurement",),
        terminated=False,
        truncated=True,
    )
    payload = record.model_dump()
    payload.update(learning=learning, reward=0.2)
    trained = WholeBodyTransition.model_validate(payload)
    replay.add(trained.model_dump_json())
    assert replay.get_data()[-1].learning == learning
    payload["reward"] = None
    with pytest.raises(ValueError, match="reward"):
        WholeBodyTransition.model_validate(payload)
    with pytest.raises(ValueError, match="dimensions"):
        TrackerLearningStep.model_validate(
            {**learning.model_dump(), "next_policy_observation": (0.0,)}
        )
