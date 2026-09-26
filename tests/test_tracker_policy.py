import json

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from myumiq_vrchat.body import BodyState, PoseSignal
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.tracker_env import TrackerGoalEnv
from myumiq_vrchat.tracker_policy import (
    OBSERVATION_SIZE,
    TrackerActor,
    export_actor,
    observation,
    pose_error,
)
from myumiq_vrchat.whole_body import target_from_vector, vector


def test_quaternion_sign_equivalence_and_shortest_arc():
    start = posture_target("standing")
    a = vector(start)
    a[:, 3:] *= -1
    same = target_from_vector(a)
    np.testing.assert_allclose(pose_error(start, same), 0, atol=1e-12)
    np.testing.assert_allclose(
        observation(start, start, np.zeros(66), 0.05),
        observation(same, same, np.zeros(66), 0.05),
        atol=1e-12,
    )
    a = vector(start)
    a[:, 3:] = [np.cos(3 * np.pi / 4), 0, 0, np.sin(3 * np.pi / 4)]
    error = pose_error(start, target_from_vector(a))
    np.testing.assert_allclose(np.linalg.norm(error[:, 3:], axis=1), np.pi / 2, atol=1e-8)


def test_goal_change_preserves_pose_previous_action_and_time_then_zero_holds():
    poses = [posture_target(p) for p in ("standing", "sitting_floor", "crouching")]
    env = TrackerGoalEnv(poses, horizon=4, recording=True)
    obs, _ = env.reset(seed=1, options={"start": poses[0], "goal": poses[1]})
    assert obs.shape == (OBSERVATION_SIZE,)
    action = np.linspace(-0.1, 0.1, 66)
    obs, *_ = env.step(action)
    body, previous, step = env.current, env.previous.copy(), env.steps
    changed = env.set_goal(poses[2])
    assert env.current is body and env.steps == step
    np.testing.assert_allclose(env.previous, previous)
    assert not np.array_equal(obs, changed)
    env.step(np.zeros(66))
    np.testing.assert_allclose(vector(body), vector(env.current), atol=1e-12)
    records = env.replay.get_data()
    assert len(records) == 2
    assert records[0].learning.reward_scope == "tracker_geometry"
    assert records[0].environment == "mock" and records[0].outcome == "simulated"
    np.testing.assert_allclose(
        records[0].learning.next_policy_observation[66:],
        records[1].learning.policy_observation[66:],
    )
    assert (
        records[0].learning.next_policy_observation[:66]
        != records[1].learning.policy_observation[:66]
    )
    assert all(
        not any(
            v
            for pair in record.action.left.controls.sticks + record.action.right.controls.sticks
            for v in pair
        )
        for record in records
    )


def test_full_body_rate_bound_and_truncation():
    a, b = posture_target("standing"), posture_target("crouching")
    env = TrackerGoalEnv([a, b], horizon=2)
    env.reset(options={"start": a, "goal": b})
    before = vector(env.current)
    _, _, terminated, truncated, _ = env.step(np.ones(66))
    after = vector(env.current)
    np.testing.assert_allclose(
        np.linalg.norm(after[:, :3] - before[:, :3], axis=1), 0.6 * 0.05, atol=1e-8
    )
    assert not terminated and not truncated
    assert env.step(np.zeros(66))[3]
    with pytest.raises(ValueError):
        env.step(np.full(66, np.nan))


def test_actor_export_round_trip_and_hash(tmp_path):
    torch = pytest.importorskip("torch")
    sac = pytest.importorskip("stable_baselines3").SAC
    torch.set_num_threads(1)
    env = TrackerGoalEnv([posture_target("standing"), posture_target("crouching")])
    model = sac(
        "MlpPolicy", env, device="cpu", buffer_size=10, policy_kwargs={"net_arch": [16, 16]}
    )
    path = tmp_path / "actor.pt"
    export_actor(model, path)
    policy = TrackerActor(path)
    obs, _ = env.reset(seed=3)
    expected = model.predict(obs, deterministic=True)[0]
    np.testing.assert_allclose(policy.predict(obs), expected, atol=1e-6)
    assert policy.manifest["action_size"] == 66 and not policy.manifest["promoted"]
    invalid = BodyState(
        head=PoseSignal(timestamp=1), left=PoseSignal(timestamp=1), right=PoseSignal(timestamp=1)
    )
    with pytest.raises(ValueError, match="complete valid body feedback"):
        policy.step(invalid, env.goal, 0.05)
    manifest = json.loads(path.with_suffix(".json").read_text())
    manifest["sha256"] = "changed"
    path.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed"):
        TrackerActor(path)
