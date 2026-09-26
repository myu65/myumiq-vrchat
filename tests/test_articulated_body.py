import numpy as np
import pytest

pytest.importorskip("gymnasium")

from myumiq_vrchat.articulated_body import ArticulatedRig, JointState, increment
from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
from myumiq_vrchat.tracker_action import integrate_tracker_action
from myumiq_vrchat.tracker_basis import RateBasis
from myumiq_vrchat.whole_body import vector


def fixture():
    parents = (-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 0, 11)
    offsets = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.4),
        (0.0, 0.0, 0.3),
        (0.0, 0.2, 0.1),
        (0.0, 0.0, -0.3),
        (0.0, 0.0, -0.25),
        (0.0, -0.2, 0.1),
        (0.0, 0.0, -0.3),
        (0.0, 0.0, -0.25),
        (0.0, 0.1, -0.4),
        (0.0, 0.0, -0.4),
        (0.0, -0.1, -0.4),
        (0.0, 0.0, -0.4),
    )
    nodes = (2, 1, 0, 5, 8, 4, 7, 9, 11, 10, 12)
    rig = ArticulatedRig(
        names=tuple(f"joint-{i}" for i in range(13)),
        parents=parents,
        offsets=offsets,
        reference_rotations=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),) * 13,
        tracker_nodes=nodes,
        orientation_nodes=nodes,
        scale=1.0,
    )
    a = JointState(np.array([0.0, 0.0, 0.9]), np.tile([1.0, 0.0, 0.0, 0.0], (13, 1)))
    b = JointState(a.root + [0.05, 0, 0], increment(a.rotations, np.ones((13, 3)) * 0.1, 0.5))
    return rig, [a, b]


def test_connected_lengths_and_physical_limits_through_random_moves_and_zero_hold():
    rig, states = fixture()
    state = states[1]
    rng = np.random.default_rng(42)
    original = vector(rig.forward(state))
    for i in range(60):
        dt = 0.01 if i % 2 else 0.1
        before = rig.forward(state)
        state, target, rates, scale = rig.advance(state, rng.uniform(-1, 1, rig.action_size), dt)
        assert 0 < scale <= 1
        assert np.linalg.norm(rates.reshape(-1, 3), axis=1).max() <= 1 + 1e-9
        np.testing.assert_allclose(
            vector(integrate_tracker_action(before, rates.reshape(11, 6), dt)),
            vector(target),
            atol=1e-9,
        )
        points = vector(target)[:, :3]
        for left, right, length in ((3, 5, 0.25), (4, 6, 0.25), (7, 9, 0.4), (8, 10, 0.4)):
            assert np.linalg.norm(points[left] - points[right]) == pytest.approx(length, abs=1e-12)
    held = vector(rig.forward(state))
    assert not np.allclose(held, original)
    state, target, rates, _ = rig.advance(state, np.zeros(rig.action_size), 0.02)
    np.testing.assert_allclose(vector(target), held, atol=1e-12)
    np.testing.assert_allclose(rates, 0, atol=1e-12)
    # Tracking-space root translation moves every landmark once; controls remain neutral.
    action = np.zeros(rig.action_size)
    action[0] = 0.1
    _, moved, _, _ = rig.advance(state, action, 0.1)
    np.testing.assert_allclose(
        vector(moved)[:, :3] - held[:, :3], np.tile([0.006, 0, 0], (11, 1)), atol=1e-12
    )
    assert not any(x for pair in moved.left.controls.sticks for x in pair)


def test_goal_change_preserves_articulation_and_replay_records_actual_decoding():
    rig, states = fixture()
    decoder = RateBasis.fit(np.random.default_rng(3).uniform(-1, 1, (100, rig.action_size)), 8)
    env = ArticulatedGoalEnv(rig, states, decoder=decoder, recording=True)
    env.reset(seed=1)
    env.step(np.ones(8) * 0.3)
    state, current, previous = env.state, env.current, env.previous.copy()
    env.set_goal(states[0])
    assert env.state is state and env.current is current and env.steps == 1
    np.testing.assert_array_equal(env.previous, previous)
    env.step(np.zeros(8))
    np.testing.assert_allclose(vector(env.current), vector(current), atol=1e-12)
    first, second = env.replay.get_data()
    assert first.intent_metadata["next_joint_state"] == second.intent_metadata["joint_state"]
    assert first.learning.observation_contract == "myumiq-articulated-goal-v1"
    assert len(first.learning.policy_observation) == 210 + 4 * 13
    assert len(first.intent_metadata["joint_action"]) == rig.action_size
    assert first.environment == "mock" and first.learning.reward_scope == "tracker_geometry"


def test_goal_anchored_export_matches_sac_and_keeps_arbitrary_goal_pose(tmp_path):
    torch = pytest.importorskip("torch")
    sac = pytest.importorskip("stable_baselines3").SAC
    from myumiq_vrchat.articulated_policy import (
        AnchoredSACPolicy,
        ArticulatedActor,
        export_articulated,
    )

    torch.set_num_threads(1)
    rig, states = fixture()
    decoder = RateBasis.fit(np.random.default_rng(3).uniform(-1, 1, (100, rig.action_size)), 8)
    env = ArticulatedGoalEnv(rig, states, decoder=decoder)
    model = sac(
        AnchoredSACPolicy, env, buffer_size=10, device="cpu", policy_kwargs={"net_arch": [16, 16]}
    )
    path = tmp_path / "joints.pt"
    export_articulated(model, path, rig, decoder)
    actor = ArticulatedActor(path)
    obs, _ = env.reset(seed=4)
    np.testing.assert_allclose(
        actor.predict(obs)[0], model.predict(obs, deterministic=True)[0], atol=1e-6
    )
    env.previous = np.ones(66) * 0.2
    obs = env.set_goal(env.state)
    np.testing.assert_array_equal(actor.predict(obs)[0], np.zeros(8))
    held = vector(env.current)
    for _ in range(20):
        action = actor.predict(obs)[0]
        obs, *_ = env.step(action)
    np.testing.assert_allclose(vector(env.current), held, atol=1e-12)


def test_fitting_observed_pose_does_not_mutate_it_or_accept_incompatible_body_shape():
    torch = pytest.importorskip("torch")
    from myumiq_vrchat.articulated_fit import fit_body, tensor_forward
    from myumiq_vrchat.whole_body import target_from_vector

    torch.set_num_threads(1)
    rig, states = fixture()
    before = vector(rig.forward(states[0])).copy()
    goal = rig.forward(states[1])
    result = tensor_forward(rig, torch.tensor(states[1].root), torch.tensor(states[1].rotations))
    np.testing.assert_allclose(result.numpy(), vector(goal), atol=1e-12)
    fitted, report = fit_body(rig, goal, states[0])
    assert report["maximum_position_error_m"] < 0.005
    assert report["joint_state_source"] == "fitted_tracker_estimate"
    assert report["convergence"] == "tracker_tolerances"
    assert report["evaluations"] < 160
    np.testing.assert_array_equal(vector(rig.forward(states[0])), before)
    bad = vector(goal)
    bad[3, 2] += 1.0
    with pytest.raises(ValueError, match="does not fit calibrated rig"):
        fit_body(rig, target_from_vector(bad), fitted, max_iter=40)


def test_fitting_exact_prior_stops_without_backward_and_cancellation_still_wins(monkeypatch):
    torch = pytest.importorskip("torch")
    from myumiq_vrchat.articulated_fit import fit_body

    rig, states = fixture()
    pose = rig.forward(states[1])

    def unexpected_backward(*args, **kwargs):
        pytest.fail("an already valid estimate must not run backward")

    monkeypatch.setattr(torch.Tensor, "backward", unexpected_backward)
    fitted, report = fit_body(rig, pose, states[1])
    assert report["evaluations"] == 1
    np.testing.assert_allclose(vector(rig.forward(fitted)), vector(pose), atol=1e-12)
    with pytest.raises(RuntimeError, match="cancelled"):
        fit_body(rig, pose, states[1], cancelled=lambda: True)
