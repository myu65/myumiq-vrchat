# ruff: noqa: E402 -- check optional learning dependencies before importing adapters
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from test_articulated_body import fixture

from myumiq_vrchat.articulated_dynamics import (
    DifferentiableRig,
    blended_goal,
    joint_teacher,
    observation,
    policy_rollout_start,
    pose_error,
    reward_components,
    turned_goal,
    worst_tracker_distance,
)
from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
from myumiq_vrchat.whole_body import vector


@pytest.mark.parametrize(
    "weight,power,worst_weight", [(20.0, 2, 0.0), (50.0, 1, 0.0), (50.0, 1, 1.0)]
)
def test_differentiable_actuator_observation_and_reward_match_numpy_and_record_floor(
    weight, power, worst_weight
):
    torch.set_num_threads(1)
    rig, states = fixture()
    dynamics = DifferentiableRig(rig).double()
    rng = np.random.default_rng(5)
    for dt in (0.01, 0.1):
        env = ArticulatedGoalEnv(
            rig,
            states,
            dt=dt,
            reference_floor=0.0,
            recording=True,
            floor_weight=weight,
            floor_power=power,
            worst_tracker_weight=worst_weight,
        )
        obs, _ = env.reset(options={"start": states[0], "goal": states[1]})
        action = rng.uniform(-1, 1, rig.action_size)
        root, q = (torch.tensor(states[0].root[None]), torch.tensor(states[0].rotations[None]))
        before, goal = dynamics(root, q), torch.tensor(vector(env.goal)[None])
        previous = torch.zeros(1, 66, dtype=torch.float64)
        np.testing.assert_allclose(
            observation(before, goal, previous, dt, q).numpy()[0], obs, atol=1e-6
        )
        new_root, new_q, after, rates, _ = dynamics.step(root, q, torch.tensor(action[None]), dt)
        expected = reward_components(
            before,
            after,
            goal,
            rates,
            previous,
            reference_floor=0.0,
            floor_weight=weight,
            floor_power=power,
            worst_tracker_weight=worst_weight,
        )
        next_obs, _, _, _, info = env.step(action)
        np.testing.assert_allclose(after.numpy()[0], vector(env.current), atol=1e-9)
        np.testing.assert_allclose(new_root.numpy()[0], env.state.root, atol=1e-9)
        np.testing.assert_allclose(new_q.numpy()[0], env.state.rotations, atol=1e-9)
        np.testing.assert_allclose(rates.numpy()[0], env.previous, atol=1e-8)
        for key, value in expected.items():
            assert value.item() == pytest.approx(info["reward_components"][key], abs=1e-9)
        np.testing.assert_allclose(
            observation(after, goal, rates, dt, new_q).numpy()[0], next_obs, atol=1e-6
        )
        record = env.replay.get_data()[0]
        assert record.intent_metadata["reference_floor"] == 0.0
        assert record.intent_metadata["reference_floor_power"] == power
        assert record.intent_metadata["reference_floor_weight"] == weight
        assert record.intent_metadata["worst_tracker_weight"] == worst_weight
        assert "reference_floor_intrusion" in record.learning.reward_components
        assert record.learning.reward_scope == "tracker_geometry"


def test_equivalent_hidden_joints_preserve_every_observed_tracker():
    rig, states = fixture()
    dynamics = DifferentiableRig(rig)
    roots = torch.tensor(np.stack([s.root for s in states]))
    q = torch.tensor(np.stack([s.rotations for s in states]))
    angles = torch.tensor(np.random.default_rng(8).uniform(-np.pi, np.pi, (2, len(rig.names))))
    changed = dynamics.equivalent_joints(q, angles)
    assert not torch.allclose(changed, q)
    np.testing.assert_allclose(
        pose_error(dynamics(roots, q), dynamics(roots, changed)).numpy(), 0, atol=1e-12
    )


def test_whole_body_heading_goal_preserves_pose_shape_height_and_root():
    from myumiq_vrchat.body import qmul, rotate

    rig, states = fixture()
    poses = np.stack([vector(rig.forward(s)) for s in states])
    angles = torch.tensor([0.7, -1.2], dtype=torch.float64, requires_grad=True)
    actual = turned_goal(torch.tensor(poses), angles)
    expected = poses.copy()
    for i, angle in enumerate(angles.detach().numpy()):
        turn = (np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2))
        pivot = poses[i, 2, :3]
        for j, row in enumerate(poses[i]):
            expected[i, j, :3] = pivot + rotate(turn, tuple(row[:3] - pivot))
            expected[i, j, 3:] = qmul(turn, tuple(row[3:]))
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12)
    torch.testing.assert_close(actual[:, :, 2], torch.tensor(poses[:, :, 2]), atol=1e-12, rtol=0)
    torch.testing.assert_close(actual[:, 2, :3], torch.tensor(poses[:, 2, :3]))
    torch.testing.assert_close(
        turned_goal(actual, -angles), torch.tensor(poses), atol=1e-12, rtol=1e-12
    )
    actual[:, :, :2].sum().backward()
    assert torch.isfinite(angles.grad).all() and angles.grad.abs().max() > 0


@pytest.mark.parametrize("worst_weight", [0.0, 1.0])
def test_model_gradient_is_finite_and_nonzero_from_stationary_action(worst_weight):
    rig, states = fixture()
    dynamics = DifferentiableRig(rig).double()
    root, q = torch.tensor(states[0].root[None]), torch.tensor(states[0].rotations[None])
    action = torch.zeros(1, rig.action_size, dtype=torch.float64, requires_grad=True)
    before = dynamics(root, q)
    goal = torch.tensor(vector(rig.forward(states[1]))[None])
    _, _, after, rates, _ = dynamics.step(root, q, action, 0.05)
    reward = sum(
        reward_components(
            before,
            after,
            goal,
            rates,
            torch.zeros_like(rates),
            0.0,
            worst_tracker_weight=worst_weight,
        ).values()
    ).mean()
    reward.backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().max() > 0.001
    np.testing.assert_allclose(after.detach().numpy(), before.numpy(), atol=1e-12)


def test_worst_tracker_objective_detects_localized_error_hidden_by_the_same_mean():
    rig, states = fixture()
    goal = torch.tensor(vector(rig.forward(states[0]))[None])
    diffuse, concentrated = goal.clone(), goal.clone()
    diffuse[:, :, 0] += 0.01
    concentrated[:, 0, 0] += 0.11
    rates = torch.zeros(1, 66, dtype=torch.float64)
    result = reward_components(concentrated, diffuse, goal, rates, rates, worst_tracker_weight=1.0)
    assert result["progress"].item() == pytest.approx(0.0, abs=1e-12)
    assert result["worst_tracker_progress"].item() == pytest.approx(1.0)
    assert worst_tracker_distance(concentrated, goal).item() == pytest.approx(0.11)
    # Angular and position errors use the unchanged acceptance ratio.
    angle = goal.clone()
    turn = angle.new_tensor([np.cos(0.35 / 2), 0.0, 0.0, np.sin(0.35 / 2)])
    from myumiq_vrchat.articulated_dynamics import multiply

    angle[:, 0, 3:] = multiply(turn[None], angle[:, 0, 3:])
    assert worst_tracker_distance(angle, goal).item() == pytest.approx(0.12)
    angle[:, :, 3:] *= -1
    assert worst_tracker_distance(angle, goal).item() == pytest.approx(0.12)
    assert worst_tracker_distance(goal, goal).item() == pytest.approx(0.0, abs=1e-12)
    for invalid in (-1.0, 11.0, float("nan")):
        with pytest.raises(ValueError, match="worst tracker"):
            ArticulatedGoalEnv(rig, states, worst_tracker_weight=invalid)


def test_offline_joint_teacher_matches_bounded_numpy_oracle_and_zero_goal():
    rig, states = fixture()
    root = torch.tensor(states[0].root[None])
    q = torch.tensor(states[0].rotations[None])
    target_root = torch.tensor(states[1].root[None])
    target_q = torch.tensor(states[1].rotations[None])
    expected = 2 * rig.rates_between(states[0], states[1], 1.0)
    expected /= max(1.0, np.linalg.norm(expected.reshape(-1, 3), axis=1).max())
    actual = joint_teacher(root, q, target_root, target_q)
    np.testing.assert_allclose(actual.numpy()[0], expected, atol=1e-10)
    torch.testing.assert_close(joint_teacher(root, q, root, -q), torch.zeros_like(actual))


def test_blended_training_goals_preserve_geometry_and_quaternion_sign_equivalence():
    rig, states = fixture()
    dynamics = DifferentiableRig(rig).double()
    root = torch.tensor(states[0].root[None])
    q = torch.tensor(states[0].rotations[None])
    other_root = torch.tensor(states[1].root[None])
    other_q = torch.tensor(states[1].rotations[None])
    fraction = torch.tensor([[0.6]], dtype=torch.float64)
    original = q.clone()
    a = blended_goal(dynamics, root, q, other_root, other_q, fraction)
    b = blended_goal(dynamics, root, -q, other_root, other_q, fraction)
    np.testing.assert_allclose(pose_error(a, b).numpy(), 0, atol=1e-12)
    assert float(a[:, 9:, 2].min()) >= 0.05 - 1e-12 and torch.equal(q, original)
    before = dynamics(root, q)
    for x, y in ((3, 5), (4, 6), (7, 9), (8, 10)):
        assert torch.linalg.vector_norm(a[:, x, :3] - a[:, y, :3]).item() == pytest.approx(
            torch.linalg.vector_norm(before[:, x, :3] - before[:, y, :3]).item(), abs=1e-12
        )


def test_policy_rollout_starts_follow_actor_and_preserve_previous_without_gradients():
    rig, states = fixture()
    dynamics = DifferentiableRig(rig).double()
    root = torch.tensor(states[0].root[None], requires_grad=True)
    q = torch.tensor(states[0].rotations[None], requires_grad=True)
    goal = dynamics(root, q).detach()
    previous = torch.zeros(1, 66, dtype=torch.float64)
    calls = []
    action = torch.zeros(1, rig.action_size, dtype=torch.float64)
    action[:, 0] = 0.2

    def actor(obs, deterministic):
        assert deterministic and not torch.is_grad_enabled()
        calls.append(obs.clone())
        return action

    next_root, next_q, rates = policy_rollout_start(
        actor, dynamics, root, q, goal, previous, 0.05, 3, 0.0
    )
    expected_root, expected_q = root, q
    for _ in range(3):
        expected_root, expected_q, _, expected_rates, _ = dynamics.step(
            expected_root, expected_q, action, 0.05
        )
    torch.testing.assert_close(next_root, expected_root)
    torch.testing.assert_close(next_q, expected_q)
    torch.testing.assert_close(rates, expected_rates)
    assert len(calls) == 3 and calls[1][0, 143:209].abs().max() > 0
    assert all(not x.requires_grad for x in (next_root, next_q, rates))
    torch.testing.assert_close(root.detach(), torch.tensor(states[0].root[None]))


def test_policy_rollout_start_stops_whole_body_before_floor_intrusion():
    rig, states = fixture()
    dynamics = DifferentiableRig(rig).double()
    root = torch.tensor(states[0].root[None])
    q = torch.tensor(states[0].rotations[None])
    current = dynamics(root, q)
    floor = float(current[:, 9:, 2].min())
    previous = torch.zeros(1, 66, dtype=torch.float64)
    action = torch.zeros(1, rig.action_size, dtype=torch.float64)
    action[:, 0], action[:, 2] = 0.3, -0.3
    result = policy_rollout_start(
        lambda *a, **kw: action, dynamics, root, q, current, previous, 0.05, 4, floor
    )
    for actual, expected in zip(result, (root, q, previous)):
        torch.testing.assert_close(actual, expected)


def test_replacing_rollout_goal_resets_rate_input_without_resetting_reached_body():
    rig, states = fixture()
    dynamics = DifferentiableRig(rig).double()
    root = torch.tensor(np.stack([s.root for s in states]))
    q = torch.tensor(np.stack([s.rotations for s in states]))
    previous = torch.zeros(2, 66, dtype=torch.float64)
    goal = dynamics(root, q)
    action = torch.zeros(2, rig.action_size, dtype=torch.float64)
    action[:, 0] = 0.2

    def actor(*args, **kwargs):
        return action

    unchanged = policy_rollout_start(actor, dynamics, root, q, goal, previous, 0.05, 3, 0.0)
    replaced = policy_rollout_start(
        actor,
        dynamics,
        root,
        q,
        goal,
        previous,
        0.05,
        3,
        0.0,
        reset_previous=torch.tensor([True, False]),
    )
    torch.testing.assert_close(replaced[0], unchanged[0])
    torch.testing.assert_close(replaced[1], unchanged[1])
    assert not torch.equal(replaced[0], root)
    torch.testing.assert_close(replaced[2][0], torch.zeros(66, dtype=torch.float64))
    torch.testing.assert_close(replaced[2][1], unchanged[2][1])
    assert unchanged[2][0].abs().max() > 0
    with pytest.raises(ValueError, match="replacement mask"):
        policy_rollout_start(
            actor,
            dynamics,
            root,
            q,
            goal,
            previous,
            0.05,
            3,
            0.0,
            reset_previous=torch.tensor([True]),
        )
