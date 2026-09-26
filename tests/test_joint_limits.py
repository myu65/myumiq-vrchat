# ruff: noqa: E402 -- optional training dependencies
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")
from test_articulated_body import fixture

from myumiq_vrchat.articulated_body import ArticulatedRig, JointState
from myumiq_vrchat.articulated_dynamics import DifferentiableRig
from myumiq_vrchat.articulated_fit import fit_body
from myumiq_vrchat.joint_limits import JointLimit, from_reference, project, quaternion, violation
from myumiq_vrchat.whole_body import vector


def bounded_fixture():
    rig, states = fixture()
    limit = JointLimit(lower=(-0.12, -0.08, -0.04), upper=(0.3, 0.2, 0.1), max_angle=0.31)
    rig = rig.model_copy(update={"joint_limits": (None,) + (limit,) * (len(rig.names) - 1)})
    return rig, states


def test_limits_are_asymmetric_local_and_sign_invariant_without_restricting_root():
    rig, states = bounded_fixture()
    q = states[0].rotations.copy()
    q[0] = quaternion(np.array([0.0, 2.1, 0.0]))  # Whole body lying/turning is allowed.
    q[4] = quaternion(np.array([0.0, -0.5, 0.0]))
    before = q.copy()
    fixed = project(q, rig.joint_limits)
    assert violation(q, rig.joint_limits)[4] == pytest.approx(0.42)
    assert violation(fixed, rig.joint_limits).max() < 1e-12
    np.testing.assert_array_equal(fixed[0], q[0])
    np.testing.assert_array_equal(q, before)
    tilted = JointState(states[0].root, fixed)
    _, held, _, _ = rig.advance(tilted, np.zeros(rig.action_size), 0.05)
    np.testing.assert_allclose(vector(held), vector(rig.forward(tilted)), atol=1e-12)
    np.testing.assert_allclose(
        violation(-q, rig.joint_limits), violation(q, rig.joint_limits), atol=1e-12
    )
    np.testing.assert_allclose(
        np.abs(np.sum(project(-q, rig.joint_limits) * fixed, axis=-1)), 1.0, atol=1e-12
    )
    # Zero command cannot silently repair an invalid observed/estimated start.
    with pytest.raises(ValueError, match="joint envelope"):
        rig.advance(JointState(states[0].root, q), np.zeros(rig.action_size), 0.05)


@pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
def test_long_adversarial_rollout_is_bounded_identical_in_training_and_execution(dt):
    torch.set_num_threads(1)
    rig, states = bounded_fixture()
    state = states[0]
    dynamics = DifferentiableRig(rig).double()
    root, q = torch.tensor(state.root[None]), torch.tensor(state.rotations[None])
    rng = np.random.default_rng(42)
    for index in range(130):
        action = np.ones(rig.action_size) if index < 80 else rng.uniform(-1, 1, rig.action_size)
        state, target, rates, scale = rig.advance(state, action, dt)
        root, q, actual, actual_rates, actual_scale = dynamics.step(
            root, q, torch.tensor(action[None]), dt
        )
        assert violation(state.rotations, rig.joint_limits).max() < 1e-10
        assert np.linalg.norm(rates.reshape(-1, 3), axis=-1).max() <= 1 + 1e-10
        np.testing.assert_allclose(vector(target), actual.numpy()[0], atol=1e-9)
        np.testing.assert_allclose(rates, actual_rates.numpy()[0], atol=1e-9)
        assert scale == pytest.approx(actual_scale.item(), abs=1e-9)
    held = vector(rig.forward(state))
    for _ in range(15):
        state, target, _, _ = rig.advance(state, np.zeros(rig.action_size), dt)
    np.testing.assert_allclose(vector(target), held, atol=1e-12)
    # At a boundary, the actor can still move away from the limit.
    away = -rig.rates_between(states[0], state, 1.0)
    away /= max(1.0, np.abs(away).max())
    moved, _, _, _ = rig.advance(state, away, dt)
    assert not np.allclose(moved.rotations, state.rotations)


def test_constraint_gradients_are_finite_at_rest_and_limits_and_allow_learning():
    rig, states = bounded_fixture()
    dynamics = DifferentiableRig(rig).double()
    for value in (0.0, 0.8):
        q = torch.tensor(states[0].rotations[None])
        root = torch.tensor(states[0].root[None])
        action = torch.full((1, rig.action_size), value, dtype=torch.float64, requires_grad=True)
        *_, target, rates, _ = dynamics.step(root, q, action, 0.1)
        loss = (target[..., :3] - 0.3).square().sum() + (rates - 0.1).square().mean()
        loss.backward()
        assert torch.isfinite(action.grad).all()
        assert action.grad.abs().max() > 0


def test_constrained_fit_accepts_legal_pose_and_rejects_observable_backwards_joint():
    torch.set_num_threads(1)
    rig, states = bounded_fixture()
    good = rig.forward(states[1])
    result, report = fit_body(rig, good, states[0])
    assert violation(result.rotations, rig.joint_limits).max() <= 1e-6
    assert report["joint_constraint_contract"]
    q = states[0].rotations.copy()
    q[2] = quaternion(np.array([0.0, -1.0, 0.0]))  # Directly observed head joint in this fixture.
    bad = rig.forward(JointState(states[0].root, q))
    original = vector(bad).copy()
    with pytest.raises(ValueError, match="does not fit calibrated rig|joint envelope"):
        fit_body(rig, bad, states[0], max_iter=80)
    np.testing.assert_array_equal(vector(bad), original)


@pytest.mark.parametrize("angles", [(0.28, 0.02, 0.02), (-0.11, 0.12, 0.06), (0.01, -0.07, -0.03)])
def test_fitting_can_reach_legal_boundary_neighbourhood_from_rest(angles):
    rig, states = bounded_fixture()
    q = np.tile(quaternion(np.array(angles)), (len(rig.names), 1))
    target = rig.forward(JointState(states[0].root, q))
    fitted, report = fit_body(rig, target, states[0], max_iter=300)
    assert report["maximum_position_error_m"] <= 0.005
    assert report["maximum_rotation_error_rad"] <= 0.02
    assert violation(fitted.rotations, rig.joint_limits).max() < 1e-12


def test_hidden_twist_augmentation_preserves_trackers_and_rejects_invalid_solutions():
    rig, states = bounded_fixture()
    # Joint 3 has a single child and its orientation is not directly observed.
    assert 3 not in rig.orientation_nodes
    dynamics = DifferentiableRig(rig).double()
    q = torch.tensor(states[0].rotations[None])
    angles = torch.zeros(1, len(rig.names), dtype=torch.float64)
    angles[0, 3] = 2.5
    proposed = dynamics.equivalent_joints(q, angles)
    torch.testing.assert_close(proposed, q, atol=0, rtol=0)


def test_reference_fit_serialization_and_legacy_identity():
    rig, states = fixture()
    old = rig.model_dump_json()
    assert "joint_limits" not in json.loads(old)
    limits = from_reference(np.stack([s.rotations for s in states]), margin=0.1)
    constrained = ArticulatedRig.model_validate_json(
        json.dumps(
            rig.model_dump(mode="json")
            | {"joint_limits": [v.model_dump(mode="json") if v else None for v in limits]}
        )
    )
    assert ArticulatedRig.model_validate_json(constrained.model_dump_json()) == constrained
    assert (
        hashlib.sha256(constrained.model_dump_json().encode()).digest()
        != hashlib.sha256(old.encode()).digest()
    )
    for state in states:
        constrained.validate_limits(state)
    with pytest.raises(ValueError, match="invalid articulated rig"):
        ArticulatedRig.model_validate_json(
            json.dumps(json.loads(constrained.model_dump_json()) | {"joint_limits": [None]})
        )


def test_constraint_identity_is_required_by_exported_actor(tmp_path):
    from stable_baselines3 import SAC

    from myumiq_vrchat.articulated_actor import ArticulatedActor
    from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
    from myumiq_vrchat.articulated_policy import AnchoredSACPolicy, export_articulated
    from myumiq_vrchat.tracker_basis import RateBasis

    rig, states = bounded_fixture()
    decoder = RateBasis(
        action_size=rig.action_size,
        rows=tuple(map(tuple, np.eye(rig.action_size))),
        variance_fraction=1.0,
    )
    env = ArticulatedGoalEnv(rig, states, decoder=decoder)
    model = SAC(
        AnchoredSACPolicy, env, device="cpu", buffer_size=10, policy_kwargs={"net_arch": [16, 16]}
    )
    path = tmp_path / "candidate.pt"
    export_articulated(model, path, rig, decoder)
    loaded = ArticulatedActor(path)
    assert loaded.rig == rig
    manifest = json.loads(path.with_suffix(".json").read_text("utf-8"))
    manifest["rig"]["joint_limits"][1]["max_angle"] += 0.01
    path.with_suffix(".json").write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(ValueError, match="constraint contract or rig digest"):
        ArticulatedActor(path)
    env.close()


def test_unreachable_goal_does_not_authorize_an_invalid_start_or_joint_output():
    from myumiq_vrchat.articulated_env import ArticulatedGoalEnv

    rig, states = bounded_fixture()
    q = states[0].rotations.copy()
    q[2] = quaternion(np.array([0.0, -1.0, 0.0]))
    bad = JointState(states[0].root, q)
    env = ArticulatedGoalEnv(rig, [states[0], bad], recording=True)
    with pytest.raises(ValueError, match="joint envelope"):
        env.reset(options={"start": bad, "goal": states[0]})
    env.reset(options={"start": states[0], "goal": bad})
    for _ in range(20):
        env.step(np.ones(rig.action_size))
    assert violation(env.state.rotations, rig.joint_limits).max() < 1e-6
    record = env.replay.get_data()[-1]
    assert record.intent_metadata["joint_constraint_contract"]
    assert not np.allclose(
        record.intent_metadata["realized_joint_action"],
        np.array(record.intent_metadata["joint_action"])
        * record.intent_metadata["joint_action_scale"],
    )
    env.close()


def test_teacher_includes_whole_body_heading_changes_under_the_same_limits(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from pretrain_articulated_prior import collect

    torch.set_num_threads(1)
    torch.manual_seed(17)
    rig, states = bounded_fixture()
    dynamics = DifferentiableRig(rig).float()
    roots = torch.tensor(np.stack([s.root for s in states]), dtype=torch.float32)
    joints = torch.tensor(np.stack([s.rotations for s in states]), dtype=torch.float32)
    obs, labels, report = collect(
        dynamics,
        roots,
        joints,
        512,
        np.random.default_rng(17),
        lambda row: None,
        heading_goal_range=1.5,
    )
    root_yaw_error = obs[:, :66].reshape(-1, 11, 6)[:, 2, 5] * 2
    turning = root_yaw_error.abs() > 0.5
    assert int(turning.sum()) > 20
    assert (root_yaw_error[turning] * labels[turning, 5] > 0).all()
    q = obs[:, 210:].reshape(-1, len(rig.names), 4)
    assert dynamics.envelope.violation(q).max() <= 1e-6
    assert report["heading_goal_range"] == 1.5
