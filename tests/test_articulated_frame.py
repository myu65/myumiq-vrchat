# ruff: noqa: E402 -- optional learning dependencies
import numpy as np
import pytest

torch = pytest.importorskip("torch")
SAC = pytest.importorskip("stable_baselines3").SAC
from test_articulated_body import fixture

from myumiq_vrchat.articulated_body import JointState, articulated_observation
from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
from myumiq_vrchat.articulated_frame import FRAME, JOINT_WORLD_FRAME, canonical_observation
from myumiq_vrchat.articulated_policy import export_articulated, policy_for_frame
from myumiq_vrchat.body import qmul, rotate
from myumiq_vrchat.tracker_basis import RateBasis


def moved(state, yaw, shift):
    turn = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    q = state.rotations.copy()
    q[0] = qmul(turn, tuple(q[0]))
    return JointState(np.array(rotate(turn, tuple(state.root))) + shift, q), turn


@pytest.mark.parametrize("pitch", [0.0, np.pi / 2])
def test_common_frame_is_invariant_to_xy_yaw_and_quaternion_sign(pitch):
    rig, states = fixture()
    q = states[0].rotations.copy()
    q[0] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
    current = JointState(states[0].root, q)
    previous = np.random.default_rng(4).uniform(-0.2, 0.2, (22, 3))
    original = articulated_observation(
        rig.forward(current), rig.forward(states[1]), previous, 0.05, q
    )
    expected, _ = canonical_observation(torch.tensor(original[None]))
    for yaw in (0.8, -2.3):
        shift = np.array([2.0, -3.0, 0.0])
        other, turn = moved(current, yaw, shift)
        goal, _ = moved(states[1], yaw, shift)
        other = JointState(other.root, -other.rotations)
        rates = np.array([rotate(turn, tuple(v)) for v in previous])
        obs = articulated_observation(
            rig.forward(other), rig.forward(goal), rates, 0.05, other.rotations
        )
        actual, _ = canonical_observation(torch.tensor(obs[None]))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    value = torch.tensor(original[None], requires_grad=True)
    transformed, _ = canonical_observation(value)
    transformed.square().sum().backward()
    assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("frame", [FRAME, JOINT_WORLD_FRAME])
def test_canonical_actor_export_is_equivariant_bounded_and_holds_the_observed_goal(tmp_path, frame):
    from myumiq_vrchat.articulated_actor import ArticulatedActor

    torch.set_num_threads(1)
    rig, states = fixture()
    decoder = RateBasis(
        action_size=rig.action_size,
        rows=tuple(map(tuple, np.eye(rig.action_size))),
        variance_fraction=1.0,
    )
    env = ArticulatedGoalEnv(rig, states, decoder=decoder)
    policy, frame_kwargs = policy_for_frame(frame, rig)
    model = SAC(
        policy,
        env,
        buffer_size=10,
        device="cpu",
        policy_kwargs={"net_arch": [32, 32], **frame_kwargs},
    )
    path = tmp_path / "canonical.pt"
    export_articulated(model, path, rig, decoder)
    actor = ArticulatedActor(path)
    assert actor.manifest["policy_frame"] == frame and actor.manifest["net_arch"] == [32, 32]
    obs, _ = env.reset(options={"start": states[0], "goal": states[1]})
    expected = actor.predict(obs)[0]
    turn = None
    other, turn = moved(states[0], 1.3, np.array([3.0, -1.0, 0.0]))
    goal, _ = moved(states[1], 1.3, np.array([3.0, -1.0, 0.0]))
    changed = articulated_observation(
        rig.forward(other), rig.forward(goal), np.zeros(66), 0.05, other.rotations
    )
    actual = actor.predict(changed)[0]
    rotated = expected.copy()
    rotated[:3], rotated[3:6] = (
        rotate(turn, tuple(expected[:3])),
        rotate(turn, tuple(expected[3:6])),
    )
    np.testing.assert_allclose(actual, rotated, atol=2e-6)
    assert np.linalg.norm(actual.reshape(-1, 3), axis=1).max() <= 1.000001
    np.testing.assert_allclose(actual, model.predict(changed, deterministic=True)[0], atol=1e-6)
    held = articulated_observation(
        rig.forward(other), rig.forward(other), np.ones(66) * 0.2, 0.05, other.rotations
    )
    np.testing.assert_array_equal(actor.predict(held)[0], np.zeros(rig.action_size))
    values = torch.tensor(obs[None], requires_grad=True)
    model.actor(values, deterministic=True).square().sum().backward()
    assert torch.isfinite(values.grad).all()
    with pytest.raises(ValueError, match="deterministic"):
        model.actor(torch.tensor(obs[None]))
    with pytest.raises(ValueError, match="stochastic"):
        model.actor.action_log_prob(torch.tensor(obs[None]))
    env.close()


def test_joint_world_actor_preserves_physical_action_under_hidden_joint_gauge():
    from myumiq_vrchat.articulated_dynamics import DifferentiableRig, observation, pose_error

    rig, states = fixture()
    dynamics = DifferentiableRig(rig).float()
    env = ArticulatedGoalEnv(
        rig,
        states,
        decoder=RateBasis(
            action_size=rig.action_size,
            rows=tuple(map(tuple, np.eye(rig.action_size))),
            variance_fraction=1.0,
        ),
    )
    policy, kwargs = policy_for_frame(JOINT_WORLD_FRAME, rig)
    model = SAC(
        policy, env, buffer_size=10, device="cpu", policy_kwargs={"net_arch": [32, 32], **kwargs}
    )
    root = torch.tensor(np.stack([s.root for s in states]), dtype=torch.float32)
    q = torch.tensor(np.stack([s.rotations for s in states]), dtype=torch.float32)
    changed = dynamics.equivalent_joints(q, torch.randn(q.shape[:2]) * 2)
    assert not torch.allclose(q, changed)
    before, goal = dynamics(root, q), dynamics(root.flip(0), q.flip(0))
    previous = torch.randn(2, 66) * 0.1
    for dt in (0.02, 0.05, 0.1):
        original = observation(before, goal, previous, dt, q)
        other = observation(dynamics(root, changed), goal, previous, dt, changed)
        a = model.actor(original, deterministic=True)
        b = model.actor(other, deterministic=True)
        after_a = dynamics.step(root, q, a, dt)[2]
        after_b = dynamics.step(root, changed, b, dt)[2]
        torch.testing.assert_close(
            pose_error(after_a, after_b), torch.zeros(2, 11, 6), atol=1e-6, rtol=0
        )
    env.close()


def test_joint_world_frame_rejects_a_nonroot_pelvis():
    from myumiq_vrchat.articulated_frame import JointWorldFrame

    rig, _ = fixture()
    nodes = list(rig.orientation_nodes)
    nodes[2] = 1
    with pytest.raises(ValueError, match="pelvis"):
        JointWorldFrame(rig.model_copy(update={"orientation_nodes": tuple(nodes)}))


def test_teacher_tracker_loss_uses_recorded_timestep_and_has_finite_student_gradients(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from pretrain_articulated_prior import tracker_loss

    from myumiq_vrchat.articulated_dynamics import DifferentiableRig, joint_teacher

    rig, states = fixture()
    dynamics = DifferentiableRig(rig).float()
    obs = torch.tensor(
        np.stack(
            [
                articulated_observation(
                    rig.forward(states[0]),
                    rig.forward(states[1]),
                    np.zeros(66),
                    dt,
                    states[0].rotations,
                )
                for dt in (0.02, 0.05, 0.1)
            ]
        )
    )
    root = torch.tensor(states[0].root[None], dtype=torch.float32)
    q = torch.tensor(states[0].rotations[None], dtype=torch.float32)
    target_root = torch.tensor(states[1].root[None], dtype=torch.float32)
    target_q = torch.tensor(states[1].rotations[None], dtype=torch.float32)
    labels = joint_teacher(root, q, target_root, target_q).expand(3, -1)
    prediction = torch.zeros_like(labels, requires_grad=True)
    loss = tracker_loss(dynamics, obs, prediction, labels)
    loss.backward()
    assert (
        loss.item() > 0
        and torch.isfinite(prediction.grad).all()
        and prediction.grad.abs().max() > 0
    )
    matched = labels.clone().requires_grad_()
    loss = tracker_loss(dynamics, obs, matched, labels)
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(matched.grad).all()
    obs[0, 209] = 0.123
    with pytest.raises(ValueError, match="timestep"):
        tracker_loss(dynamics, obs, prediction, labels)


def test_dataset_aggregation_observes_student_motion_and_keeps_teacher_labels(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from pretrain_articulated_prior import collect

    from myumiq_vrchat.articulated_dynamics import DifferentiableRig

    rig, states = fixture()
    dynamics = DifferentiableRig(rig).float()
    roots = torch.tensor(np.stack([s.root for s in states]), dtype=torch.float32)
    joints = torch.tensor(np.stack([s.rotations for s in states]), dtype=torch.float32)

    def predict(obs):
        action = obs.new_zeros((len(obs), rig.action_size))
        action[:, 0] = 0.1
        return action

    torch.manual_seed(71)
    obs, labels, stats = collect(
        dynamics,
        roots,
        joints,
        4096,
        np.random.default_rng(71),
        lambda _: None,
        rollin_actor=SimpleNamespace(model=predict),
        rollin_probability=1.0,
        goal_steps=64,
    )
    assert stats["rejected_floor_actions"] == 0 and stats["rejected_student_floor_actions"] == 0
    assert stats["student_rollin_actions"] == 4096 and stats["teacher_goal_steps"] == 64
    torch.testing.assert_close(obs[:, 209].unique(), torch.tensor([0.4, 1.0, 2.0]))
    assert obs[1024, 209] != obs[0, 209]
    dt = obs[0, 209] * 0.05
    torch.testing.assert_close(
        obs[128:256, 80] - obs[:128, 80], torch.full((128,), 0.6 * 0.1 * dt), atol=1e-7, rtol=1e-5
    )
    previous = obs[128:, 143:209].reshape(-1, 11, 6)
    torch.testing.assert_close(
        previous[:, :, 0], torch.full_like(previous[:, :, 0], 0.1), atol=1e-5, rtol=1e-4
    )
    assert not torch.allclose(labels, predict(obs))
    assert torch.isfinite(obs).all() and torch.isfinite(labels).all()
