# ruff: noqa: E402 -- optional learning dependencies
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
SAC = pytest.importorskip("stable_baselines3").SAC
from pamiq_core.data import DataUsersDict
from pamiq_core.data.impls import SequentialBuffer
from pamiq_core.model import TrainingModelsDict
from pamiq_core.torch import TorchTrainingModel
from test_articulated_body import fixture

from myumiq_vrchat.articulated_dynamics import DifferentiableRig
from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
from myumiq_vrchat.articulated_experience import (
    ExperienceCase,
    experience_case,
    load_experience_cases,
    require_disjoint_sessions,
)
from myumiq_vrchat.articulated_policy import AnchoredSACPolicy
from myumiq_vrchat.experience_refinement import (
    ExperienceRefinementTrainer,
    eligible_candidate,
    nonfoot_floor_penalty,
    preserve_observed_residual,
    reference_starts,
)
from myumiq_vrchat.tracker_action import integrate_tracker_action
from myumiq_vrchat.tracker_basis import RateBasis
from myumiq_vrchat.whole_body import target_from_vector, vector


def test_replay_import_requires_actual_confirmed_compatible_feedback(tmp_path):
    rig, states = fixture()
    decoder = RateBasis(
        action_size=rig.action_size,
        variance_fraction=1.0,
        rows=tuple(map(tuple, np.eye(rig.action_size))),
    )
    actor = SimpleNamespace(rig=rig, decoder=decoder, manifest={"sha256": "b" * 64})
    env = ArticulatedGoalEnv(rig, states, decoder=decoder, recording=True)
    env.reset(options={"start": states[0], "goal": states[1]})
    env.step(np.zeros(rig.action_size))
    simulated = env.replay.get_data()[0]
    with pytest.raises(ValueError, match="unconfirmed"):
        experience_case(simulated, actor)

    def device(observation):
        body = observation.body
        signals = {
            k: getattr(body, k).model_copy(update={"source": "openvr_raw"})
            for k in type(body).model_fields
            if hasattr(getattr(body, k), "source")
        }
        return observation.model_copy(update={"body": body.model_copy(update=signals)})

    record = simulated.model_copy(
        update={
            "environment": "vrchat",
            "outcome": "device_feedback",
            "policy_id": "articulated:" + "b" * 16,
            "observation": device(simulated.observation),
            "next_observation": device(simulated.next_observation),
            "learning": simulated.learning.model_copy(
                update={"evidence_ids": ("session-a:readback:1",)}
            ),
            "intent_metadata": {
                **simulated.intent_metadata,
                "joint_state_source": "fitted_tracker_estimate",
                "intent_generation": 7,
                "pose_goal": env.goal.model_dump(mode="json"),
                "previous_rates": [0.0] * 66,
                "expected_tracker_pose": simulated.action.model_dump(mode="json"),
            },
        }
    )
    case = experience_case(record, actor)
    assert case.session == "session-a" and case.action == "session-a:7"
    path = tmp_path / "replay.jsonl"
    path.write_text(
        "\n".join(r.model_dump_json() for r in (record, simulated, record)) + "\n", "utf-8"
    )
    cases, report = load_experience_cases(path, actor)
    assert len(cases) == 1 and report["confirmed"] == 1 and report["rejected"]["ValueError"] == 1
    assert report["rejected"]["duplicate_readback"] == 1
    bad = record.model_copy(
        update={"intent_metadata": {**record.intent_metadata, "decoder_id": "different"}}
    )
    with pytest.raises(ValueError, match="incompatible"):
        experience_case(bad, actor)
    next_body = record.next_observation.body
    signal = next_body.head.model_copy(
        update={"pose": next_body.head.pose.model_copy(update={"position": (8.0, 0.0, 1.0)})}
    )
    bad = record.model_copy(
        update={
            "next_observation": record.next_observation.model_copy(
                update={"body": next_body.model_copy(update={"head": signal})}
            )
        }
    )
    with pytest.raises(ValueError, match="does not confirm"):
        experience_case(bad, actor)
    env.close()


def test_predicted_increments_preserve_observed_tracker_residuals():
    rig, states = fixture()
    state = states[0]
    current = vector(rig.forward(state))
    current[:, 0] += 0.003
    current[:, 3:] = np.roll(current[:, 3:], 1, axis=1)  # Different observed orientation.
    dynamics = DifferentiableRig(rig).double()
    root, q = torch.tensor(state.root[None]), torch.tensor(state.rotations[None])
    before = dynamics(root, q)
    action = np.random.default_rng(9).uniform(-0.2, 0.2, rig.action_size)
    _, _, after, rates, _ = dynamics.step(root, q, torch.tensor(action[None]), 0.05)
    observed = preserve_observed_residual(before, after, torch.tensor(current[None]))
    actual = integrate_tracker_action(
        target_from_vector(current), rates.numpy()[0].reshape(11, 6), 0.05
    )
    np.testing.assert_allclose(observed.numpy()[0], vector(actual), atol=1e-8)
    np.testing.assert_allclose(
        preserve_observed_residual(before, before, torch.tensor(current[None])).numpy()[0],
        current,
        atol=1e-12,
    )


def test_reference_evaluation_holds_completed_static_goals_and_restarts_settling():
    from myumiq_vrchat.articulated_evaluation import evaluate

    rig, states = fixture()

    class DriftingActor:
        calls = 0

        def predict(self, obs, deterministic=True):
            self.calls += 1
            action = np.zeros(rig.action_size)
            action[2] = -0.1
            return action, None

    raw_actor, finite_actor = DriftingActor(), DriftingActor()
    raw, _ = evaluate(raw_actor, rig, None, [states[0], states[0]], reference_floor=0.0)
    finite, _ = evaluate(
        finite_actor, rig, None, [states[0], states[0]], reference_floor=0.0, settle_on_goal=True
    )
    assert raw["minimum_foot_height_m"] < 0
    assert finite["minimum_foot_height_m"] > 0
    assert finite["control_contract"] == "static_goal_settling_v1"
    assert finite["static_goals_reached"] == 12
    assert finite_actor.calls == 6 * 2 * 3  # three moving samples, then 150 ms settles.
    assert raw_actor.calls == 6 * 100


@pytest.mark.parametrize("rehearsal", [False, True])
@pytest.mark.parametrize("whole_body_floor", [False, True])
def test_pamiq_candidate_updates_are_isolated_and_budgeted(tmp_path, rehearsal, whole_body_floor):
    torch.set_num_threads(1)
    torch.manual_seed(43)
    rig, states = fixture()
    env = ArticulatedGoalEnv(rig, states)
    model = SAC(
        AnchoredSACPolicy, env, buffer_size=10, device="cpu", policy_kwargs={"net_arch": [16, 16]}
    )
    initial = {k: v.clone() for k, v in model.actor.state_dict().items()}
    case = ExperienceCase(
        "train",
        "train:1",
        "readback:1",
        states[0].root,
        states[0].rotations,
        vector(rig.forward(states[0])),
        vector(rig.forward(states[1])),
        np.zeros(66),
        0.05,
    )
    buffer = SequentialBuffer(4)
    buffer.add(case)
    models = TrainingModelsDict(
        {"candidate": TorchTrainingModel(model.actor, has_inference_model=False)}
    )
    trainer = ExperienceRefinementTrainer(
        rig,
        updates=2,
        horizon=1,
        batch_size=2,
        reference_rehearsal=rehearsal,
        whole_body_floor=whole_body_floor,
    )
    trainer.attach_training_models(models)
    buffers = {"experience": buffer}
    if rehearsal:
        references = reference_starts(rig, states, 0.0, count=8)
        assert all(c.current[9:, 2].min() >= 0.019999 for c in references)
        assert all(not hasattr(c, "session") and not hasattr(c, "evidence") for c in references)
        buffers["reference"] = SequentialBuffer(8)
        for reference in references:
            buffers["reference"].add(reference)
    trainer.attach_data_users(DataUsersDict.from_data_buffers(buffers))
    assert not models.inference_models_dict
    assert trainer.run() and not trainer.run()
    assert trainer.total_updates == 2 and np.isfinite(trainer.losses).all()
    assert any(not torch.equal(initial[k], v) for k, v in model.actor.state_dict().items())
    for key, value in trainer.prior.state_dict().items():
        torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
    assert trainer.optimizers["actor"].state
    calls = []

    def accept(metrics):
        calls.append(metrics)
        return len(calls) == 2

    selected, _ = trainer.select_step(lambda: {}, accept, 2)
    assert selected["scale"] == 0.5 and not trainer.optimizers["actor"].state
    trainer.save_state(tmp_path / "trainer")
    # PAMIQ caches optimizer state at the end of run(); clearing the live Adam
    # alone must not accidentally preserve the rejected full-step moments.
    assert not torch.load(tmp_path / "trainer/actor.optim.pt", weights_only=True)["state"]
    bad = ExperienceRefinementTrainer(rig, updates=1)
    with pytest.raises(ValueError, match="linked inference"):
        bad.attach_training_models(
            TrainingModelsDict({"candidate": TorchTrainingModel(torch.nn.Linear(1, 1))})
        )
    with pytest.raises(ValueError, match="independent replay sessions"):
        require_disjoint_sessions([case], [replace(case, action="train:2")])
    require_disjoint_sessions([case], [replace(case, session="heldout")])
    env.close()


def test_nonfoot_clearance_detects_and_trains_low_hand_with_feet_clear():
    trackers = torch.zeros(2, 11, 7)
    trackers[:, :, 2] = 0.1
    trackers[0, 3, 2] = -0.03
    trackers.requires_grad_()
    penalty = nonfoot_floor_penalty(trackers, 0.0)
    torch.testing.assert_close(penalty, torch.tensor([0.05, 0.0]))
    penalty.sum().backward()
    assert trackers.grad[0, 3, 2] < 0  # descent pushes the low hand upward
    assert trackers.grad[:, 9:, :].abs().sum() == 0  # separate foot objective


@pytest.mark.parametrize(
    "regression", ["none", "floor", "endpoints", "length", "replay", "contract", "settling"]
)
def test_average_improvement_cannot_hide_constraint_or_reference_regression(regression):
    before = dict(floor_blocked=0, reached=20, mean_endpoint_ratio=0.8)
    after = dict(floor_blocked=0, reached=20, mean_endpoint_ratio=0.7)
    reference_before = dict(
        mean_endpoint_error=0.1,
        endpoints_worse_than_hold=0,
        control_contract="static_goal_settling_v1",
        static_goals_reached=12,
    )
    reference_after = dict(
        mean_endpoint_error=0.09,
        endpoints_worse_than_hold=0,
        minimum_foot_height_m=0.01,
        maximum_forearm_shin_length_change_m=1e-12,
        control_contract="static_goal_settling_v1",
        static_goals_reached=12,
    )
    if regression == "contract":
        reference_after["control_contract"] = "continuous_actor_v1"
    if regression == "settling":
        reference_after["static_goals_reached"] = 11
    if regression == "floor":
        reference_after["minimum_foot_height_m"] = -0.0001
    if regression == "endpoints":
        reference_after["endpoints_worse_than_hold"] = 1
    if regression == "length":
        reference_after["maximum_forearm_shin_length_change_m"] = 0.01
    if regression == "replay":
        after["floor_blocked"] = 1
    assert eligible_candidate(before, after, reference_before, reference_after, 0.0) is (
        regression == "none"
    )
