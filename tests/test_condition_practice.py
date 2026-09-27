# ruff: noqa: E402 -- optional learning dependencies
import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from myumiq_vrchat.condition_objective import ConditionObjective, ConditionStart, condition_distance
from myumiq_vrchat.condition_training import admissible
from myumiq_vrchat.motion_quality import trajectory_quality


def test_original_masked_conditions_have_gradient_but_unrequested_axes_do_not():
    current = torch.zeros(1, 11, 7)
    current[..., 3] = 1
    desired = current.clone()
    desired[0, 0, :3] = torch.tensor([1.0, 1.0, -0.15])
    desired[0, 0, 3:] *= -1  # quaternion sign is not a rotation change
    scale = torch.zeros(1, 11, 6)
    scale[0, 0, 2] = 25
    current.requires_grad_()
    distance = condition_distance(current, desired, scale)
    assert distance.item() == pytest.approx(3.75)
    distance.sum().backward()
    assert current.grad[0, 0, 2] > 0
    assert current.grad[0, 0, :2].abs().sum() == 0
    assert current.grad[..., 3:].abs().sum() == 0


def test_trajectory_reports_foot_slide_and_abrupt_stop_instead_of_only_endpoint():
    dt = 0.05
    steady = np.zeros((8, 11, 3))
    steady[:, :, 2] = 0.1
    steady[:, 3, 0] = np.arange(8) * dt * 0.2
    motion = trajectory_quality(steady, dt)
    assert motion["maximum_speed_m_s"] == pytest.approx(0.2)
    assert motion["maximum_acceleration_m_s2"] < 1e-12
    stopped = trajectory_quality(np.concatenate([steady, steady[-1:]]), dt)
    assert stopped["maximum_acceleration_m_s2"] == pytest.approx(4)
    steady[-1, 9, 0] = 0.1
    assert trajectory_quality(steady, dt)["maximum_foot_displacement_m"] == 0.1
    with pytest.raises(ValueError):
        trajectory_quality(steady, float("nan"))


def test_condition_selection_rejects_endpoint_gain_with_new_trajectory_defect():
    quality = dict(
        maximum_foot_displacement_m=0.01, maximum_acceleration_m_s2=1.0, maximum_jerk_m_s3=10.0
    )
    before = [
        dict(
            id="unseen",
            accepted=True,
            floor_failure=False,
            condition_ratio=0.9,
            motion_quality=quality,
        )
    ]
    after = deepcopy(before)
    after[0]["condition_ratio"] = 0.7
    reference = dict(
        static_goals_reached=4,
        minimum_foot_height_m=0.01,
        mean_endpoint_error=0.03,
        endpoints_worse_than_hold=0,
    )
    assert admissible(before, after, reference, reference)
    old_reference = reference | {
        "trials": [{"seed": 0, "endpoints": [{"settled": True}, {"settled": False}]}]
    }
    new_reference = reference | {
        "trials": [{"seed": 0, "endpoints": [{"settled": False}, {"settled": True}]}]
    }
    assert not admissible(before, after, old_reference, new_reference)
    for key in quality:
        defective = deepcopy(after)
        defective[0]["motion_quality"][key] = 100
        assert not admissible(before, defective, reference, reference)
    after[0]["floor_failure"] = True
    assert not admissible(before, after, reference, reference)


def test_condition_practice_updates_isolated_pamiq_candidate():
    from pamiq_core.data import DataUsersDict
    from pamiq_core.data.impls import SequentialBuffer
    from pamiq_core.model import TrainingModelsDict
    from pamiq_core.torch import TorchTrainingModel
    from stable_baselines3 import SAC
    from test_articulated_body import fixture

    from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
    from myumiq_vrchat.articulated_policy import AnchoredSACPolicy
    from myumiq_vrchat.experience_refinement import ExperienceRefinementTrainer
    from myumiq_vrchat.whole_body import vector

    torch.set_num_threads(1)
    rig, states = fixture()
    current, goal = (vector(rig.forward(s)) for s in states)
    scale = np.zeros((11, 6))
    scale[0, 2] = 25
    case = ConditionStart(
        states[0].root, states[0].rotations, current, goal, np.zeros(66), 0.05, goal, scale
    )
    data = SequentialBuffer(2)
    data.add(case)
    data.add(replace(case, current=goal, desired=current))
    env = ArticulatedGoalEnv(rig, states)
    model = SAC(
        AnchoredSACPolicy, env, buffer_size=10, device="cpu", policy_kwargs={"net_arch": [16]}
    )
    initial = deepcopy(model.actor.state_dict())
    trainer = ExperienceRefinementTrainer(
        rig,
        updates=2,
        horizon=2,
        batch_size=2,
        whole_body_floor=True,
        extra_objective_factory=ConditionObjective,
        rollout_start_steps=3,
    )
    models = TrainingModelsDict(
        {"candidate": TorchTrainingModel(model.actor, has_inference_model=False)}
    )
    trainer.attach_training_models(models)
    trainer.attach_data_users(DataUsersDict.from_data_buffers({"experience": data}))
    assert trainer.run() and trainer.total_updates == 2
    assert not models.inference_models_dict
    assert np.isfinite(trainer.losses).all()
    assert any(
        not torch.equal(value, model.actor.state_dict()[key]) for key, value in initial.items()
    )
    for key, value in trainer.prior.state_dict().items():
        torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
    env.close()


def test_practice_cli_loads_strict_json_and_evaluates_deterministically(tmp_path, monkeypatch):
    from stable_baselines3 import SAC
    from test_articulated_body import fixture

    from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
    from myumiq_vrchat.articulated_policy import AnchoredSACPolicy, export_articulated
    from myumiq_vrchat.condition_training import main
    from myumiq_vrchat.tracker_basis import RateBasis

    torch.set_num_threads(1)
    rig, states = fixture()
    decoder = RateBasis(
        action_size=rig.action_size,
        variance_fraction=1.0,
        rows=tuple(map(tuple, np.eye(rig.action_size))),
    )
    env = ArticulatedGoalEnv(rig, states, decoder=decoder)
    model = SAC(
        AnchoredSACPolicy, env, buffer_size=10, device="cpu", policy_kwargs={"net_arch": [16]}
    )
    model.reference_floor = 0.0
    prior = tmp_path / "prior"
    prior.mkdir()
    export_articulated(model, prior / "candidate-actor.pt", rig, decoder)
    torch.save({"actor": model.actor.state_dict()}, prior / "training-state.pt")
    tasks = dict(
        actor=str(prior / "candidate-actor.pt"),
        actor_sha256=hashlib.sha256((prior / "candidate-actor.pt").read_bytes()).hexdigest(),
        rig_sha256=hashlib.sha256(rig.model_dump_json().encode()).hexdigest(),
        reference_floor=0.0,
        goals={"STAND": rig.forward(states[0]).model_dump(mode="json")},
    )
    task_path = tmp_path / "tasks.json"
    task_path.write_text(json.dumps(tasks), "utf-8")
    cases = [
        dict(
            id=split,
            start="STAND",
            split=split,
            goal=dict(
                duration_s=1.0,
                conditions=[
                    dict(
                        part="head",
                        frame="current",
                        position=[None, None, z],
                        position_tolerance=0.04,
                    )
                ],
            ),
        )
        for split, z in (("train", 0.01), ("heldout", 0.02))
    ]
    case_path = tmp_path / "cases.json"
    # A measured full-body start must not be registered as a runtime skill.
    cases[1].pop("start")
    cases[1]["start_pose"] = rig.forward(states[0]).model_dump(mode="json")
    case_path.write_text(json.dumps(cases), "utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "result.json").write_text('{"license":"CC0"}', "utf-8")
    (corpus / "rig.json").write_text(rig.model_dump_json(), "utf-8")
    poses = [
        dict(
            split=split,
            joint_state=dict(root=s.root.tolist(), local_orientations=s.rotations.tolist()),
        )
        for split in ("train", "heldout")
        for s in states
    ]
    (corpus / "poses.json").write_text(json.dumps(poses), "utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "condition_training",
            "--tasks",
            str(task_path),
            "--cases",
            str(case_path),
            "--prior",
            str(prior),
            "--reference-corpus",
            str(corpus),
            "--out",
            str(out),
            "--updates",
            "1",
            "--rollout-start-steps",
            "0",
            "--reference-fraction",
            "0.5",
            "--reference-anchor-weight",
            "0.5",
        ],
    )
    main()
    result = json.loads((out / "result.json").read_text("utf-8"))
    assert result["updates"] == 1 and result["sac_updates"] == 0
    assert result["training_ids"] == ["train"] and result["validation_ids"] == ["heldout"]
    assert not result["live_inference_linked"] and not result["promoted"]
    assert (out / "practice-manifest.json").is_file()
    manifest = json.loads((out / "practice-manifest.json").read_text("utf-8"))
    assert manifest["baseline_actor_sha256"] == tasks["actor_sha256"]
    assert (out / "baseline.json").is_file()
    # Resuming must not silently accept a different model as the original baseline.
    manifest["baseline_actor_sha256"] = "0" * 64
    (out / "practice-manifest.json").write_text(json.dumps(manifest), "utf-8")
    import sys

    monkeypatch.setattr("sys.argv", [str(out) if v == str(prior) else v for v in sys.argv])
    with pytest.raises(ValueError, match="baseline actor changed"):
        main()
    env.close()


def test_explicit_practice_start_rejects_ambiguous_or_below_floor_pose():
    from types import SimpleNamespace

    from test_articulated_body import fixture

    from myumiq_vrchat.condition_validation import ConditionCase, starting_pose

    rig, states = fixture()
    pose = rig.forward(states[0]).model_dump(mode="json")
    raw = dict(
        id="observed",
        split="train",
        start_pose=pose,
        goal=dict(duration_s=12.0, conditions=[dict(part="head", position=[None, None, 1.4])]),
    )
    case = ConditionCase.model_validate_json(json.dumps(raw))
    settings = SimpleNamespace(goals={}, reference_floor=0.0)
    assert starting_pose(settings, case).is_full_body
    with pytest.raises(ValueError, match="one named start"):
        ConditionCase.model_validate_json(json.dumps(raw | {"start": "STAND"}))
    pose["left_foot"]["position"][2] = -0.1
    case = ConditionCase.model_validate_json(json.dumps(raw))
    with pytest.raises(ValueError, match="tracker plane"):
        starting_pose(settings, case)
