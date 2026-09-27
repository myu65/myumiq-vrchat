"""Bounded condition practice with isolated PAMIQ weights and visual admission pending."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .articulated_body import JointState, articulated_observation
from .articulated_fit import fit_body
from .body_conditions import complete_goal, resolve_conditions
from .condition_objective import ConditionObjective, ConditionStart, condition_scale
from .condition_validation import ConditionCase, starting_pose
from .whole_body import vector


def prepare_cases(actor, settings, cases):
    prepared = []
    for case in cases:
        start = starting_pose(settings, case)
        initial = JointState(
            np.asarray(start.pelvis.position), np.tile([1.0, 0, 0, 0], (len(actor.rig.names), 1))
        )
        state, _ = fit_body(actor.rig, start, initial)
        resolved = resolve_conditions(case.goal, start, case.world, case.observed_at)
        goal, completion = complete_goal(
            actor.rig, start, resolved, reference_floor=settings.reference_floor, initial=state
        )
        training = ConditionStart(
            state.root,
            state.rotations,
            vector(start),
            vector(goal),
            np.zeros(66),
            0.05,
            resolved.desired,
            condition_scale(resolved),
        )
        prepared.append((case, start, goal, resolved, training, completion))
    return prepared


def assess(actor, prepared, floor):
    from .posture_validation import transition

    rows = []
    for case, start, goal, resolved, _, _ in prepared:
        result = transition(
            actor,
            start,
            goal,
            duration_s=case.goal.duration_s,
            floor=floor,
            criterion=resolved.measure,
        )
        from .body import BodyTarget

        measured = resolved.measure(BodyTarget.model_validate_json(json.dumps(result["endpoint"])))
        ratios = [
            max(
                row["position_error_m"] / c.position_tolerance,
                row["rotation_error_rad"] / c.angular_tolerance,
            )
            for c, row in zip(case.goal.conditions, measured["conditions"])
        ]
        rows.append(
            dict(
                id=case.id,
                split=case.split,
                condition_ratio=max(ratios),
                conditions=measured,
                **result,
            )
        )
    return rows


def admissible(before, after, reference_before, reference_after, floor=0.0):
    """Validation selection, never automatic promotion or proof of human motion."""
    if not before or {r["id"] for r in before} != {r["id"] for r in after}:
        return False
    old = {r["id"]: r for r in before}
    if any(not r["accepted"] or r["floor_failure"] for r in after):
        return False
    if any(old[r["id"]]["accepted"] and not r["accepted"] for r in after):
        return False
    original_trials = {t["seed"]: t for t in reference_before.get("trials", [])}
    updated_trials = {t["seed"]: t for t in reference_after.get("trials", [])}
    if original_trials.keys() != updated_trials.keys():
        return False
    for seed, trial in original_trials.items():
        endpoints = trial["endpoints"]
        updated = updated_trials[seed]["endpoints"]
        if len(endpoints) != len(updated) or any(
            a["settled"] and not b["settled"] for a, b in zip(endpoints, updated)
        ):
            return False
    # Compare complete paths, including stopping. A mean endpoint gain cannot
    # excuse a new foot slide or a much harsher acceleration in another case.
    for row in after:
        a, b = row["motion_quality"], old[row["id"]]["motion_quality"]
        for key, slack in (
            ("maximum_foot_displacement_m", 0.003),
            ("maximum_acceleration_m_s2", 0.2),
            ("maximum_jerk_m_s3", 4.0),
        ):
            if a[key] > b[key] * 1.1 + slack:
                return False
    return bool(
        np.mean([r["condition_ratio"] for r in after])
        < np.mean([r["condition_ratio"] for r in before]) * 0.99
        and reference_after["static_goals_reached"] >= reference_before["static_goals_reached"]
        and reference_after["minimum_foot_height_m"] >= floor
        and reference_after["mean_endpoint_error"] <= reference_before["mean_endpoint_error"] * 1.05
        and reference_after["endpoints_worse_than_hold"]
        <= reference_before["endpoints_worse_than_hold"]
    )


def read_cases(path, settings):
    raw = json.loads(path.read_text("utf-8"))
    if not isinstance(raw, list) or not 2 <= len(raw) <= 64:
        raise ValueError("provide two to 64 bounded practice cases")
    cases = [ConditionCase.model_validate_json(json.dumps(c)) for c in raw]
    signatures = [
        (
            starting_pose(settings, c).model_dump_json(),
            c.goal.model_dump_json(),
            c.world.model_dump_json(),
        )
        for c in cases
    ]
    if (
        len(set(c.id for c in cases)) != len(cases)
        or len(set(signatures)) != len(cases)
        or set(c.split for c in cases) != {"train", "heldout"}
    ):
        raise ValueError("distinct cases and both independent condition splits required")
    return cases


def main():
    from pamiq_core.data import DataUsersDict
    from pamiq_core.data.impls import SequentialBuffer
    from pamiq_core.model import TrainingModelsDict
    from pamiq_core.torch import TorchTrainingModel
    from stable_baselines3 import SAC

    from .articulated_actor import ArticulatedActor
    from .articulated_body import ArticulatedRig
    from .articulated_checkpoint import save_training_state
    from .articulated_env import ArticulatedGoalEnv
    from .articulated_evaluation import evaluate
    from .articulated_policy import export_articulated, policy_for_frame
    from .articulated_tasks import ArticulatedTasks
    from .cli import outside_repo
    from .experience_refinement import ExperienceRefinementTrainer, reference_starts

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tasks", "cases", "prior", "reference-corpus", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--baseline", type=Path, help="original actor .pt; retained across practice rounds"
    )
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-start-steps", type=int, default=40)
    parser.add_argument("--reference-fraction", type=float, default=0.25)
    parser.add_argument("--reference-anchor-weight", type=float, default=0.05)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(43)
    settings = ArticulatedTasks.model_validate_json(outside_repo(args.tasks).read_text("utf-8"))
    cases = read_cases(outside_repo(args.cases), settings)
    prior_dir = outside_repo(args.prior)
    actor = ArticulatedActor(prior_dir / "candidate-actor.pt")
    baseline_path = args.baseline
    baseline_hash = None
    previous_manifest = prior_dir / "practice-manifest.json"
    if baseline_path is None and previous_manifest.is_file():
        previous = json.loads(previous_manifest.read_text("utf-8"))
        if previous.get("baseline_actor"):
            baseline_path = Path(previous["baseline_actor"])
            baseline_hash = previous["baseline_actor_sha256"]
    previous_result = prior_dir / "result.json"
    if baseline_path is None and previous_result.is_file():
        previous = json.loads(previous_result.read_text("utf-8"))
        if previous.get("scope") == "synthetic_conditions_model_rollout_not_avatar_training":
            raise ValueError("resuming older practice requires --baseline for the original actor")
    baseline_path = outside_repo(baseline_path or (prior_dir / "candidate-actor.pt"))
    baseline = ArticulatedActor(baseline_path)
    if baseline_hash is not None and baseline.manifest["sha256"] != baseline_hash:
        raise ValueError("original baseline actor changed between practice rounds")
    rig, decoder = actor.rig, actor.decoder
    if (
        baseline.rig != rig
        or baseline.decoder != decoder
        or baseline.manifest["reference_floor"] != settings.reference_floor
    ):
        raise ValueError("original baseline differs from practice rig, decoder or floor")
    if (
        actor.manifest["sha256"] != settings.actor_sha256
        or actor.manifest["rig_sha256"] != settings.rig_sha256
        or actor.manifest["reference_floor"] != settings.reference_floor
        or not np.array_equal(np.asarray(decoder.rows), np.eye(rig.action_size))
    ):
        raise ValueError("practice actor, rig, floor or decoder differs from configuration")
    corpus = outside_repo(args.reference_corpus)
    report = json.loads((corpus / "result.json").read_text("utf-8"))
    if report.get("license") != "CC0" or rig != ArticulatedRig.model_validate_json(
        (corpus / "rig.json").read_text("utf-8")
    ):
        raise ValueError("compatible licensed reference corpus required")
    rows = json.loads((corpus / "poses.json").read_text("utf-8"))
    states = {
        split: [
            JointState(
                np.asarray(r["joint_state"]["root"]),
                np.asarray(r["joint_state"]["local_orientations"]),
            )
            for r in rows
            if r["split"] == split
        ]
        for split in ("train", "heldout")
    }
    prepared = prepare_cases(actor, settings, cases)
    training = [p[4] for p in prepared if p[0].split == "train"]
    validation = [p for p in prepared if p[0].split == "heldout"]
    env = ArticulatedGoalEnv(
        rig, states["heldout"], decoder=decoder, reference_floor=settings.reference_floor
    )
    policy, options = policy_for_frame(actor.manifest["policy_frame"], rig)
    model = SAC(
        policy,
        env,
        device="cpu",
        buffer_size=10,
        policy_kwargs={"net_arch": actor.manifest["net_arch"], **options},
    )
    saved = torch.load(prior_dir / "training-state.pt", map_location="cpu", weights_only=True)
    model.actor.load_state_dict(saved["actor"])
    for p in prepared:
        obs = articulated_observation(p[1], p[2], p[4].previous, 0.05, p[4].joints)
        np.testing.assert_allclose(
            actor.predict(obs)[0], model.predict(obs, deterministic=True)[0], atol=1e-6
        )
    trainer = ExperienceRefinementTrainer(
        rig,
        updates=args.updates,
        learning_rate=args.learning_rate,
        horizon=6,
        floor_weight=100,
        reference_floor=settings.reference_floor,
        whole_body_floor=True,
        reference_rehearsal=True,
        reference_fraction=args.reference_fraction,
        reference_anchor_weight=args.reference_anchor_weight,
        extra_objective_factory=ConditionObjective,
        rollout_start_steps=args.rollout_start_steps,
        progress=lambda n, loss: print(
            json.dumps(dict(stage="practice", updates=n, loss=loss)), flush=True
        ),
    )
    trainer.attach_training_models(
        TrainingModelsDict(
            {"candidate": TorchTrainingModel(model.actor, has_inference_model=False)}
        )
    )
    buffers = {}
    for key, values in (
        ("experience", training),
        ("reference", reference_starts(rig, states["train"], settings.reference_floor)),
    ):
        buffers[key] = SequentialBuffer(len(values))
        for value in values:
            buffers[key].add(value)
    trainer.attach_data_users(DataUsersDict.from_data_buffers(buffers))
    out = outside_repo(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "practice-manifest.json").write_text(
        json.dumps(
            dict(
                tasks_sha256=hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
                cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                initial_actor_sha256=actor.manifest["sha256"],
                baseline_actor=str(baseline_path),
                baseline_actor_sha256=baseline.manifest["sha256"],
                rig_sha256=settings.rig_sha256,
                reference_floor=settings.reference_floor,
                learning_rate=args.learning_rate,
                updates=args.updates,
                rollout_start_steps=args.rollout_start_steps,
                horizon=trainer.horizon,
                batch_size=trainer.batch_size,
                objective=trainer.objective,
                condition_weight=0.1,
                smoothness_weight=0.04,
                reference_fraction=args.reference_fraction,
                reference_anchor_weight=args.reference_anchor_weight,
                corpus_report_sha256=hashlib.sha256(
                    (corpus / "result.json").read_bytes()
                ).hexdigest(),
                completion=[
                    dict(
                        id=p[0].id, split=p[0].split, report=p[5], goal=p[2].model_dump(mode="json")
                    )
                    for p in prepared
                ],
                live_inference_linked=False,
            ),
            indent=2,
        ),
        "utf-8",
    )
    wrapped = SimpleNamespace(
        rig=rig,
        decoder=decoder,
        predict=lambda obs, deterministic=True: model.predict(obs, deterministic=deterministic),
    )
    before = assess(wrapped, validation, settings.reference_floor)
    reference_before, _ = evaluate(
        model, rig, decoder, states["heldout"], settle_on_goal=True, **trainer.objective
    )
    baseline_wrapped = SimpleNamespace(
        predict=lambda obs, deterministic=True: baseline.predict(obs)
    )
    baseline_conditions = assess(baseline, validation, settings.reference_floor)
    baseline_reference, _ = evaluate(
        baseline_wrapped, rig, decoder, states["heldout"], settle_on_goal=True, **trainer.objective
    )
    (out / "baseline.json").write_text(
        json.dumps(dict(conditions=baseline_conditions, reference=baseline_reference), indent=2),
        "utf-8",
    )
    (out / "before.json").write_text(
        json.dumps(dict(conditions=before, reference=reference_before), indent=2), "utf-8"
    )
    print(
        json.dumps(
            dict(
                stage="training",
                train=len(training),
                validation=len(validation),
                before_reached=sum(r["accepted"] for r in before),
            )
        ),
        flush=True,
    )
    assert trainer.run()

    def evaluate_candidate():
        reference, _ = evaluate(
            model, rig, decoder, states["heldout"], settle_on_goal=True, **trainer.objective
        )
        return dict(
            conditions=assess(wrapped, validation, settings.reference_floor), reference=reference
        )

    selected, attempts = trainer.select_step(
        evaluate_candidate,
        lambda m: (
            admissible(
                before, m["conditions"], reference_before, m["reference"], settings.reference_floor
            )
            and admissible(
                baseline_conditions,
                m["conditions"],
                baseline_reference,
                m["reference"],
                settings.reference_floor,
            )
        ),
        2,
    )
    model.algorithm_label = "PAMIQ condition practice with kinematic trajectory proxies"
    model.model_based_updates = actor.manifest["model_based_updates"] + trainer.total_updates
    model.prior_updates = actor.manifest["motion_bc_updates"]
    model.reference_floor = settings.reference_floor
    model.floor_weight, model.floor_power, model.worst_tracker_weight = 100, 1, 1.0
    export_articulated(model, out / "candidate-actor.pt", rig, decoder)
    save_training_state(model, trainer.optimizers["actor"], out / "training-state.pt")
    trainer.save_state(out / "pamiq-trainer")
    (out / "result.json").write_text(
        json.dumps(
            dict(
                scope="synthetic_conditions_model_rollout_not_avatar_training",
                initial_actor_sha256=actor.manifest["sha256"],
                baseline_actor_sha256=baseline.manifest["sha256"],
                cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                training_ids=[p[0].id for p in prepared if p[0].split == "train"],
                validation_ids=[p[0].id for p in validation],
                updates=trainer.total_updates,
                sac_updates=0,
                live_inference_linked=False,
                promoted=False,
                avatar_verified=False,
                before=before,
                reference_before=reference_before,
                after=selected,
                attempts=attempts,
                eligible_for_controlled_trial=selected["eligible_for_controlled_trial"],
            ),
            indent=2,
        ),
        "utf-8",
    )
    print(
        json.dumps(
            dict(
                stage="finished",
                eligible=selected["eligible_for_controlled_trial"],
                accepted=sum(r["accepted"] for r in selected["conditions"]),
                total=len(validation),
            )
        ),
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
