"""Train an isolated PAMIQ candidate from confirmed replay; never hot-swap live weights."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pamiq_core.data import DataUsersDict
from pamiq_core.data.impls import SequentialBuffer
from pamiq_core.model import TrainingModelsDict
from pamiq_core.torch import TorchTrainingModel
from stable_baselines3 import SAC

from myumiq_vrchat.articulated_actor import ArticulatedActor
from myumiq_vrchat.articulated_body import ArticulatedRig, JointState, articulated_observation
from myumiq_vrchat.articulated_checkpoint import save_training_state
from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
from myumiq_vrchat.articulated_evaluation import evaluate
from myumiq_vrchat.articulated_experience import load_experience_cases, require_disjoint_sessions
from myumiq_vrchat.articulated_policy import export_articulated, policy_for_frame
from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.experience_refinement import (
    ExperienceRefinementTrainer,
    eligible_candidate,
    evaluate_observed_starts,
    reference_starts,
)
from myumiq_vrchat.whole_body import target_from_vector


def save_selection_evidence(output, attempts):
    """Keep detailed poses in bounded per-attempt artifacts, not duplicated in IPC."""
    summaries = []
    for index, attempt in enumerate(attempts):
        name = f"backtracking-{index:02}.json"
        (output / name).write_text(json.dumps(attempt, indent=2), "utf-8")
        summaries.append(
            {
                **attempt,
                "report": name,
                "reference_after": {
                    key: value
                    for key, value in attempt["reference_after"].items()
                    if key != "trials"
                },
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--train-replay", type=Path, required=True)
    parser.add_argument("--heldout-replay", type=Path, required=True)
    parser.add_argument("--reference-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--floor-weight", type=float, default=50.0)
    parser.add_argument("--reference-rehearsal", action="store_true")
    parser.add_argument("--backtracking-steps", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    args.prior = outside_repo(args.prior)
    args.reference_corpus = outside_repo(args.reference_corpus)
    torch.set_num_threads(1)
    torch.manual_seed(43)
    prior = ArticulatedActor(args.prior / "candidate-actor.pt")
    rig, decoder = prior.rig, prior.decoder
    if not np.array_equal(np.asarray(decoder.rows), np.eye(rig.action_size)):
        raise ValueError("this refinement requires the full-rank identity joint decoder")
    train, train_manifest = load_experience_cases(outside_repo(args.train_replay), prior)
    heldout, heldout_manifest = load_experience_cases(outside_repo(args.heldout_replay), prior)
    require_disjoint_sessions(train, heldout)
    report = json.loads((args.reference_corpus / "result.json").read_text("utf-8"))
    if report.get("license") != "CC0" or rig != ArticulatedRig.model_validate_json(
        (args.reference_corpus / "rig.json").read_text("utf-8")
    ):
        raise ValueError("compatible licensed reference regression corpus required")
    rows = json.loads((args.reference_corpus / "poses.json").read_text("utf-8"))
    references = [
        JointState(
            np.asarray(r["joint_state"]["root"]), np.asarray(r["joint_state"]["local_orientations"])
        )
        for r in rows
        if r["split"] == "heldout"
    ]
    env = ArticulatedGoalEnv(
        rig, references, decoder=decoder, reference_floor=prior.manifest["reference_floor"]
    )
    policy, options = policy_for_frame(prior.manifest["policy_frame"], rig)
    model = SAC(
        policy,
        env,
        device="cpu",
        buffer_size=10,
        policy_kwargs={"net_arch": prior.manifest["net_arch"], **options},
    )
    saved = torch.load(args.prior / "training-state.pt", map_location="cpu", weights_only=True)
    model.actor.load_state_dict(saved["actor"])
    for case in train[:8]:
        obs = articulated_observation(
            target_from_vector(case.current),
            target_from_vector(case.goal),
            case.previous,
            case.dt,
            case.joints,
        )
        np.testing.assert_allclose(
            prior.predict(obs)[0], model.predict(obs, deterministic=True)[0], atol=1e-6
        )
    trainer = ExperienceRefinementTrainer(
        rig,
        updates=args.updates,
        learning_rate=args.learning_rate,
        reference_floor=prior.manifest["reference_floor"],
        floor_weight=args.floor_weight,
        reference_rehearsal=args.reference_rehearsal,
    )
    wrapped = TorchTrainingModel(model.actor, has_inference_model=False)
    trainer.attach_training_models(TrainingModelsDict({"candidate": wrapped}))
    data = SequentialBuffer(len(train))
    for case in train:
        data.add(case)
    buffers = {"experience": data}
    rehearsals = []
    if args.reference_rehearsal:
        training_references = [
            JointState(
                np.asarray(r["joint_state"]["root"]),
                np.asarray(r["joint_state"]["local_orientations"]),
            )
            for r in rows
            if r["split"] == "train"
        ]
        rehearsals = reference_starts(rig, training_references, prior.manifest["reference_floor"])
        buffers["reference"] = SequentialBuffer(len(rehearsals))
        for case in rehearsals:
            buffers["reference"].add(case)
    trainer.attach_data_users(DataUsersDict.from_data_buffers(buffers))
    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        training=train_manifest,
        evaluation=heldout_manifest,
        initial_actor_sha256=prior.manifest["sha256"],
        rig_sha256=prior.manifest["rig_sha256"],
        training_evidence=[c.evidence for c in train],
        evaluation_evidence=[c.evidence for c in heldout],
        updates=args.updates,
        learning_rate=args.learning_rate,
        floor_weight=args.floor_weight,
        backtracking_steps=args.backtracking_steps,
        reference_rehearsal=dict(
            enabled=args.reference_rehearsal,
            cases=len(rehearsals),
            source_split="train",
            scope="synthetic_kinematic_reference_not_device_experience",
        ),
        live_inference_linked=False,
    )
    (output / "experience-manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    objective = trainer.objective
    before = evaluate_observed_starts(
        model.actor, rig, heldout, reference_floor=objective["reference_floor"]
    )
    reference_before, _ = evaluate(
        model, rig, decoder, references, settle_on_goal=True, **objective
    )
    continuous_before, _ = evaluate(model, rig, decoder, references, **objective)
    began = time.monotonic()
    print(json.dumps(dict(stage="training", cases=len(train), heldout=len(heldout))), flush=True)
    assert trainer.run()

    def assess():
        replay = evaluate_observed_starts(
            model.actor, rig, heldout, reference_floor=objective["reference_floor"]
        )
        reference, _ = evaluate(model, rig, decoder, references, settle_on_goal=True, **objective)
        return dict(after=replay, reference_after=reference)

    selected, attempts = trainer.select_step(
        assess,
        lambda metrics: eligible_candidate(
            before,
            metrics["after"],
            reference_before,
            metrics["reference_after"],
            objective["reference_floor"],
        ),
        args.backtracking_steps,
    )
    after, reference_after = selected["after"], selected["reference_after"]
    model.algorithm_label = "PAMIQ model-based refinement from confirmed replay starts"
    model.model_based_updates = prior.manifest["model_based_updates"] + trainer.total_updates
    model.prior_updates = prior.manifest["motion_bc_updates"]
    model.reference_floor = objective["reference_floor"]
    model.floor_weight, model.floor_power = objective["floor_weight"], objective["floor_power"]
    model.worst_tracker_weight = objective["worst_tracker_weight"]
    export_articulated(model, output / "candidate-actor.pt", rig, decoder)
    save_training_state(model, trainer.optimizers["actor"], output / "training-state.pt")
    trainer.save_state(output / "pamiq-trainer")
    continuous_after, _ = evaluate(model, rig, decoder, references, **objective)
    candidate = ArticulatedActor(output / "candidate-actor.pt")
    # Model evaluation is a prerequisite for a controlled trial, not live promotion.
    eligible = selected["eligible_for_controlled_trial"]
    result = dict(
        scope="observed_starts_model_rollout_not_avatar_or_world_learning",
        elapsed_s=time.monotonic() - began,
        updates=trainer.total_updates,
        sac_updates=0,
        training=train_manifest,
        heldout=heldout_manifest,
        before=before,
        after=after,
        reference_before=reference_before,
        reference_after=reference_after,
        continuous_actor_reference_before=continuous_before,
        continuous_actor_reference_after=continuous_after,
        candidate_sha256=candidate.manifest["sha256"],
        eligible_for_controlled_trial=eligible,
        backtracking=dict(
            max_halvings=args.backtracking_steps,
            exported_scale=selected["scale"],
            optimizer_reset=selected["scale"] < 1.0,
            attempts=save_selection_evidence(output, attempts),
        ),
        promoted=False,
        avatar_verified=False,
        automatic_live_sync=False,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2), "utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ("updates", "before", "after", "eligible_for_controlled_trial")}
        ),
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
