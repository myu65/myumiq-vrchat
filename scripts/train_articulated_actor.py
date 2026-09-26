"""Train and audit a coordinated joint-rate SAC candidate, without live promotion."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from train_tracker_actor import Progress

from myumiq_vrchat.articulated_body import ArticulatedRig
from myumiq_vrchat.articulated_env import ArticulatedGoalEnv, state_data
from myumiq_vrchat.articulated_evaluation import evaluate as evaluate
from myumiq_vrchat.articulated_policy import (
    AnchoredSACPolicy,
    ArticulatedActor,
    export_articulated,
)
from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.motion import GltfMotion
from myumiq_vrchat.tracker_basis import RateBasis
from myumiq_vrchat.tracker_policy import pose_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gltf", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--motion-prior", action="store_true")
    parser.add_argument("--diverse-motion", action="store_true")
    args = parser.parse_args()
    if not 1000 <= args.steps <= 100000 or not 1 <= args.components <= 63:
        raise ValueError("invalid bounded training budget or basis size")
    if "CC0" not in args.license.read_text("utf-8"):
        raise ValueError("operator-supplied CC0 source license required")
    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    motion = GltfMotion(args.gltf)
    rig = ArticulatedRig.from_motion(motion)
    train_clips = ("Idle_Loop", "Idle_Talking_Loop", "Crouch_Idle_Loop", "Sitting_Enter")
    if args.diverse_motion:
        train_clips += (
            "Interact",
            "PickUp_Table",
            "Walk_Loop",
            "Sitting_Talking_Loop",
            "Fixing_Kneeling",
            "Idle_Torch_Loop",
            "Dance_Loop",
        )
    heldout_clips = ("Sitting_Exit", "Crouch_Fwd_Loop")
    train, withheld, examples, data = [], [], [], []
    import_errors = []
    for clip in train_clips + heldout_clips:
        for i, (t, pose) in enumerate(motion.retarget(clip, hz=10)):
            if i % 4:
                continue
            state = rig.sample(motion, clip, t)
            error = float(np.linalg.norm(pose_error(rig.forward(state), pose)[:, :3], axis=1).max())
            import_errors.append(error)
            if error > 0.001:
                raise ValueError("source animation changes reference bone offsets")
            (train if clip in train_clips else withheld).append(state)
            data.append(
                {
                    "clip": clip,
                    "time_s": t,
                    "split": "train" if clip in train_clips else "heldout",
                    "joint_state": state_data(state),
                    "pose": rig.forward(state).model_dump(mode="json"),
                }
            )
        if clip in train_clips:
            sequence = [
                rig.sample(motion, clip, float(t))
                for t in np.arange(0, motion.duration(clip) + 1e-6, 0.05)
            ]
            for before, after in zip(sequence, sequence[1:]):
                rates = rig.rates_between(before, after, 0.05)
                rates /= max(1.0, float(np.linalg.norm(rates.reshape(-1, 3), axis=1).max()))
                examples.extend((rates, -rates))
    decoder = RateBasis.fit(np.stack(examples), args.components)
    (output / "rig.json").write_text(rig.model_dump_json(indent=2), "utf-8")
    (output / "joint-rate-basis.json").write_text(decoder.model_dump_json(indent=2), "utf-8")
    (output / "poses.json").write_text(json.dumps(data), "utf-8")
    env = ArticulatedGoalEnv(rig, train, decoder=decoder)
    learner = SAC
    if args.motion_prior:
        from myumiq_vrchat.tracker_training import MotionPriorSAC

        learner = MotionPriorSAC
    model = learner(
        AnchoredSACPolicy,
        env,
        device="cpu",
        seed=43,
        learning_starts=256,
        buffer_size=min(args.steps + 1, 50000),
        batch_size=128,
        ent_coef=0.003,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
    )
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(-3.0)
    hold, _ = evaluate(None, rig, decoder, withheld)
    initial, _ = evaluate(model, rig, decoder, withheld)
    oracle, _ = evaluate(None, rig, decoder, withheld, oracle=True)
    began = time.monotonic()
    try:
        prior = None
        if args.motion_prior:
            from myumiq_vrchat.articulated_training import motion_examples

            inputs, actions = motion_examples(rig, motion, train_clips, decoder)
            model.set_prior(inputs, actions)
            losses = [model.prior_update(256) for _ in range(3000)]
            prior_eval, _ = evaluate(model, rig, decoder, withheld)
            prior = {
                "examples": len(inputs),
                "initial_updates": 3000,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "evaluation": prior_eval,
                "augmentation": "forward/reversed motion directions; common velocity limits; training clips only",
            }
            (output / "prior.json").write_text(json.dumps(prior, indent=2), "utf-8")
            export_articulated(model, output / "prior-actor.pt", rig, decoder)
        model.learn(total_timesteps=args.steps, callback=Progress(output / "progress.json"))
        model.save(output / "candidate-sac", exclude=["prior_observations", "prior_actions"])
        export_articulated(model, output / "candidate-actor.pt", rig, decoder)
        model.save_replay_buffer(output / "sac-replay.pkl")
        actor = ArticulatedActor(output / "candidate-actor.pt")
        final, replay = evaluate(actor, rig, decoder, withheld, record=True)
        replay.save_state(output / "evaluation-experience.jsonl")
        faster, _ = evaluate(actor, rig, decoder, withheld, dt=0.02)
        stationary, _ = evaluate(actor, rig, decoder, withheld, dt=0.02, hold=True)
        result = {
            "scope": "ideal_articulated_tracker_actuator_not_avatar",
            "steps": model.num_timesteps,
            "gradient_updates": model._n_updates,
            "elapsed_s": time.monotonic() - began,
            "joint_count": len(rig.names),
            "joint_action_size": rig.action_size,
            "latent_action_size": args.components,
            "decoder_variance_fraction": decoder.variance_fraction,
            "motion_prior": prior,
            "prior_updates": getattr(model, "prior_updates", 0),
            "maximum_source_reconstruction_error_m": max(import_errors),
            "hold_baseline": hold,
            "untrained": initial,
            "oracle_diagnostic_not_learned": oracle,
            "candidate": final,
            "faster_timestep": faster,
            "already_at_goal": stationary,
            "improved_over_hold": final["mean_endpoint_error"] < hold["mean_endpoint_error"],
            "train_clips": train_clips,
            "heldout_clips": heldout_clips,
            "source_files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [args.gltf, args.license]
                + [args.gltf.parent / b["uri"] for b in motion.data["buffers"]]
            },
            "license": "CC0",
            "promoted": False,
            "avatar_verified": False,
            "world_motion_verified": False,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), "utf-8")
        print(
            json.dumps(
                {
                    k: v["mean_endpoint_error"]
                    for k, v in result.items()
                    if isinstance(v, dict) and "mean_endpoint_error" in v
                }
            ),
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
