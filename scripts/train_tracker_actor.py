"""SAC candidate for coordinated pose transitions; no automatic VRChat promotion."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.motion import GltfMotion
from myumiq_vrchat.tracker_env import TrackerGoalEnv, distance
from myumiq_vrchat.tracker_policy import export_actor


class Progress(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.started = time.monotonic()

    def _on_step(self):
        if self.num_timesteps % 1000 == 0:
            self.output.write_text(
                json.dumps(
                    {
                        "steps": self.num_timesteps,
                        "updates": self.model._n_updates,
                        "elapsed_s": time.monotonic() - self.started,
                    }
                ),
                "utf-8",
            )
        return True


def evaluate(model, poses, *, record=False, decoder=None):
    env = TrackerGoalEnv(poses, horizon=100, recording=record, decoder=decoder)
    trials = []
    try:
        for seed in range(6):
            obs, _ = env.reset(seed=8000 + seed)
            goal = env.goal
            held = env.current
            initial = distance(held, goal)
            switched = None
            for step in range(100):
                if step == 50:
                    goal = poses[(seed + 3) % len(poses)]
                    prior = env.current
                    obs = env.set_goal(goal)
                    assert env.current is prior  # task changes do not reset the pose
                    switched = distance(prior, goal)
                action = (
                    model.predict(obs, deterministic=True)[0]
                    if model
                    else np.zeros(env.action_space.shape)
                )
                obs, _, _, _, info = env.step(action)
                if step == 49:
                    first = info["pose_error"]
            trials.append(
                {
                    "initial_error": initial,
                    "first_goal_error": first,
                    "switch_initial_error": switched,
                    "switched_goal_error": info["pose_error"],
                    "hold_final_error": distance(held, goal),
                }
            )
        values = [v for t in trials for v in (t["first_goal_error"], t["switched_goal_error"])]
        return {"mean_endpoint_error": float(np.mean(values)), "trials": trials}, env.replay
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gltf", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--motion-prior", action="store_true")
    parser.add_argument(
        "--components",
        type=int,
        default=0,
        help="learned whole-body rate basis size; 0 uses 66 rates",
    )
    args = parser.parse_args()
    if not 1000 <= args.steps <= 100000:
        raise ValueError("steps must be 1000..100000")
    if not 0 <= args.components <= 66 or (args.components and args.motion_prior):
        raise ValueError("component count must be 0..66; latent BC is not enabled")
    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=False)
    license_text = args.license.read_text("utf-8")
    if "CC0" not in license_text:
        raise ValueError("operator-supplied CC0 motion license required")
    torch.set_num_threads(1)
    motion = GltfMotion(args.gltf)
    train_clips = ("Idle_Loop", "Idle_Talking_Loop", "Crouch_Idle_Loop", "Sitting_Enter")
    heldout_clips = ("Sitting_Exit", "Crouch_Fwd_Loop")
    train = []
    heldout = []
    data = []
    for clip in train_clips + heldout_clips:
        frames = motion.retarget(clip, hz=10)
        selected = [f for i, f in enumerate(frames) if i % 4 == 0]
        target = train if clip in train_clips else heldout
        for t, pose in selected:
            target.append(pose)
            data.append(
                {
                    "clip": clip,
                    "time_s": t,
                    "split": "train" if clip in train_clips else "heldout",
                    "pose": pose.model_dump(mode="json"),
                }
            )
    (output / "poses.json").write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    decoder = None
    if args.components:
        from myumiq_vrchat.tracker_basis import RateBasis
        from myumiq_vrchat.tracker_training import motion_examples

        _, rates = motion_examples(motion, train_clips)
        decoder = RateBasis.fit(rates, args.components)
        (output / "rate-basis.json").write_text(decoder.model_dump_json(indent=2), "utf-8")
    env = TrackerGoalEnv(train, decoder=decoder)
    learner = SAC
    if args.motion_prior:
        from myumiq_vrchat.tracker_training import MotionPriorSAC

        learner = MotionPriorSAC
    model = learner(
        "MlpPolicy",
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
    hold, _ = evaluate(None, heldout)
    initial, _ = evaluate(model, heldout, decoder=decoder)
    start = time.monotonic()
    try:
        prior_report = None
        if args.motion_prior:
            from myumiq_vrchat.tracker_training import motion_examples

            inputs, actions = motion_examples(motion, train_clips)
            model.set_prior(inputs, actions)
            losses = [model.prior_update(256) for _ in range(3000)]
            pretrained, _ = evaluate(model, heldout)
            prior_report = {
                "examples": len(inputs),
                "pretraining_updates": 3000,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "evaluation": pretrained,
                "augmentation": "forward and reversed retargeted training clips; heldout clips excluded",
            }
            (output / "prior.json").write_text(json.dumps(prior_report, indent=2), "utf-8")
            model.save(output / "motion-prior-sac", exclude=["prior_observations", "prior_actions"])
            export_actor(model, output / "motion-prior-actor.pt")
        model.learn(total_timesteps=args.steps, callback=Progress(output / "progress.json"))
        model.save(output / "candidate-sac", exclude=["prior_observations", "prior_actions"])
        export_actor(model, output / "candidate-actor.pt", decoder=decoder)
        model.save_replay_buffer(output / "sac-replay.pkl")
        final, replay = evaluate(model, heldout, record=True, decoder=decoder)
        replay.save_state(output / "evaluation-experience.jsonl")
        result = {
            "steps": model.num_timesteps,
            "gradient_updates": model._n_updates,
            "elapsed_s": time.monotonic() - start,
            "scope": "ideal_tracker_geometry_not_avatar",
            "action_size": 66,
            "latent_action_size": args.components or None,
            "decoder_variance_fraction": decoder.variance_fraction if decoder else None,
            "train_poses": len(train),
            "heldout_poses": len(heldout),
            "train_clips": train_clips,
            "heldout_clips": heldout_clips,
            "hold_baseline": hold,
            "untrained_actor": initial,
            "candidate": final,
            "motion_prior": prior_report,
            "prior_updates": getattr(model, "prior_updates", 0),
            "improved_over_hold": final["mean_endpoint_error"] < hold["mean_endpoint_error"],
            "improved_over_untrained": final["mean_endpoint_error"]
            < initial["mean_endpoint_error"],
            "source_gltf_sha256": hashlib.sha256(args.gltf.read_bytes()).hexdigest(),
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
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
