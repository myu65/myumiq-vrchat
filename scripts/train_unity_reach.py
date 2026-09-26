"""Train/evaluate SAC against an already running isolated Unity MotorLab player."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.reach_policy import export_reach_actor
from myumiq_vrchat.unity_gym import UnityReachEnv


def evaluate(env, model, seeds):
    rows = []
    env.recording = False
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            total = 0.0
            for step in range(150):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rows.append({"seed": seed, "return": total, "steps": step + 1, **info})
    finally:
        env.recording = True
    return {"success_rate": sum(row["is_success"] for row in rows)/len(rows),
            "mean_distance_m": float(np.mean([row["distance"] for row in rows])),
            "episodes": rows}


class Progress(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def _on_step(self):
        if self.num_timesteps % 1000 == 0:
            self.output.write_text(json.dumps({"steps": self.num_timesteps, "time": time.time()}))
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()
    if not 1000 <= args.steps <= 200000:
        raise ValueError("training steps must be 1000..200000")
    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = UnityReachEnv(outside_repo(args.ready), replay_size=args.steps + 1000)
    started = time.perf_counter()
    seeds = list(range(10000, 10020))
    result = {"environment": "unity", "algorithm": "SAC", "vrchat_transfer_verified": False}
    try:
        model = SAC(
            "MlpPolicy", env, device="cpu", seed=71, verbose=0,
            learning_rate=3e-4, buffer_size=args.steps + 1000, learning_starts=1000,
            batch_size=128, policy_kwargs={"net_arch": [64, 64]},
            train_freq=1, gradient_steps=1, ent_coef="auto_0.1",
        )
        result["before"] = evaluate(env, model, seeds)
        (output / "before.json").write_text(json.dumps(result["before"], indent=2))
        model.learn(total_timesteps=args.steps, callback=Progress(output / "progress.json"))
        model.save(output / "sac-reach")
        export_reach_actor(model, output / "reach-actor.pt")
        model.save_replay_buffer(output / "sac-replay.pkl")
        result["after"] = evaluate(env, model, seeds)
        result["optimizer_updates"] = model._n_updates
        result["policy_updated"] = model._n_updates > 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.replay.save_state(output / "experience.jsonl")
        env.close()
        result["transitions"] = env.transitions
        result["elapsed_s"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
