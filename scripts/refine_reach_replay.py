"""Offline SAC candidate update from explicit PAMIQ REACH experience; no promotion."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.reach_policy import export_reach_actor
from myumiq_vrchat.reach_replay import reach_sample
from myumiq_vrchat.replay import Transition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--prior-replay", type=Path, help="locally generated SAC replay to retain")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.updates <= 10000:
        raise ValueError("updates must be 1..10000")
    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = SAC.load(outside_repo(args.model), device="cpu")
    model.set_logger(configure(str(output), format_strings=[]))
    model.replay_buffer.reset()
    if args.prior_replay is not None:
        model.load_replay_buffer(outside_repo(args.prior_replay))
    prior_count = model.replay_buffer.size()
    count, environments = 0, set()
    with outside_repo(args.experience).open(encoding="utf-8") as stream:
        for line in stream:
            record = Transition.model_validate_json(line)
            before, action, reward, after, done, info = reach_sample(record)
            model.replay_buffer.add(
                np.asarray([before], dtype=np.float32), np.asarray([after], dtype=np.float32),
                np.asarray([action], dtype=np.float32), np.asarray([reward], dtype=np.float32),
                np.asarray([done], dtype=np.float32), [info],
            )
            count += 1
            environments.add(record.environment)
    if count < 128:
        raise ValueError("at least 128 validated transitions required")
    previous = torch.cat([p.detach().flatten().clone() for p in model.actor.parameters()])
    model.train(gradient_steps=args.updates, batch_size=128)
    current = torch.cat([p.detach().flatten() for p in model.actor.parameters()])
    delta = float(torch.linalg.vector_norm(current - previous))
    if not np.isfinite(delta) or delta == 0:
        raise ValueError("actor update missing or non-finite")
    model.save(output / "candidate-sac")
    export_reach_actor(model, output / "candidate-actor.pt")
    result = {"loaded_transitions": count, "prior_transitions": prior_count,
              "retained_transitions": model.replay_buffer.size(),
              "environments": sorted(environments), "updates": args.updates,
              "actor_parameter_delta_l2": delta, "promoted": False,
              "performance_improvement_verified": False}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
