"""CPU SAC candidate training; never activates a policy in VRChat."""

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from myumiq_vrchat.humanoid_env import WholeBodyTaskEnv


class Progress(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def _on_step(self):
        if self.num_timesteps % 1000 == 0:
            self.output.write_text(json.dumps({'steps': self.num_timesteps}), encoding='utf-8')
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=30000)
    args = parser.parse_args()
    if not 1000 <= args.steps <= 500000:
        raise ValueError('steps outside experiment bounds')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = WholeBodyTaskEnv()
    # First learn physical support. Goal-sequence generalization requires separate evaluation.
    env.set_task(0., 0., 1.4)
    model = SAC('MlpPolicy', env, device='cpu', seed=23, learning_starts=1000,
                buffer_size=min(args.steps+1, 100000), batch_size=128,
                policy_kwargs={'net_arch': [128, 128]}, verbose=0)
    try:
        model.learn(total_timesteps=args.steps, callback=Progress(args.output/'progress.json'))
        model.save(args.output/'candidate')
        model.save_replay_buffer(args.output/'replay.pkl')
        (args.output/'result.json').write_text(json.dumps({
            'steps': model.num_timesteps, 'gradient_updates': model._n_updates,
            'contract': env.contract, 'task': env.task.tolist(), 'seed': 23,
            'promoted': False, 'vrchat_transfer_verified': False,
        }, indent=2), encoding='utf-8')
    finally:
        env.close()


if __name__ == '__main__':
    main()
