"""Compare a physics candidate against zero torque on held-out task sequences."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from myumiq_vrchat.humanoid_env import WholeBodyTaskEnv


def evaluate(model, seeds, horizon=300):
    env = WholeBodyTaskEnv()
    rows = []
    try:
        for seed in seeds:
            env.set_task(0., 0., 1.4)
            observation, _ = env.reset(seed=seed)
            total = 0.
            errors = []
            switches = 0
            phases = []
            for step in range(horizon):
                if step in (100, 200):
                    # Change goal without resetting pose/velocity or action history.
                    env.set_task(.2 if step == 100 else -.1, .1, 1.35)
                    action_size = env.action_space.shape[0]
                    observation[-action_size-3:-action_size] = env.task
                    switches += 1
                action = np.zeros(17) if model is None else model.predict(observation, deterministic=True)[0]
                observation, reward, terminated, truncated, info = env.step(action)
                total += reward
                errors.append(float(np.linalg.norm(np.array([info['x_velocity'], info['y_velocity']])-env.task[:2])))
                phase_index = step // 100
                if len(phases) <= phase_index:
                    phases.append({'task': env.task.tolist(), 'steps': 0,
                                   'velocity_errors': [], 'height_errors': []})
                phase = phases[phase_index]
                phase['steps'] += 1
                phase['velocity_errors'].append(errors[-1])
                phase['height_errors'].append(abs(float(env.unwrapped.data.qpos[2])-env.task[2]))
                if terminated or truncated:
                    break
            for phase in phases:
                phase['mean_velocity_error'] = float(np.mean(phase.pop('velocity_errors')))
                phase['mean_height_error'] = float(np.mean(phase.pop('height_errors')))
            rows.append(dict(seed=seed, steps=step+1, survived_horizon=not terminated and step+1==horizon,
                             task_switches= switches, total_reward=total,
                             mean_velocity_error=float(np.mean(errors)), phases=phases,
                             termination='fallen' if terminated else 'time_limit' if truncated else 'evaluation_horizon'))
    finally:
        env.close()
    return dict(episodes=rows, mean_steps=float(np.mean([r['steps'] for r in rows])),
                mean_reward=float(np.mean([r['total_reward'] for r in rows])),
                completed_both_task_changes=sum(r['task_switches']==2 and r['survived_horizon'] for r in rows)/len(rows),
                survived_horizon=sum(r['survived_horizon'] for r in rows)/len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = SAC.load(args.candidate, device='cpu')
    seeds = list(range(10000, 10010))
    report = {'contract': WholeBodyTaskEnv.contract, 'seeds': seeds,
              'baseline': evaluate(None, seeds), 'candidate': evaluate(model, seeds),
              'scope': 'articulated physics only; no VRChat transfer', 'promoted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: {a: b for a, b in report[k].items() if a != 'episodes'}
                      for k in ('baseline', 'candidate')}))


if __name__ == '__main__':
    main()
