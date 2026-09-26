"""Motion-derived joint-rate demonstrations for the existing alternating SAC/BC trainer."""

import numpy as np

from .articulated_env import articulated_observation


def motion_examples(rig, motion, clips, decoder):
    inputs, actions = [], []
    dt = 0.05
    basis = np.asarray(decoder.rows)
    for clip in clips:
        states = [
            rig.sample(motion, clip, float(t))
            for t in np.arange(0, motion.duration(clip) + 1e-6, dt)
        ]
        for sequence in (states, list(reversed(states))):
            poses = [rig.forward(state) for state in sequence]
            previous = np.zeros(66)
            for i, (before, after) in enumerate(zip(sequence, sequence[1:])):
                raw = rig.rates_between(before, after, dt)
                raw /= max(1.0, float(np.linalg.norm(raw.reshape(-1, 3), axis=1).max()))
                latent = raw @ basis.T
                latent /= max(1.0, float(np.abs(latent).max()))
                for ahead in (1, 4, 12, 24):
                    goal = poses[min(i + ahead, len(poses) - 1)]
                    inputs.append(
                        articulated_observation(poses[i], goal, previous, dt, before.rotations)
                    )
                    actions.append(latent)
                inputs.append(
                    articulated_observation(poses[i], poses[i], np.zeros(66), dt, before.rotations)
                )
                actions.append(np.zeros(len(decoder.rows)))
                _, _, previous, _ = rig.advance(before, decoder.decode(latent), dt)
    return np.asarray(inputs, dtype=np.float32), np.asarray(actions, dtype=np.float32)
