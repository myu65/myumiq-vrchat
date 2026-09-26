"""Motion-derived whole-body prior around the existing SAC trainer."""

import numpy as np
import torch
from stable_baselines3 import SAC

from .tracker_policy import observation, pose_error


def motion_examples(motion, clips):
    inputs, actions = [], []
    dt = 0.05
    for clip in clips:
        poses = [pose for _, pose in motion.retarget(clip, hz=20)]
        for sequence in (poses, list(reversed(poses))):
            previous = np.zeros(66)
            for i in range(len(sequence) - 1):
                raw = (
                    pose_error(sequence[i], sequence[i + 1])
                    / np.array([0.6, 0.6, 0.6, 2.0, 2.0, 2.0])
                    / dt
                )
                scale = max(
                    1.0,
                    float(np.linalg.norm(raw[:, :3], axis=1).max()),
                    float(np.linalg.norm(raw[:, 3:], axis=1).max()),
                )
                action = (raw / scale).ravel().astype(np.float32)
                for ahead in (1, 4, 12, 24):
                    goal = sequence[min(i + ahead, len(sequence) - 1)]
                    inputs.append(observation(sequence[i], goal, previous, dt))
                    actions.append(action)
                # Holding a demonstrated pose is also part of the policy's task.
                inputs.append(observation(sequence[i], sequence[i], np.zeros(66), dt))
                actions.append(np.zeros(66, dtype=np.float32))
                previous = action
    return np.stack(inputs), np.stack(actions)


class MotionPriorSAC(SAC):
    """Alternates SAC with motion BC updates; no custom scheduler or SB3 fork."""

    algorithm_label = "SAC with alternating whole-body motion BC"
    prior_observations = prior_actions = None
    prior_updates = 0

    def set_prior(self, observations, actions):
        self.prior_observations = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        self.prior_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

    def prior_update(self, batch_size=128):
        if self.prior_observations is None:
            return None
        indices = torch.randint(len(self.prior_observations), (batch_size,), device=self.device)
        predicted = self.actor(self.prior_observations[indices], deterministic=True)
        loss = torch.nn.functional.mse_loss(predicted, self.prior_actions[indices])
        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        self.prior_updates += 1
        return float(loss.detach())

    def train(self, gradient_steps, batch_size=64):
        super().train(gradient_steps, batch_size)
        for _ in range(gradient_steps):
            self.prior_update(batch_size)
