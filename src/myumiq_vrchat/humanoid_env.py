"""Task-conditioned articulated physics, explicitly not a VRChat dynamics model."""

import gymnasium as gym
import numpy as np


class WholeBodyTaskEnv(gym.Wrapper):
    contract = "myumiq-humanoid-torque-v1"

    def __init__(self, **kwargs):
        super().__init__(gym.make("Humanoid-v5", **kwargs))
        self.task = np.array([0.0, 0.0, 1.4], dtype=np.float64)
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (self.env.observation_space.shape[0] + 3 + self.action_space.shape[0],),
            dtype=np.float64,
        )

    def set_task(self, velocity_x, velocity_y, torso_height):
        task = np.array([velocity_x, velocity_y, torso_height], dtype=np.float64)
        if not np.isfinite(task).all() or np.linalg.norm(task[:2]) > 2 or not 0.8 <= task[2] <= 1.8:
            raise ValueError("invalid body task")
        self.task = task

    def _observation(self, observation):
        return np.concatenate((observation, self.task, self.previous_action))

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.previous_action.fill(0.0)
        return self._observation(observation), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("invalid joint action")
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            raise ValueError("joint action outside actuator limits")
        observation, _, terminated, truncated, info = self.env.step(action)
        velocity = np.array([info["x_velocity"], info["y_velocity"]])
        height = float(self.unwrapped.data.qpos[2])
        components = {
            "velocity_tracking": float(np.exp(-2 * np.sum((velocity - self.task[:2]) ** 2))),
            "height_tracking": float(np.exp(-10 * (height - self.task[2]) ** 2)),
            "effort": -float(0.01 * np.sum(action**2)),
            "action_change": -float(0.05 * np.sum((action - self.previous_action) ** 2)),
            "fall": -5.0 if terminated else 0.0,
        }
        self.previous_action = action.copy()
        info.update(
            reward_components=components,
            motor_contract=self.contract,
            contact_count=int(self.unwrapped.data.ncon),
            locomotion="physical_positional",
            vrchat_transfer_verified=False,
        )
        return self._observation(observation), sum(components.values()), terminated, truncated, info
