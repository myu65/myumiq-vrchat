"""Ideal tracker actuator pretraining; does not simulate VRChat IK or world motion."""

import gymnasium as gym
import numpy as np

from .body import BodyGoal, BodyTask, WorldState, simulated_body
from .replay import Observation, ReplayBuffer, TrackerLearningStep, WholeBodyTransition
from .tracker_action import ACTION_SHAPE, integrate_tracker_action
from .tracker_policy import (
    OBSERVATION_CONTRACT,
    OBSERVATION_SIZE,
    distance,
    observation,
    reward_components,
)
from .whole_body import PARTS, vector


class TrackerGoalEnv(gym.Env):
    metadata = {"render_modes": []}
    observation_contract = OBSERVATION_CONTRACT
    policy_id = "tracker-sac-candidate"

    def __init__(self, poses, *, dt=0.05, horizon=160, recording=False, decoder=None):
        if len(poses) < 2 or not 0 < dt <= 0.1 or not 2 <= horizon <= 1000:
            raise ValueError("invalid tracker pretraining configuration")
        if decoder is not None and decoder.action_size != 66:
            raise ValueError("tracker environment requires a 66-rate decoder")
        for pose in poses:
            vector(pose)
        self.poses, self.dt, self.horizon = tuple(poses), dt, horizon
        self.recording = recording
        self.decoder = decoder
        self.replay = ReplayBuffer(10000)
        self.observation_space = gym.spaces.Box(
            -10, 10, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(len(decoder.rows) if decoder else 66,), dtype=np.float32
        )
        self.current = self.goal = None
        self.previous = np.zeros(66, dtype=np.float32)
        self.steps = self.episode = 0

    def set_goal(self, goal):
        vector(goal)
        self.goal = goal
        return self.observe() if self.current else None

    def observe(self):
        return observation(self.current, self.goal, self.previous, self.dt)

    def evaluate_reward(self, before, after, rates):
        return reward_components(before, after, self.goal, rates, self.previous)

    def advance(self, action):
        latent = np.asarray(action, dtype=np.float64)
        rates = self.decoder.decode(latent) if self.decoder else latent
        if rates.shape != (66,):
            raise ValueError("whole-body action must have 66 rates")
        return (
            integrate_tracker_action(self.current, rates.reshape(ACTION_SHAPE), self.dt),
            rates,
            {
                "latent_action": latent.tolist() if self.decoder else None,
                "decoder_id": self.decoder.identity if self.decoder else None,
            },
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        indices = self.np_random.choice(len(self.poses), size=2, replace=False)
        self.current = options.get("start", self.poses[indices[0]])
        self.goal = options.get("goal", self.poses[indices[1]])
        self.previous = np.zeros(66, dtype=np.float32)
        self.steps = 0
        self.episode += 1
        return self.observe(), {"scope": "tracker_geometry"}

    def step(self, action):
        if self.current is None:
            raise RuntimeError("reset required")
        before = self.current
        obs = self.observe()
        after, rates, metadata = self.advance(action)
        components = self.evaluate_reward(before, after, rates)
        self.current, self.previous = after, rates.copy()
        self.steps += 1
        following = self.observe()
        error = distance(after, self.goal)
        # A time limit is not a terminal task state: success should also learn to hold.
        terminated, truncated = False, self.steps >= self.horizon
        reward = sum(components.values())
        if self.recording:
            t = self.steps * self.dt
            evidence = f"ideal-tracker:episode-{self.episode}:step-{self.steps}"
            self.replay.add(
                WholeBodyTransition(
                    observation=Observation(
                        timestamp=t - self.dt,
                        body=simulated_body(before, t - self.dt),
                        world=WorldState(),
                    ),
                    next_observation=Observation(
                        timestamp=t, body=simulated_body(after, t), world=WorldState()
                    ),
                    body_goal=BodyGoal(
                        tasks=(BodyTask(id="pose-goal", kind="posture", effectors=PARTS),),
                        duration_s=min(20.0, self.horizon * self.dt),
                    ),
                    action=after,
                    policy_id=self.policy_id,
                    reward=reward,
                    outcome="simulated",
                    environment="mock",
                    intent_metadata={
                        "target_pose": self.goal.model_dump(mode="json"),
                        "scope": "ideal_tracker_actuator_not_avatar",
                        **metadata,
                    },
                    learning=TrackerLearningStep(
                        rates=tuple(rates),
                        dt=self.dt,
                        policy_observation=tuple(obs),
                        next_policy_observation=tuple(following),
                        observation_contract=self.observation_contract,
                        reward_components=components,
                        reward_scope="tracker_geometry",
                        evidence_ids=(evidence,),
                        terminated=terminated,
                        truncated=truncated,
                    ),
                ).model_dump_json()
            )
        return (
            following,
            reward,
            terminated,
            truncated,
            {
                "pose_error": error,
                "is_success": error < 0.03,
                "scope": "tracker_geometry",
                "reward_components": components,
                "avatar_verified": False,
            },
        )
