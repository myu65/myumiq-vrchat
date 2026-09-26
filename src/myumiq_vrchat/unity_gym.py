"""Gymnasium adapter with common PAMIQ experience records for Unity motor training."""

import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from .body import HandTarget, Pose, WorldObject, WorldState, rest_target, simulated_body
from .cognition import Decision, Goal
from .motor import MotionCommand
from .replay import Observation, ReplayBuffer, Transition
from .unity_reach import ReachStep, UnityReachClient, integrate_reach


class UnityReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ready_file: Path, replay_size: int = 100000):
        self.client = UnityReachClient(ready_file)
        self.observation_space = gym.spaces.Box(-2, 2, shape=(9,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.replay = ReplayBuffer(replay_size)
        self.current: ReachStep | None = None
        self.recording = True
        self.transitions = 0
        self._seed = 0
        self.goal = Goal(skill="REACH", target="reach-target", hand="right", duration_s=5)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
        else:
            self._seed = int(self.np_random.integers(0, 2**30))
        self.current = self.client.reset(self._seed)
        return np.asarray(self.current.observation, dtype=np.float32), {"seed": self._seed}

    @staticmethod
    def observation(state: ReachStep) -> Observation:
        target = rest_target().model_copy(
            update={
                "right": HandTarget(pose=Pose(position=state.hand)),
            }
        )
        now = time.perf_counter()
        return Observation(
            timestamp=now,
            body=simulated_body(target, now),
            world=WorldState(
                objects=(
                    WorldObject(
                        name="reach-target",
                        position=state.goal,
                        source="fixture",
                        kind="object",
                    ),
                )
            ),
        )

    def step(self, action):
        if self.current is None:
            raise RuntimeError("reset required")
        before = self.current
        values = tuple(float(v) for v in action)
        position, _ = integrate_reach(before.hand, before.observation[6:], values)
        requested = rest_target().model_copy(
            update={
                "right": HandTarget(pose=Pose(position=position)),
            }
        )
        old_observation = self.observation(before)
        after = self.client.step(values)
        if self.recording:
            record = Transition(
                observation=old_observation,
                next_observation=self.observation(after),
                decision=Decision(goal=self.goal, source="fixed"),
                command=MotionCommand(goal=self.goal, elapsed_s=before.step / 30),
                action=requested,
                reward=after.reward,
                outcome="simulated",
                environment="unity",
                motor_observation=before.observation,
                motor_action=values,
                next_motor_observation=after.observation,
                motor_contract="myumiq-reach-v1",
                reward_kind="reach_task_v1",
                terminated=after.terminated,
                truncated=after.truncated,
            )
            self.replay.add(record.model_dump_json())
            self.transitions += 1
        self.current = after
        return (
            np.asarray(after.observation, dtype=np.float32),
            after.reward,
            after.terminated,
            after.truncated,
            {"is_success": after.success, "distance": float(np.linalg.norm(after.observation[:3]))},
        )

    def close(self):
        self.client.close()
