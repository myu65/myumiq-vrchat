"""Ideal articulated actuator training on the existing SAC/PAMIQ environment boundary."""

import hashlib

import gymnasium as gym
import numpy as np

from .articulated_body import OBSERVATION_CONTRACT as CONTRACT
from .articulated_body import articulated_observation as articulated_observation
from .joint_limits import CONTRACT as LIMIT_CONTRACT
from .tracker_env import TrackerGoalEnv
from .tracker_policy import worst_tracker_distance
from .whole_body import vector


def state_data(state):
    return {"root": state.root.tolist(), "local_orientations": state.rotations.tolist()}


class ArticulatedGoalEnv(TrackerGoalEnv):
    observation_contract = CONTRACT
    policy_id = "articulated-policy-candidate"

    def __init__(
        self,
        rig,
        states,
        *,
        decoder=None,
        reference_floor=None,
        floor_weight=20.0,
        floor_power=2,
        worst_tracker_weight=0.0,
        **kwargs,
    ):
        super().__init__([rig.forward(s) for s in states], **kwargs)
        if reference_floor is not None and (
            not np.isfinite(reference_floor) or abs(reference_floor) > 10
        ):
            raise ValueError("invalid source reference floor")
        self.reference_floor = reference_floor
        if (
            not np.isfinite(floor_weight)
            or not 0 < floor_weight <= 1000
            or floor_power not in (1, 2)
        ):
            raise ValueError("invalid reference floor objective")
        self.floor_weight, self.floor_power = floor_weight, floor_power
        if not np.isfinite(worst_tracker_weight) or not 0 <= worst_tracker_weight <= 10:
            raise ValueError("invalid worst tracker objective")
        self.worst_tracker_weight = worst_tracker_weight
        if decoder is not None and decoder.action_size != rig.action_size:
            raise ValueError("joint decoder does not match the rig")
        self.rig, self.states, self.decoder = rig, tuple(states), decoder
        self.rig_id = hashlib.sha256(rig.model_dump_json().encode()).hexdigest()
        self.state = self.goal_state = None
        self.observation_space = gym.spaces.Box(
            -10, 10, shape=(210 + len(rig.names) * 4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(len(decoder.rows) if decoder else rig.action_size,), dtype=np.float32
        )

    def observe(self):
        return articulated_observation(
            self.current, self.goal, self.previous, self.dt, self.state.rotations
        )

    def evaluate_reward(self, before, after, rates):
        result = super().evaluate_reward(before, after, rates)
        if self.worst_tracker_weight:
            old, new = (
                worst_tracker_distance(before, self.goal),
                worst_tracker_distance(after, self.goal),
            )
            result["worst_tracker_progress"] = 10 * self.worst_tracker_weight * (old - new)
            result["worst_tracker_error"] = -0.2 * self.worst_tracker_weight * new
        if self.reference_floor is not None:
            heights = vector(after)[9:, 2]
            result["reference_floor_intrusion"] = -self.floor_weight * float(
                np.mean(np.maximum(0, self.reference_floor + 0.02 - heights) ** self.floor_power)
            )
        return result

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        indices = self.np_random.choice(len(self.states), 2, replace=False)
        self.state = options.get("start", self.states[indices[0]])
        self.goal_state = options.get("goal", self.states[indices[1]])
        self.rig.validate_limits(self.state)
        self.current, self.goal = self.rig.forward(self.state), self.rig.forward(self.goal_state)
        self.previous = np.zeros(66, dtype=np.float32)
        self.steps = 0
        self.episode += 1
        return self.observe(), {"scope": "tracker_geometry", "rig_id": self.rig_id}

    def set_goal(self, state):
        self.goal_state = state
        self.goal = self.rig.forward(state)
        return self.observe() if self.current else None

    def advance(self, action):
        raw = np.asarray(action, dtype=float)
        joint_rates = self.decoder.decode(raw) if self.decoder else raw
        before = state_data(self.state)
        previous_state = self.state
        self.state, target, rates, scale = self.rig.advance(self.state, joint_rates, self.dt)
        return (
            target,
            rates,
            {
                "latent_action": raw.tolist() if self.decoder else None,
                "decoder_id": self.decoder.identity if self.decoder else None,
                "rig_id": self.rig_id,
                "joint_action": joint_rates.tolist(),
                "joint_action_scale": scale,
                "realized_joint_action": self.rig.rates_between(
                    previous_state, self.state, self.dt
                ).tolist(),
                "joint_constraint_contract": LIMIT_CONTRACT if self.rig.joint_limits else None,
                "joint_state": before,
                "next_joint_state": state_data(self.state),
                "joint_state_source": "simulated_imported_skeleton",
                "reference_floor": self.reference_floor,
                "reference_floor_weight": self.floor_weight,
                "reference_floor_power": self.floor_power,
                "worst_tracker_weight": self.worst_tracker_weight,
                "floor_source": "canonical_source_reference_plane"
                if self.reference_floor is not None
                else None,
            },
        )
