"""Versioned JSON transitions in PAMIQ's bounded replay buffer/persistence hooks."""

import math
from pathlib import Path
from typing import Literal

from pamiq_core.data import DataBuffer
from pamiq_core.data.impls import SequentialBuffer
from pydantic import Field, TypeAdapter, model_validator

from .body import ActuationTarget, BodyGoal, BodyState, Frozen, Number, SignedUnit, WorldState
from .cognition import Decision
from .motor import MotionCommand


class Observation(Frozen):
    timestamp: Number
    world: WorldState
    body: BodyState


class Transition(Frozen):
    schema_version: Literal[1] = 1
    observation: Observation
    decision: Decision
    command: MotionCommand
    action: ActuationTarget
    next_observation: Observation
    reward: Number | None = None
    outcome: Literal["simulated", "device_feedback", "visual_feedback", "unobserved"]
    environment: Literal["unspecified", "mock", "unity", "vrchat"] = "unspecified"
    motor_observation: tuple[Number, ...] | None = Field(default=None, max_length=32)
    motor_action: tuple[Number, ...] | None = Field(default=None, max_length=32)
    next_motor_observation: tuple[Number, ...] | None = Field(default=None, max_length=32)
    motor_contract: Literal["myumiq-reach-v1"] | None = None
    reward_kind: Literal["device_progress", "reach_task_v1"] | None = None
    terminated: bool | None = None
    truncated: bool | None = None


def device_progress_reward(
    before: Observation, action: ActuationTarget, after: Observation
) -> float | None:
    """Proprioceptive target progress only; this is not avatar or social success."""
    distances = []
    for name in ("head", "left", "right"):
        old = getattr(before.body, name)
        new = getattr(after.body, name)
        target = getattr(action, name)
        if name != "head":
            target = target.pose
        if not old.valid or not new.valid or old.pose is None or new.pose is None:
            return None
        distances.append(
            math.dist(old.pose.position, target.position)
            - math.dist(new.pose.position, target.position)
        )
        if name == "head":

            def angle(pose):
                dot = abs(sum(a * b for a, b in zip(pose.orientation, target.orientation)))
                return 2 * math.acos(min(1.0, dot))

            distances.append(0.2 * (angle(old.pose) - angle(new.pose)))
    return max(-1.0, min(1.0, sum(distances)))


class TrackerLearningStep(Frozen):
    contract: Literal["myumiq-tracker-rates-v1"] = "myumiq-tracker-rates-v1"
    # Flattened PARTS-order xyz linear/angular rates, distinct from output poses.
    rates: tuple[SignedUnit, ...] = Field(min_length=66, max_length=66)
    dt: Number = Field(gt=0, le=0.1)
    policy_observation: tuple[Number, ...] = Field(min_length=1, max_length=4096)
    next_policy_observation: tuple[Number, ...] = Field(min_length=1, max_length=4096)
    observation_contract: str = Field(min_length=1, max_length=100)
    reward_components: dict[str, Number] = Field(min_length=1)
    reward_scope: Literal[
        "tracker_geometry", "avatar_visual", "world_navigation", "partner_feedback"
    ]
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    terminated: bool
    truncated: bool

    @model_validator(mode="after")
    def matched_observations(self):
        if len(self.policy_observation) != len(self.next_policy_observation):
            raise ValueError("policy observation dimensions changed within transition")
        if any(not value.strip() for value in self.evidence_ids):
            raise ValueError("reward evidence reference must not be empty")
        return self


class WholeBodyTransition(Frozen):
    schema_version: Literal[2] = 2
    observation: Observation
    body_goal: BodyGoal
    action: ActuationTarget
    next_observation: Observation
    policy_id: str = Field(min_length=1, max_length=100)
    reward: Number | None = None
    outcome: Literal["simulated", "device_feedback", "unobserved"]
    environment: Literal["mock", "unity", "vrchat"]
    intent_metadata: dict = Field(default_factory=dict)
    learning: TrackerLearningStep | None = None

    @model_validator(mode="after")
    def learning_reward(self):
        if self.learning is not None:
            if self.reward is None or not math.isclose(
                self.reward, sum(self.learning.reward_components.values()), abs_tol=1e-8
            ):
                raise ValueError("learning requires a reward matching its evidence components")
        return self


RECORD = TypeAdapter(Transition | WholeBodyTransition)


class ReplayBuffer(DataBuffer[str, list[Transition | WholeBodyTransition]]):
    """Queue encoded records so long sessions do not retain pose object graphs."""

    def __init__(self, max_size: int):
        super().__init__(max_queue_size=max_size)
        self._buffer = SequentialBuffer[str](max_size)

    @property
    def max_size(self) -> int:
        return self._buffer.max_size

    def add(self, data: str) -> None:
        if len(data) > 131072:
            raise ValueError("oversized replay record")
        # Validate at the buffer boundary; collection already validated the model.
        self._buffer.add(RECORD.validate_json(data).model_dump_json())

    def get_data(self) -> list[Transition | WholeBodyTransition]:
        return [RECORD.validate_json(line) for line in self._buffer.get_data()]

    def __len__(self) -> int:
        return len(self._buffer)

    def encoded_snapshot(self) -> list[str]:
        """Copy immutable records on the collector owner before background I/O."""
        return self._buffer.get_data()

    def save_state(self, path: Path) -> None:
        path = path.with_suffix(".jsonl")
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as f:
            for line in self._buffer.get_data():
                f.write(line + "\n")
        temporary.replace(path)

    def load_state(self, path: Path) -> None:
        # Transactional validation: a malformed record leaves the old buffer intact.
        loaded = SequentialBuffer[str](self.max_size)
        with path.with_suffix(".jsonl").open(encoding="utf-8") as f:
            for line in f:
                if len(line) > 131072:
                    raise ValueError("oversized replay record")
                loaded.add(RECORD.validate_json(line).model_dump_json())
        self._buffer = loaded


def summarize(path: Path) -> dict:
    counts, sources, total = {}, set(), 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            t = RECORD.validate_json(line)
            skill = t.decision.goal.skill if isinstance(t, Transition) else "WHOLE_BODY"
            counts[skill] = counts.get(skill, 0) + 1
            sources.add(t.next_observation.body.head.source)
            total += 1
    return {
        "transitions": total,
        "skills": counts,
        "head_sources": sorted(sources),
        "avatar_visual_confirmation": False,
    }
