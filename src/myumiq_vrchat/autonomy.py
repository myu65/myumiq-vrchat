"""Slow, bounded deliberation state; never runs inside the motor policy."""

import json
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

from .body import WorldState
from .cognition import Decision, Goal, LLMConfig, decide_async


@dataclass
class Drives:
    curiosity: float = 0.5
    boredom: float = 0.0
    social_desire: float = 0.55
    fatigue: float = 0.0

    def advance(self, seconds: float, *, interacting: bool) -> None:
        if not 0 <= seconds <= 1:
            raise ValueError("drive update interval outside 0..1s")
        self.boredom = min(
            1.0, max(0.0, self.boredom + seconds * (-0.08 if interacting else 0.015))
        )
        self.curiosity = min(
            1.0, max(0.0, self.curiosity + seconds * (0.005 if not interacting else -0.03))
        )
        self.social_desire = min(
            1.0, max(0.0, self.social_desire + seconds * (0.003 if not interacting else -0.04))
        )
        self.fatigue = min(
            1.0, max(0.0, self.fatigue + seconds * (0.015 if interacting else -0.02))
        )


@dataclass
class InteractionMemory:
    recent: list[dict] = field(default_factory=list)
    familiarity: dict[str, float] = field(default_factory=dict)
    affinity: dict[str, float] = field(default_factory=dict)

    def record(self, decision: Decision, outcome: str, reward: float) -> None:
        self.recent.append(
            {"skill": decision.goal.skill, "outcome": outcome, "reward": round(reward, 3)}
        )
        del self.recent[:-8]

    def record_interaction(self, player: str, *, positive: bool | None = None) -> None:
        """Only an actual identified interaction changes social affinity."""
        if not player or len(player) > 80:
            raise ValueError("invalid player identifier")
        self.familiarity[player] = min(1.0, max(0.0, self.familiarity.get(player, 0.0)) + 0.05)
        if positive is not None:
            delta = 0.05 if positive else -0.05
            self.affinity[player] = min(1.0, max(-1.0, self.affinity.get(player, 0.0) + delta))

    @classmethod
    def load(cls, path: Path) -> "InteractionMemory":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if set(data) not in ({"recent", "familiarity"}, {"recent", "familiarity", "affinity"}):
            raise ValueError("invalid interaction memory fields")
        recent = data["recent"]
        familiarity = data["familiarity"]
        affinity = data.get("affinity", {})
        if (
            not isinstance(recent, list)
            or len(recent) > 8
            or not isinstance(familiarity, dict)
            or len(familiarity) > 200
            or any(not isinstance(key, str) or not key or len(key) > 80 for key in familiarity)
            or any(
                not isinstance(value, (int, float)) or not -1 <= value <= 1
                for value in familiarity.values()
            )
        ):
            raise ValueError("invalid interaction memory values")
        if (
            not isinstance(affinity, dict)
            or len(affinity) > 200
            or any(not isinstance(key, str) or not key or len(key) > 80 for key in affinity)
            or any(
                not isinstance(value, (int, float)) or not -1 <= value <= 1
                for value in affinity.values()
            )
        ):
            raise ValueError("invalid affinity values")
        return cls(
            recent=recent,
            familiarity={key: float(value) for key, value in familiarity.items()},
            affinity={key: float(value) for key, value in affinity.items()},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "recent": self.recent[-8:],
                    "familiarity": self.familiarity,
                    "affinity": self.affinity,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)


class AutonomousPlanner:
    """One outstanding LLM request at a time; expired actions hold observed pose."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        initial_instruction: str = "内部状態の条件に従い、次の小さな行動を選ぶ",
        memory: InteractionMemory | None = None,
    ):
        self.config = config
        self.initial_instruction = initial_instruction
        self.drives = Drives()
        self.memory = memory or InteractionMemory()
        self.pending: Future[Decision] | None = None
        self.last_decision: Decision | None = None
        self.last_started: float | None = None
        self.next_request_at = 0.0
        self.requests = 0
        self.last_reward = 0.0

    def observe_reward(self, reward: float) -> None:
        self.last_reward += reward

    def interrupt(self) -> None:
        """Discard pre-conversation plans without waiting on their HTTP worker."""
        if self.last_decision is not None:
            self.memory.record(self.last_decision, "interrupted", self.last_reward)
        self.pending = None
        self.last_decision = None
        self.last_started = None
        self.last_reward = 0.0

    def tick(self, now: float, world: WorldState, *, dt: float) -> Decision | None:
        interacting = (
            self.last_decision is not None
            and self.last_decision.goal.skill not in ("WAIT", "RETURN_TO_REST")
            and self.last_started is not None
            and now - self.last_started < self.last_decision.goal.duration_s
        )
        self.drives.advance(min(max(dt, 0.0), 1.0), interacting=interacting)
        if self.pending is not None:
            if not self.pending.done():
                return None
            decision = self.pending.result()
            self.pending = None
            decision.goal.validate_world(world)
            self.last_decision, self.last_started = decision, now
            return decision
        if self.last_decision is not None and self.last_started is not None:
            if now - self.last_started < self.last_decision.goal.duration_s:
                return None
        if now < self.next_request_at:
            return None
        if self.last_decision is not None:
            outcome = "observed_progress" if self.last_reward > 0 else "duration_elapsed"
            self.memory.record(self.last_decision, outcome, self.last_reward)
            self.last_reward = 0.0
        instruction = (
            f"{self.initial_instruction}\n"
            "対象がなくfatigueが低く、social_desireが0.50以上またはboredomが0.55以上なら右手のWAVEを選ぶ。"
            f"内部状態: curiosity={self.drives.curiosity:.2f}, boredom={self.drives.boredom:.2f}, "
            f"social_desire={self.drives.social_desire:.2f}, fatigue={self.drives.fatigue:.2f}. "
            f"最近の経験: {self.memory.recent[-3:]}. 接触の親しさ: {self.memory.familiarity}. "
            f"好意: {self.memory.affinity}. "
            "疲れていたらWAIT。既知の対象だけを選ぶ。"
            "1つの安全な身体行動を選ぶ。"
        )
        self.pending = decide_async(self.config, instruction[:2000], world)
        self.requests += 1
        self.next_request_at = now + 0.5
        return Decision(goal=Goal(skill="WAIT", duration_s=1.0), source="fixed")
