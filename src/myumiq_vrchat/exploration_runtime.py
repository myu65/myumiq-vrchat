"""Slow exploration sensing and memory; no direct device writes."""

from .exploration import (
    DIRECTED_MOVEMENT,
    NAVIGATION_SKILLS,
    ContinuousExplorer,
    Explorer,
    PrivateHomeGate,
)


class ExplorationRuntime:
    def __init__(self, runner):
        self.runner = runner
        self.explorer = (
            ContinuousExplorer()
            if runner.owner.config.exploration.mode == "continuous"
            else Explorer()
        )
        self.gate = PrivateHomeGate(
            runner.owner.root, runner.owner.config.exploration.allow_controlled_home
        )
        self.next_gate = 0.0
        self.allowed = False
        self.gate_pending = None
        self.gate_observation = None

    def _gate_allowed(self, now, configured):
        from .purpose_runtime import background

        if configured and self.gate_pending is None and now >= self.next_gate:
            self.gate_pending = (background(self.gate.valid), now)
            self.next_gate = now + 0.5
        if self.gate_pending and self.gate_pending[0].done():
            future, started = self.gate_pending
            self.gate_pending = None
            try:
                self.gate_observation = (started, future.result() is True)
            except Exception:
                self.gate_observation = (started, False)
        observation = self.gate_observation
        return bool(
            configured and observation and observation[1] and 0 <= now - observation[0] < 0.75
        )

    def tick(self, now):
        runner, owner = self.runner, self.runner.owner
        configured = owner.config.exploration.enabled
        self.allowed = self._gate_allowed(now, configured)
        capability = runner.registry.get("EXPLORE_HOME")
        gait_available = (
            owner.learned_motor is None or "WALK_IN_PLACE" in owner.learned_motor.motions
        )
        visual_at = owner.world.timestamp if runner.services.vision else None
        visual_ready = visual_at is not None and 0 <= now - visual_at < 0.75
        capability.available = (
            self.allowed and gait_available and visual_ready and self.explorer.unchanged < 3
        )
        capability.duration_s = 20.0
        capability.status = (
            owner.config.exploration.mode + "_mapless"
            if capability.available
            else "gate_or_visual_response_unavailable"
        )
        for name in DIRECTED_MOVEMENT:
            item = runner.registry.get(name)
            item.available, item.status = capability.available, capability.status
            item.duration_s = 8.0 if name == "MOVE_FORWARD" else 4.0
        timing = owner.action_timing(now)
        skill = owner.choice[2].skill
        active = (
            self.allowed
            and gait_available
            and self.explorer.unchanged < 3
            and owner.enabled
            and skill in NAVIGATION_SKILLS
            and owner.choice[0] == owner.generation
            and timing["phase"] in ("running", "waiting_observation")
        )
        snapshot = None
        if active and runner.services.vision:
            try:
                snapshot = runner.services.vision.decision_snapshot()
            except Exception:
                pass
        motion_ready = (
            owner.learned_motor is None
            or owner.learned_motor.playback is not None
            and owner.learned_motor.playback.phase > 0
        )
        owner.exploration_lease = self.explorer.tick(
            owner.choice[0],
            now,
            snapshot,
            active,
            motion_ready,
            direction=DIRECTED_MOVEMENT.get(skill),
        )
        while self.explorer.outcomes:
            item = self.explorer.outcomes.pop(0)
            item["intent_id"] = owner.choice[0]
            runner.shared.episode("exploration", item)
            runner.shared.working["exploration"] = item
            runner.shared.save()
            runner.emit("exploration_outcome", **item)
        owner.exploration_status = dict(
            phase=self.explorer.phase,
            cycles=self.explorer.cycles,
            unchanged=self.explorer.unchanged,
            controlled_home_gate=self.allowed,
            home_gate_pending=self.gate_pending is not None,
            home_gate_age_s=now - self.gate_observation[0] if self.gate_observation else None,
            visual_frame_at=visual_at,
            visual_ready=visual_ready,
            motion_ready=motion_ready,
            last=runner.shared.working.get("exploration"),
        )
        owner.health["exploration"] = owner.exploration_status
