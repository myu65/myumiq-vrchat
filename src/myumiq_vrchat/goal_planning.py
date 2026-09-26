"""Slow goal proposals, independent of speech and of finite body action ownership."""

from dataclasses import asdict

from .autonomy import Drives
from .purposes import request_purpose


class GoalPlanning:
    def __init__(self, runner):
        self.runner = runner
        self.pending = None
        self.next_request = 0.0

    def context_key(self):
        runner = self.runner
        commitment = runner.shared.commitment or {}
        decision = getattr(runner, "body_decision", None)
        return (
            runner.owner.control_generation,
            runner.goal_context_revision,
            bool(decision and decision.stop_latched),
            tuple(commitment.get(k) for k in ("id", "status", "criterion", "target")),
        )

    def tick(self, now):
        from .purpose_runtime import background, body_summary

        runner, settings = self.runner, self.runner.settings
        owner = runner.owner
        if self.pending:
            future, started, context_key = self.pending
            if not future.done():
                return
            self.pending = None
            if (
                context_key != self.context_key()
                or not 0 <= now - started <= settings.planner_max_age_s
            ):
                owner.health["planner"] = {"state": "discarded", "reason": "stale_snapshot"}
                self.next_request = now + settings.retry_s
                return
            try:
                purpose = future.result()
                # Use the current capability/target gates, never the remote claim of feasibility.
                resolution = runner.registry.resolve(purpose, owner.world)
                if resolution.missing:
                    queued = runner.queue_learning(purpose)
                    if not queued:
                        raise ValueError("planner proposal has no configured learning route")
                    runner.emit(
                        "planner_learning",
                        capabilities=queued,
                        inference=getattr(purpose, "_inference", {}),
                    )
                    owner.health["planner"] = {"state": "learning_queued", "capabilities": queued}
                    self.next_request = now + settings.planner_interval_s
                    return
                if resolution.blockers:
                    raise ValueError(
                        "planner proposal requires unavailable context or capabilities"
                    )
                if purpose.focus and purpose.criterion in ("observe_target", "interaction"):
                    owner.world.locate(purpose.focus)
                runner.shared.propose_goal(purpose)
                runner.emit(
                    "planner_proposal",
                    purpose=purpose.model_dump(mode="json"),
                    inference=getattr(purpose, "_inference", {}),
                )
                owner.health["planner"] = {"state": "proposed", "model": settings.planner.model}
            except Exception as exc:
                owner.health["planner"] = {"state": "unavailable", "error": str(exc)[:200]}
                runner.emit("planner_failed", error=str(exc)[:200])
            self.next_request = now + settings.planner_interval_s
            return
        if now < self.next_request:
            return
        decision = getattr(runner, "body_decision", None)
        if decision and decision.stop_latched:
            owner.health["planner"] = {"state": "holding", "reason": "explicit_stop"}
            return
        world = owner.world
        kwargs = {}
        if settings.planner_use_image:
            from .visual_context import visual_context

            visual = visual_context(runner.services.vision, world, now, use_image=True)
            world = visual.world
            kwargs["visual"] = visual
        drives, body = Drives(**asdict(owner.drives)), body_summary(owner.snapshot)
        memory, history = list(owner.memory.recent), list(runner.history)
        capabilities, context = runner.registry.summary(), runner.shared.context()
        self.pending = (
            background(
                lambda: request_purpose(
                    settings.planner,
                    world,
                    drives,
                    body,
                    memory,
                    capabilities,
                    history,
                    context,
                    **kwargs,
                )
            ),
            now,
            self.context_key(),
        )
        owner.health["planner"] = {"state": "thinking", "model": settings.planner.model}
