"""Slow goal/plan executive using existing motor ownership and PAMIQ metadata."""

import json
import math
import time
import uuid
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path
from threading import Event, Thread

from pydantic import Field, model_validator

from .agent_memory import AgentMemory
from .autonomy import Drives
from .body import Frozen, Number
from .capabilities import CapabilityRegistry, posture_capability
from .capability_learning import LearningSettings, make_task, restore_candidate, train_task
from .exploration import NAVIGATION_SKILLS
from .generation import GenerationSession
from .generation_config import GenerationConfig
from .purposes import fallback_purpose, request_purpose


class PurposeSettings(Frozen):
    state: Path
    memory: Path | None = None
    learning: LearningSettings = LearningSettings()
    max_plan_s: Number = Field(default=120, ge=10, le=300)
    retry_s: Number = Field(default=10, ge=2, le=60)
    planner: GenerationConfig | None = None
    planner_interval_s: Number = Field(default=30, ge=5, le=300)
    planner_max_age_s: Number = Field(default=30, ge=1, le=120)
    planner_use_image: bool = False

    @model_validator(mode="after")
    def external_state(self):
        if self.state.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError("purpose state must be outside repository")
        return self


def background(fn):
    future = Future()

    def work():
        try:
            future.set_result(fn())
        except Exception as exc:
            future.set_exception(exc)

    Thread(target=work, daemon=True).start()
    return future


def body_summary(body):
    if body is None:
        return {}
    return {
        "head_height": body.head.pose.position[2] if body.head.valid else None,
        "pelvis_height": body.pelvis.pose.position[2] if body.pelvis.valid else None,
        "timestamp": body.head.timestamp,
        "source": body.head.source,
        "full_body": all(
            body.signal_for(p).valid for p in ("head", "pelvis", "left_foot", "right_foot")
        ),
    }


def action_summary(item, goal_id, index):
    """Activation identity and context; not a claim of request fulfillment."""
    if item is None:
        return None
    return {
        "action_id": f"{goal_id}:{index}",
        "intent": item["intent"].model_dump(mode="json"),
        "started": item["started"],
        **{
            k: item[k]
            for k in (
                "candidate_id",
                "intent_generation",
                "conversation_epoch",
                "context_utterance_id",
                "decision_stimulus_epoch",
                "target_geometry",
                "execution",
            )
            if k in item
        },
    }


class PurposeRunner:
    def __init__(self, owner, services):
        self.owner, self.services, self.settings = owner, services, owner.config.purpose
        self.registry = CapabilityRegistry(walking=owner.model is not None)
        for name, lesson in self.settings.learning.motions.items():
            item = self.registry.get(name)
            item.method = "imitation"
            item.description = lesson.description or lesson.clip
        self.shared = AgentMemory(
            self.settings.memory or self.settings.state.with_suffix(".sqlite3")
        )
        from .exploration_runtime import ExplorationRuntime

        self.exploration = ExplorationRuntime(self)
        self.utterances = []
        self.listening_until = 0.0
        self.history, self.tasks = [], []
        self.pending = self.training = None
        self.dialogue_pending = None
        self.dialogue_generation = GenerationSession(owner.config.llm)
        self.retired_dialogues = []
        self.reply_stream = None
        self.stream_heard = None
        self.stream_epoch = None
        self.reply_pending = None
        self.dialogue_epoch = 0
        self.dialogue_wait = None
        self.assessment_followup = None
        self.goal_context_revision = 0
        self.latest_audio_input = None
        self.last_proactive_at = -float("inf")
        self.cancel_training = Event()
        self.goal = self.goal_id = None
        self.origin = "local_llm"
        self.actions, self.index, self.running = [], 0, None
        self.epoch = owner.generation
        self.next_request = self.pause_until = 0.0
        self.plan_started = 0.0
        self.route = None
        self.deferrals = 0
        self.plan_verified = True
        from .embodied_decision import EmbodiedDecision

        self.body_decision = EmbodiedDecision(self) if owner.config.decision else None
        from .goal_planning import GoalPlanning

        self.planner = GoalPlanning(self) if self.settings.planner else None
        path = self.settings.state
        if path.exists():
            if path.stat().st_size > 2_000_000:
                raise ValueError("purpose state too large")
            data = json.loads(path.read_text("utf-8"))
            if data.get("version") != 1:
                raise ValueError("unsupported purpose state")
            self.registry.restore_statistics(data.get("capabilities", {}))
            self.history = data.get("history", [])[-30:]
            self.tasks = data.get("learning_tasks", [])[-40:]
            for task in self.tasks:
                if task["status"] in ("queued", "running"):
                    task["status"] = "interrupted"
            restored = set()
            for trained in reversed(self.tasks):
                if trained.get("kind") == "replay_refinement":
                    continue  # Evaluated candidates never become live actors on restore.
                name = trained["capability"]
                if trained["status"] != "trained" or name in restored:
                    continue
                restored.add(name)
                if (
                    owner.learned_motor
                    and name in owner.learned_motor.settings.motions
                    or not owner.learned_motor
                    and owner.model is not None
                    and name == "WALK_IN_PLACE"
                ):
                    continue  # an explicit configured reference takes precedence on startup
                try:
                    model = restore_candidate(trained["result"], path.parent / "learning")
                    self._install_motion(name, model, trained["result"])
                except Exception as exc:
                    trained.update(status="invalid_artifact", error=str(exc)[:200])
        if owner.learned_motor:
            owner.learned_motor.configure_registry(self.registry)
        self.save()

    def _install_motion(self, name, model, report):
        if posture_capability(name):
            raise ValueError(
                "posture references require actor transition/recovery validation and explicit configuration"
            )
        if self.owner.learned_motor:
            from .motion_prior import MotionReference

            reference = MotionReference(
                policy=Path(report["policy"]),
                sha256=report["sha256"],
                playback_rate=report.get("playback_rate", 0.25),
                hand=report.get("hand"),
                description=report.get("description", ""),
            )
            self.owner.learned_motor.install_motion(name, model, reference)
            self.owner.learned_motor.configure_registry(self.registry)
        else:
            from .whole_body import PeriodicImitation

            if name != "WALK_IN_PLACE" or not isinstance(model, PeriodicImitation):
                raise ValueError("this motion requires the articulated motor")
            self.owner.model = model
            self.owner.policy_id = "imitation:" + report["sha256"][:16]
        item = self.registry.get(name)
        item.available, item.status, item.policy = True, "learned", report["policy"]

    def queue_learning(self, purpose):
        """Acquire supported missing motions without replacing the active body goal."""
        queued = []
        resolution = self.registry.resolve(purpose, self.owner.world)
        for name in set(resolution.missing):
            existing = next((t for t in self.tasks if t["capability"] == name), None)
            if existing is not None:
                if existing["status"] in ("queued", "running"):
                    queued.append(name)
                continue
            task = make_task(name, None, self.registry, self.settings.learning)
            if task["blockers"]:
                continue
            if name != "WALK_IN_PLACE" and self.owner.learned_motor is None:
                continue
            task["requested_purpose"] = purpose.model_dump(mode="json")
            self.tasks.append(task)
            self.emit("learning_task", task=task)
            queued.append(name)
        if queued:
            self.save()
        return queued

    def save(self):
        path = self.settings.state
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "version": 1,
                    "capabilities": self.registry.dump(),
                    "history": self.history[-30:],
                    "learning_tasks": self.tasks[-40:],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "utf-8",
        )
        tmp.replace(path)

    def emit(self, kind, **data):
        with (self.owner.root / "goals.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"time": time.time(), "event": kind, "goal_id": self.goal_id, **data},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def finish(self, status, evidence=None):
        if self.goal is not None:
            item = {
                "goal_id": self.goal_id,
                "goal": self.goal.description,
                "status": status,
                "steps_observed": self.index,
                "evidence": evidence,
                "action": action_summary(self.running, self.goal_id, self.index)
                or (evidence.get("action") if isinstance(evidence, dict) else None),
                "scope": "plan_execution_not_unobserved_social_or_contact_success",
            }
            self.history.append(item)
            del self.history[:-30]
            self.owner.memory.recent.append(item)
            del self.owner.memory.recent[:-8]
            self.emit("goal_finished", **{k: v for k, v in item.items() if k != "goal_id"})
            self.owner.memory.save(self.owner.config.memory)
            self.shared.plan_result(status, evidence)
        self.goal = self.goal_id = None
        self.actions, self.running = [], None
        self.owner.purpose_metadata = {}
        self.save()

    def interrupt(self, reason):
        self.goal_context_revision += 1
        self.finish("interrupted", reason)
        self.dialogue_epoch += 1
        self._preempt_dialogue()
        self.utterances.clear()
        self.listening_until = 0.0
        self.owner.attention = None
        self.owner.gesture = None
        if self.body_decision:
            self.body_decision.clear_body_request()
            self.body_decision.conversation_changed()
        # Obsolete speech cannot hold up a new body or conversation request.
        self.epoch = self.owner.generation

    def adopt_operator_body_goal(self):
        """A new body goal preempts body work, without cancelling conversation."""
        self.goal_context_revision += 1
        self.finish("interrupted", "operator_body_goal")
        self.owner.attention = self.owner.gesture = None
        if self.body_decision:
            self.body_decision.clear_body_request()
            self.body_decision.conversation_changed()
            # An older utterance cannot supersede the newly submitted goal.
            self.body_decision.handled_request = next(
                (
                    t["episode_id"]
                    for t in reversed(self.shared.working["turns"])
                    if t["role"] == "user"
                ),
                None,
            )
        self.epoch = self.owner.generation

    def accept(self, goal, now, origin="local_llm", *, retry_utterance_id=None):
        self.origin = origin
        self.goal, self.goal_id = goal, uuid.uuid4().hex
        self.shared.start_plan(goal, self.goal_id, manage_commitment=origin != "body_decision")
        self.index, self.plan_started, self.deferrals = 0, now, 0
        self.plan_verified = True
        self.registry.get("TALK").available = self.services.voice is not None
        self.registry.get("TALK").status = (
            "local_output_only" if self.services.voice else "unavailable"
        )
        resolution = self.registry.resolve(goal, self.owner.world)
        for capability, intent in resolution.actions:
            retry = (
                origin == "body_decision"
                and self.body_decision is not None
                and self.body_decision.retry_available(intent, retry_utterance_id)
            )
            if self.shared.repeated_failure(capability, self.condition(intent)) and not retry:
                resolution.blockers.append("repeated_failure_same_conditions:" + capability)
        if resolution.blockers:
            resolution.actions.clear()
        self.route = resolution.route
        self.emit(
            "goal_created",
            purpose=goal.model_dump(mode="json"),
            origin=origin,
            route=resolution.route,
            blockers=resolution.blockers,
            novelty={
                "missing_capabilities": resolution.missing,
                "unseen_targets": [
                    o.name for o in self.owner.world.objects if o.name not in self.registry.seen
                ],
            },
        )
        if resolution.missing:
            pending = False
            for capability in set(resolution.missing):
                task = next((t for t in self.tasks if t["capability"] == capability), None)
                if task is None:
                    task = make_task(
                        capability, self.goal_id, self.registry, self.settings.learning
                    )
                    task["requested_purpose"] = goal.model_dump(mode="json")
                    self.tasks.append(task)
                elif task["status"] in ("blocked", "interrupted"):
                    refreshed = make_task(
                        capability, self.goal_id, self.registry, self.settings.learning
                    )
                    if not refreshed["blockers"]:
                        task.update(refreshed)
                        task["requested_purpose"] = goal.model_dump(mode="json")
                if task["status"] in ("queued", "running"):
                    pending = True
                self.emit("learning_task", task=task)
            self.save()
            if not pending:
                self.finish(
                    "blocked_learning",
                    resolution.blockers or "learning task requires data/evaluation",
                )
                self.next_request = now + self.settings.retry_s
            return
        if resolution.blockers:
            self.finish("blocked_context", resolution.blockers)
            self.next_request = now + self.settings.retry_s
            return
        self.actions = resolution.actions

    def _learning(self):
        from .replay_learning_job import make_replay_task, train_replay_task

        if (
            self.owner.enabled
            and self.settings.learning.acquire_configured_motions
            and self.owner.learned_motor is not None
            and self.training is None
            and not any(t["status"] in ("queued", "running") for t in self.tasks)
        ):
            for name in self.settings.learning.motions:
                if self.registry.is_available(name) or any(
                    t["capability"] == name for t in self.tasks
                ):
                    continue
                task = make_task(name, None, self.registry, self.settings.learning)
                self.tasks.append(task)
                self.emit("learning_task", task=task, source="configured_motion_discovery")
                self.save()
                break
        refinement = self.settings.learning.replay_refinement
        if (
            refinement is not None
            and self.owner.enabled
            and self.training is None
            and not any(t["status"] in ("queued", "running") for t in self.tasks)
        ):
            task = make_replay_task(refinement)
            if not any(
                t.get("configuration_id") == task["configuration_id"]
                and t["status"] != "interrupted"
                for t in self.tasks
            ):
                self.tasks.append(task)
                self.emit("learning_task", task=task, source="configured_replay_refinement")
                self.save()
        if self.training and self.training[0].done() and self.running is None:
            future, task = self.training
            self.training = None
            try:
                model, report = future.result()
                if task.get("kind") == "replay_refinement":
                    task.update(
                        status=(
                            "candidate_ready"
                            if report["eligible_for_controlled_trial"]
                            else "candidate_rejected"
                        ),
                        result=report,
                    )
                    self.owner.health["real_experience_rl"] = dict(
                        state=task["status"], scope=report["scope"], live_promoted=False
                    )
                    self.emit("learning_result", task=task)
                    self.save()
                    return
                if posture_capability(task["capability"]):
                    # Reference interpolation alone cannot admit an out-of-distribution
                    # posture to a running actor. Keep it for explicit actor evaluation.
                    task.update(
                        status="candidate_pending_validation",
                        result=report,
                        blockers=["actor_transition_recovery_validation_required"],
                    )
                    self.registry.get(task["capability"]).status = "awaiting_actor_validation"
                    if self.goal is not None and self.goal_id == task["goal_id"]:
                        self.finish("blocked_candidate_validation", task["blockers"])
                    self.emit("learning_result", task=task)
                    self.save()
                    return
                self._install_motion(task["capability"], model, report)
                task.update(status="trained", result=report)
                self.shared.learned(task)
                if self.goal_id == task["goal_id"] and self.goal:
                    # Resume the requested plan after learning without requesting training again.
                    resumed = self.goal.model_copy(
                        update={
                            "steps": tuple(
                                s.model_copy(update={"learn": False}) for s in self.goal.steps
                            )
                        }
                    )
                    resolution = self.registry.resolve(resumed, self.owner.world)
                    if not resolution.missing and not resolution.blockers:
                        self.actions = resolution.actions
                    else:
                        self.finish(
                            "blocked_after_learning", resolution.blockers or resolution.missing
                        )
            except Exception as exc:
                task.update(status="failed", error=str(exc)[:250])
                if task.get("kind") == "replay_refinement":
                    self.owner.health["real_experience_rl"] = {
                        "state": "failed",
                        "error": str(exc)[:200],
                    }
                else:
                    self.registry.get(task["capability"]).status = "training_failed"
                if self.goal_id == task["goal_id"]:
                    self.finish("learning_failed", str(exc)[:250])
            self.emit("learning_result", task=task)
            self.save()
        if self.training is None:
            task = next((t for t in self.tasks if t["status"] == "queued"), None)
            if task:
                task["status"] = "running"
                output = self.settings.state.parent / "learning" / task["id"]
                if task.get("kind") == "replay_refinement":
                    self.owner.health["real_experience_rl"] = {
                        "state": "training_candidate",
                        "scope": task["scope"],
                        "live_promoted": False,
                    }
                    self.training = (
                        background(
                            lambda: train_replay_task(refinement, output, self.cancel_training)
                        ),
                        task,
                    )
                else:
                    self.registry.get(task["capability"]).status = "learning"
                    self.training = (
                        background(
                            lambda: train_task(
                                task, self.settings.learning, output, self.cancel_training
                            )
                        ),
                        task,
                    )
                self.save()

    def _observe_step(self, now):
        item = self.running
        timing = self.owner.action_timing(
            now,
            (
                item.get("intent_generation", self.owner.choice[0]),
                item["started"],
                item["intent"],
                self.origin,
            ),
        )
        item["execution"] = timing
        if timing["phase"] in ("preparing", "waiting_observation"):
            return False
        body = self.owner.snapshot
        if body is not None and body.head.valid and 0 <= now - body.head.timestamp < 0.5:
            item["observations"] += 1
            names = (
                (item["intent"].hand,)
                if item["intent"].hand
                else ("left_foot", "right_foot")
                if item["intent"].skill == "WALK_IN_PLACE"
                else ("head",)
            )
            for name in names:
                signal = getattr(body, name)
                if signal.valid and signal.pose and item["start"].get(name) is not None:
                    item["movement"] = max(
                        item["movement"], math.dist(signal.pose.position, item["start"][name])
                    )
            if item["intent"].skill == "LOOK_AT":
                from .gaze import image_target

                try:
                    visual = image_target(self.owner.world, item["intent"].target, now)
                    if visual.last_seen != item.get("visual_frame"):
                        item["visual_frame"] = visual.last_seen
                        item["centred_frames"] = (
                            item.get("centred_frames", 0) + 1
                            if math.hypot(*visual.image_position) < 0.12
                            else 0
                        )
                except ValueError:
                    item["centred_frames"] = 0
        if timing["phase"] == "running":
            return False
        intent, capability = item["intent"], item["capability"]
        outcome = None
        scope = "unavailable"
        learned_evidence = {}
        if timing["phase"] in ("failed", "cancelled"):
            outcome = False if timing["phase"] == "failed" else None
            scope = "motor_execution"
            learned_evidence = {"execution": timing}
        elif timing["phase"] == "completed" and self.owner.learned_motor:
            # Completion carries the exact validated settling observations.
            # A newer snapshot or a later camera gap cannot erase that outcome.
            learned_evidence = self.owner.learned_motor.goal_evidence(
                body, now, item.get("intent_generation")
            )
            outcome, scope = learned_evidence["success"], learned_evidence["scope"]
        elif (
            item["observations"] >= 3 and body is not None and 0 <= now - body.head.timestamp < 0.5
        ):
            scope = "simulated" if body.head.source == "simulated" else "device_execution_only"
            if self.owner.learned_motor and intent.skill not in ("WAIT", *NAVIGATION_SKILLS):
                learned_evidence = self.owner.learned_motor.goal_evidence(
                    body, now, item.get("intent_generation")
                )
                if intent.skill == "LOOK_AT" and self.owner.learned_motor.facing:
                    facing = self.owner.learned_motor.facing.evidence(self.owner.world, now)
                    learned_evidence = {
                        **learned_evidence,
                        **facing,
                        "success": facing["success"]
                        if learned_evidence["success"] is True
                        else learned_evidence["success"],
                    }
                outcome, scope = learned_evidence["success"], learned_evidence["scope"]
            elif (
                intent.skill in ("SIT", "LIE", "CROUCH", "STAND", "RETURN_TO_REST")
                and body.pelvis.valid
            ):
                from .postures import posture_target

                target = posture_target(
                    {"SIT": "sitting_floor", "LIE": "lying", "CROUCH": "crouching"}.get(
                        intent.skill, "standing"
                    )
                )
                outcome = math.dist(body.pelvis.pose.position, target.pelvis.position) < 0.12
            elif intent.skill in ("WAVE", "WALK_IN_PLACE"):
                outcome = item["movement"] > 0.015
            elif intent.skill in ("LOOK_AT", "REACH"):
                try:
                    intent.validate_world(self.owner.world)
                    from .gaze import yaw_pitch

                    target = self.owner.world.locate(intent.target)
                    signal = body.head if intent.skill == "LOOK_AT" else getattr(body, intent.hand)
                    if signal.valid and signal.pose is not None:
                        if intent.skill == "REACH":
                            outcome = math.dist(signal.pose.position, target) < 0.08
                        else:
                            obj = next(
                                o for o in self.owner.world.objects if o.name == intent.target
                            )
                            if obj.source == "vision":
                                from .gaze import image_target

                                image_target(self.owner.world, intent.target, now)
                                outcome = item.get("centred_frames", 0) >= 2
                                scope = "fresh_visual_alignment"
                            else:
                                d = [a - b for a, b in zip(target, self.owner.rest.head.position)]
                                desired = (
                                    max(-0.8, min(0.8, math.atan2(d[1], d[0]))),
                                    max(-0.5, min(0.5, -math.atan2(d[2], math.hypot(*d[:2])))),
                                )
                                outcome = (
                                    max(abs(a - b) for a, b in zip(yaw_pitch(signal.pose), desired))
                                    < 0.12
                                )
                except ValueError:
                    outcome = None
            elif intent.skill == "WAIT" and capability != "TALK":
                outcome, scope = True, "wait_duration_observed"
            elif intent.skill in NAVIGATION_SKILLS:
                last = self.owner.exploration_status.get("last")
                if last and last["timestamp"] >= item["started"]:
                    outcome = last["outcome"] == "changed_view"
                    scope = "image_change_not_metric_displacement"
        self.registry.observe(capability, outcome, scope)
        if outcome and intent.target:
            self.registry.seen[intent.target] = self.registry.seen.get(intent.target, 0) + 1
        evidence = {
            "capability": capability,
            "success": outcome,
            "scope": scope,
            "movement_m": item["movement"],
            "observations": item["observations"],
            "action": action_summary(item, self.goal_id, self.index),
            **learned_evidence,
        }
        condition = item.get("condition", self.condition(intent))
        self.shared.skill_outcome(capability, condition, outcome, scope, evidence, intent.target)
        self.emit("step_outcome", step=self.index, evidence=evidence)
        if self.body_decision:
            self.body_decision.mark_request(
                item.get("context_utterance_id"),
                execution="completed"
                if outcome is True
                else "failed"
                if outcome is False
                else "unverified",
                skill=capability,
                evidence_scope=scope,
            )
        self.owner.memory.recent.append({"goal_id": self.goal_id, "skill": capability, **evidence})
        del self.owner.memory.recent[:-8]
        self.owner.memory.save(self.owner.config.memory)
        self.running = None
        if item["deferred"]:
            self.deferrals += 1
            if self.deferrals >= 3:
                self.finish("deferred_replan", evidence)
        elif timing["phase"] == "cancelled":
            self.finish("interrupted", evidence)
        elif outcome is False:
            self.finish("execution_failed", evidence)
        else:
            self.plan_verified = self.plan_verified and outcome is True
            self.index += 1
            if self.index >= len(self.actions):
                self.finish(
                    "plan_completed" if self.plan_verified else "plan_completed_unverified",
                    evidence,
                )
        self.save()
        return True

    def _score(self, intent):
        # Compatibility path only: configured Decision Layers own body selection.
        return intent, None

    def tick(self, now):
        owner = self.owner
        self._learning()
        if owner.learned_motor:
            owner.learned_motor.configure_registry(self.registry)
        if owner.generation != self.epoch:
            if owner.choice[3] == "operator_body_goal":
                self.adopt_operator_body_goal()
            else:
                self.interrupt("manual_or_voice_preemption")
        self.exploration.tick(now)
        talk = self.registry.get("TALK")
        talk.available = self.services.voice is not None
        talk.status = "local_output_only" if talk.available else "unavailable"
        self.owner.memory_metadata = {
            "revision": self.shared.revision,
            "commitment_id": self.shared.commitment["id"] if self.shared.commitment else None,
            "topic": self.shared.working["topic"],
            "partner": self.shared.working["partner"],
            "attention": self.shared.working["attention"],
        }
        owner.health["purpose"] = {
            "goal_id": self.goal_id,
            "goal": self.goal.description if self.goal else None,
            "step": self.index,
            "route": self.route,
            "origin": self.origin,
            "steps": len(self.actions),
            "capabilities": self.registry.summary(),
            "learning_tasks": self.tasks[-5:],
        }
        owner.health["memory"] = {
            **self.owner.memory_metadata,
            "commitment": self.shared.commitment,
            "conversation_turns": len(self.shared.working["turns"]),
        }
        if owner.enabled:
            self._dialogue_tick(now)
        if not owner.enabled:
            return
        if self.body_decision:
            if self.planner:
                self.planner.tick(now)
            self.body_decision.tick(now)
            return
        if now < self.pause_until:
            return
        if self.goal and now - self.plan_started > self.settings.max_plan_s:
            self.finish("plan_timeout")
        if self.running:
            self._observe_step(now)
            return
        if self.pending:
            future, kind, epoch = self.pending
            if not future.done():
                return
            self.pending = None
            if epoch != owner.generation:
                return
            try:
                value = future.result()
                if kind == "purpose":
                    owner.health["llm"] = {"state": "running", "model": owner.config.llm.model}
                    self.accept(value, now)
                elif self.goal and self.index < len(self.actions):
                    intent, ranking = value
                    if ranking and ranking["state"] == "running":
                        if now - ranking["result"]["captured_at"] > owner.config.decision.max_age_s:
                            raise ValueError("decision expired before plan execution")
                    intent.validate_world(owner.world)
                    if ranking:
                        owner.health["decision"] = ranking
                    source = (
                        "local_llm_plan"
                        if self.origin == "local_llm"
                        else "operator_goal_plan"
                        if self.origin == "operator_goal"
                        else "drive_fallback_plan"
                    )
                    if ranking and ranking["state"] == "running":
                        source += "+decision:" + ranking["result"]["backend"]
                    capability, planned = self.actions[self.index]
                    deferred = intent != planned
                    owner.purpose_metadata = {
                        "goal_id": self.goal_id,
                        "purpose": self.goal.description,
                        "success_description": self.goal.success_description,
                        "step": self.index,
                        "capability": capability,
                        "deferred": deferred,
                    }
                    owner._choose(now, intent, source)
                    self.epoch = owner.generation
                    body = owner.snapshot
                    start = {
                        name: getattr(body, name).pose.position
                        for name in ("head", "left", "right", "left_foot", "right_foot")
                        if body and getattr(body, name).valid
                    }
                    self.running = {
                        "intent": intent,
                        "capability": "WAIT" if deferred else capability,
                        "started": now,
                        "start": start,
                        "movement": 0.0,
                        "observations": 0,
                        "deferred": deferred,
                        "intent_generation": owner.choice[0],
                        "condition": self.condition(intent),
                    }
                    if intent.speech:
                        self.services.speak(intent.speech, now)
            except Exception as exc:
                if kind == "purpose":
                    owner.health["llm"] = {"state": "degraded", "error": str(exc)[:200]}
                    self.accept(fallback_purpose(owner.drives, owner.world), now, "drive_fallback")
                else:
                    self.finish("plan_invalidated", str(exc)[:200])
                self.next_request = now + self.settings.retry_s
            return
        if self.goal:
            if self.index < len(self.actions):
                intent = self.actions[self.index][1]
                self.scoring_context = self.shared.context()
                self.pending = (background(lambda: self._score(intent)), "step", owner.generation)
            return
        if now >= self.next_request:
            world, drives = owner.world, Drives(**asdict(owner.drives))
            body, memory = body_summary(owner.snapshot), list(owner.memory.recent)
            caps, history = self.registry.summary(), list(self.history)
            context = self.shared.context()
            self.pending = (
                background(
                    lambda: request_purpose(
                        owner.config.llm, world, drives, body, memory, caps, history, context
                    )
                ),
                "purpose",
                owner.generation,
            )
            owner.health["llm"] = {"state": "planning", "model": owner.config.llm.model}

    def close(self):
        if self.body_decision:
            self.body_decision.close()
        self._discard_stream("shutdown")
        self.dialogue_generation.cancel()
        self.dialogue_generation.close()
        for _, session in self.retired_dialogues:
            session.cancel()
            session.close()
        self.cancel_training.set()
        if self.training:
            try:
                self.training[0].result(timeout=3)
            except Exception:
                pass
            self.training[1]["status"] = "interrupted"
        self.finish("interrupted", "shutdown")
        self.shared.close()

    def condition(self, intent):
        body = self.owner.snapshot
        target = next((o for o in self.owner.world.objects if o.name == intent.target), None)
        height = body.head.pose.position[2] if body and body.head.valid else None
        condition = {
            "target_source": target.source if target else "none",
            "height": "low" if height is not None and height < 1.2 else "standing_or_unknown",
            "hand": intent.hand,
            "policy": self.owner.learned_motor.policy_id
            if self.owner.learned_motor
            else self.owner.policy_id
            if intent.skill == "WALK_IN_PLACE"
            else "procedural",
            "distance": "near"
            if target and math.dist(target.position, (0.0, 0.0, 1.6)) < 0.65
            else "far_or_none",
        }
        if self.owner.learned_motor:
            condition.update(self.owner.learned_motor.task_identity(intent))
        if intent.skill == "LOOK_AT" and target and target.source == "vision":
            # Detector IDs are ephemeral. Compare the actual control conditions,
            # not identity churn or failures of the old absolute-coordinate motor.
            condition.update(
                visual_control="image_feedback_hold_v2",
                gaze_policy=self.owner.gaze_policy.model_dump(mode="json")
                if self.owner.gaze_policy
                else "procedural",
                duration_s=intent.duration_s,
                image_region=tuple(math.floor(v / 0.25) for v in target.image_position)
                if target.image_position
                else None,
                confident=target.confidence >= 0.5,
            )
        return json.dumps(condition, sort_keys=True)

    def on_event(self, event, now):
        if (
            event.kind in ("asr_no_speech", "asr_error", "audio_input_overflow")
            and self.owner.enabled
        ):
            self.shared.episode(
                event.kind,
                {
                    "input_id": event.utterance_id,
                    "audio_end_at": event.audio_end_at,
                    "detail": event.text,
                },
            )
            self.owner.health["audio_input"] = {"state": event.kind, "detail": event.text}
            captured = (
                {"session": event.input_session, "sequence": event.input_sequence}
                if event.input_session is not None and event.input_sequence is not None
                else None
            )
            if self._input_is_current({"input": captured}):
                self.listening_until = now
                self.shared.working.pop("partial_transcript", None)
                self.shared.working["attention"] = "environment"
            self.shared.save()
            return
        if event.kind == "explore" and self.owner.enabled:
            from .autonomous_body import Intent
            from .purposes import Purpose, SkillRequest

            self.owner._choose(now, Intent(skill="WAIT", duration_s=1), "operator_exploration_goal")
            self.interrupt("new_exploration_goal")
            self.accept(
                Purpose(
                    description="private Homeを探索して、まだ見ていない景色を知りたい",
                    reason="探索の目的を与えられた",
                    success_description="移動前後の景色を観測し、探索履歴を残す",
                    continuity="replace",
                    steps=(SkillRequest(capability="EXPLORE_HOME", duration_s=20),),
                ),
                now,
                origin="operator_goal",
            )
            return
        if event.kind in ("speech_active", "partial_transcript") and self.owner.enabled:
            self.listening_until = now + 15
            self.shared.working["attention"] = "conversation"
            if event.kind == "partial_transcript":
                self.shared.working["partial_transcript"] = event.text
                self.owner.health["partial_asr"] = {"state": "provisional", "source": event.source}
            return
        if event.kind not in ("speech_started", "utterance") or not self.owner.enabled:
            return
        if event.kind == "speech_started":
            if event.input_session is not None and event.input_sequence is not None:
                self.latest_audio_input = (event.input_session, event.input_sequence)
            # These inputs are already in shared memory. Only their pending
            # response requests are obsolete; the body goal is untouched.
            self.utterances.clear()
            self.emit("speech_attention", capture_to_executive_s=max(0, now - event.timestamp))
        target = event.target or self.shared.working["focus"]
        if event.kind == "utterance" and event.text:
            input_context = None
            if event.input_session is not None and event.input_sequence is not None:
                input_context = {
                    "session": event.input_session,
                    "sequence": event.input_sequence,
                    "utterance_id": event.utterance_id,
                    "audio_start_at": event.audio_start_at,
                    "audio_end_at": event.audio_end_at,
                }
                if self.latest_audio_input is None or (
                    self.latest_audio_input[0] == event.input_session
                    and event.input_sequence > self.latest_audio_input[1]
                ):
                    self.latest_audio_input = (event.input_session, event.input_sequence)
            binding = self.owner.speaker_binding
            if self._input_is_current({"input": input_context}):
                self._preempt_dialogue()  # Record any submitted prefix before the new user turn.
            partner = event.speaker_id
            if partner is None and binding and now < binding["expires"]:
                partner = binding["id"]
                target = binding.get("target") or target
            heard = self.shared.hear(
                event.text,
                partner=partner,
                source=event.source,
                focus=target,
                input_context=input_context,
            )
            reply_eligible = self._input_is_current(heard)
            self.emit(
                "utterance_received",
                episode_id=heard["episode_id"],
                partner=partner,
                source=event.source,
                input=input_context,
                reply_eligible=reply_eligible,
            )
            if not reply_eligible:
                # This is still a heard fact, but no new conversational turn.
                # In particular, it must not cancel a newer pending reply.
                return
            self.owner.health["audio_input"] = {"state": "recognized", "input": input_context}
            self.goal_context_revision += 1
            self.utterances[:] = [heard]
            if self.body_decision:
                self.body_decision.conversation_changed()
                from .shared_dialogue import is_immediate_stop

                if is_immediate_stop(event.text):
                    from .autonomous_body import Intent

                    self.owner.exploration_lease = None
                    self.owner._choose(now, Intent(skill="WAIT"), "local_explicit_stop")
                    self.finish("interrupted", "local_explicit_stop")
                    self.epoch = self.owner.generation
                    self.body_decision.assess_request(
                        {"request_status": "stop", "request_reason": "明示された停止"},
                        heard["episode_id"],
                        now,
                    )
                    self.body_decision.mark_request(
                        heard["episode_id"], execution="accepted", skill="WAIT"
                    )
                    self.emit("local_stop_accepted", episode_id=heard["episode_id"])
        self.shared.working.pop("partial_transcript", None)
        self.dialogue_epoch += 1
        self._preempt_dialogue()
        self.listening_until = now + 15
        self.shared.working["attention"] = "conversation"
        if self.services.voice:
            self.services.voice.pipeline.output.stop()
        self.shared.save()

    def queue_autonomous_dialogue(self, purpose, now):
        """A thought may request speech, independently of the current body goal."""
        settings = self.owner.config.dialogue
        requested = ([purpose.comment] if purpose.comment else []) + [
            s.speech for s in purpose.steps if s.capability == "TALK" and s.speech
        ]
        if (
            not requested
            or not settings.proactive_speech
            or not self.owner.enabled
            or self.services.voice is None
            or now < self.listening_until
            or self.utterances
            or self.dialogue_pending
            or self.reply_pending
            or now - self.last_proactive_at < settings.proactive_interval_s
            or self.body_decision
            and self.body_decision.stop_latched
        ):
            return False
        self.last_proactive_at = now
        intention = dict(purpose=purpose.description, proposal=requested[0], source="planner")
        episode = self.shared.episode("speech_intention", intention)
        self.utterances.append(
            dict(
                episode_id=episode,
                partner=None,
                autonomous=True,
                expires_at=now + 20,
                text=json.dumps(intention, ensure_ascii=False),
            )
        )
        self.emit("autonomous_speech_requested", episode_id=episode, **intention)
        return True

    def _input_is_current(self, heard):
        if heard.get("autonomous"):
            return time.perf_counter() < heard["expires_at"] and not (
                self.body_decision and self.body_decision.stop_latched
            )
        captured = heard.get("input")
        if captured is None:
            return True  # Existing text/operator sources have no audio session.
        identity = (captured["session"], captured["sequence"])
        voice = self.services.voice
        if voice is not None:
            return voice.pipeline.input_is_current(*identity)
        return self.latest_audio_input == identity

    def _dialogue_tick(self, now):
        """Dialogue progresses independently; only the executive commits shared memory."""
        owner = self.owner
        if self.dialogue_pending and self.dialogue_pending[2] != self.dialogue_epoch:
            self._preempt_dialogue()
        self.retired_dialogues[:] = [
            (future, session) for future, session in self.retired_dialogues if not future.done()
        ]
        self._submit_reply(now)
        followup = self.assessment_followup
        if followup:
            heard, epoch, deadline = followup
            if (
                epoch != self.dialogue_epoch
                or now >= deadline
                or not owner.enabled
                or not self._input_is_current(heard)
            ):
                self.assessment_followup = None
            elif not self.dialogue_pending and not self.reply_pending and not self.utterances:
                assessment = self.body_decision.request_assessment if self.body_decision else None
                if assessment and assessment["utterance_id"] == heard["episode_id"]:
                    self.assessment_followup = None
                    self.utterances.append(heard)
                    self.dialogue_wait = None
                    self.emit("dialogue_assessment_followup", episode_id=heard["episode_id"])
        active = (
            self.utterances
            or self.reply_pending
            or (self.dialogue_pending and self.dialogue_pending[1] == "dialogue")
            or now < self.listening_until
        )
        if not active:
            if self.shared.working["attention"] == "conversation":
                self.shared.working["attention"] = "environment"
            return False
        if self.dialogue_pending:
            future, kind, epoch = self.dialogue_pending
            if not future.done():
                self._pump_sentence(now)
                return True
            self.dialogue_pending = None
            if kind == "dialogue" and epoch == self.dialogue_epoch:
                heard = self.reply_to
                try:
                    value = future.result()
                    if self.reply_stream is not None:
                        self.reply_stream.finish(value.reply)
                    self.reply_pending = (
                        value,
                        heard,
                        epoch,
                        now + (20.0 if self.reply_stream else 5.0),
                    )
                    self._submit_reply(now)
                except Exception as exc:
                    self._discard_stream("generation_failed")
                    self.shared.episode(
                        "dialogue_failed", {"error": str(exc)[:180]}, heard["partner"]
                    )
                    owner.health["dialogue"] = {"state": "failed", "error": str(exc)[:200]}
                self.listening_until = now
                self.shared.working["attention"] = "environment"
                self.shared.save()
        if self.dialogue_pending is None and self.reply_pending is None and self.utterances:
            from .shared_dialogue import request_dialogue

            heard = self.utterances[0]
            from .shared_dialogue import DialogueSettings

            settings = getattr(owner.config, "dialogue", DialogueSettings())
            if len(self.retired_dialogues) >= 2:
                owner.health["dialogue"] = {"state": "waiting_adapter_cancellation"}
                return True  # Bound non-cooperative extension workers; coalesce new replies.
            assessment = self.body_decision.request_assessment if self.body_decision else None
            selection = owner.config.decision.selection if owner.config.decision else None
            if self.dialogue_wait is None or self.dialogue_wait[0] != heard["episode_id"]:
                self.dialogue_wait = (heard["episode_id"], now)
            if (
                settings.await_body_assessment
                and not heard.get("autonomous")
                and selection
                and selection.understand_requests
                and (not assessment or assessment["utterance_id"] != heard["episode_id"])
                and now - self.dialogue_wait[1] < settings.action_wait_s
            ):
                owner.health["dialogue"] = {"state": "waiting_body_assessment"}
                return True
            self.utterances.pop(0)
            if not self._input_is_current(heard):
                return True
            self.reply_to = heard
            context = self.shared.context(heard["text"])
            from .visual_context import visual_context

            visual = visual_context(
                self.services.vision,
                owner.world,
                now,
                use_image=settings.use_image,
                max_age_s=settings.max_image_age_s,
            )
            world, drives, body = visual.world, asdict(owner.drives), body_summary(owner.snapshot)
            action_context = (
                dict(assessment)
                if assessment and assessment["utterance_id"] == heard["episode_id"]
                else {
                    "status": ("pending" if settings.await_body_assessment else "parallel_pending")
                    if selection and selection.understand_requests
                    else "not_assessed"
                }
            )
            action_context["current_action"] = action_summary(
                self.running, self.goal_id, self.index
            )
            capabilities = self.registry.summary()
            self.emit(
                "dialogue_request",
                episode_id=heard["episode_id"],
                model=owner.config.llm.model,
                text=heard["text"],
                world=world.model_dump(mode="json"),
                drives=drives,
                body=body,
                context=context,
                visual=visual.metadata(now),
                action_context=action_context,
            )
            from .dialogue_stream import SentenceReply

            stream = SentenceReply() if settings.stream_sentences and self.services.voice else None
            self.reply_stream, self.stream_heard, self.stream_epoch = (
                stream,
                heard,
                self.dialogue_epoch,
            )
            generation = self.dialogue_generation  # Immutable request owner across preemption.
            self.dialogue_pending = (
                background(
                    lambda: request_dialogue(
                        owner.config.llm,
                        world,
                        drives,
                        body,
                        context,
                        heard["text"],
                        generation=generation,
                        visual=visual,
                        capabilities=capabilities,
                        action_context=action_context,
                        **({"on_text": stream.feed} if stream else {}),
                        **({"autonomous": True} if heard.get("autonomous") else {}),
                    )
                ),
                "dialogue",
                self.dialogue_epoch,
            )
            owner.health["dialogue"] = {
                "state": "thinking",
                "memory_revision": self.shared.revision,
            }
        return True

    def _discard_stream(self, reason):
        stream, heard = self.reply_stream, self.stream_heard
        if stream is not None and stream.submitted:
            self.shared.reply(stream.submitted, "会話", (), heard, True)
            self.emit(
                "dialogue_partial_reply",
                episode_id=heard["episode_id"],
                submitted_text=stream.submitted,
                reason=reason,
                delivery="unverified",
            )
        self.reply_stream = self.stream_heard = self.stream_epoch = None

    def _preempt_dialogue(self):
        self._discard_stream("preempted")
        self.reply_pending = self.assessment_followup = None
        if self.dialogue_pending is not None:
            future, _, _ = self.dialogue_pending
            session = self.dialogue_generation
            session.cancel()
            session.close()
            self.retired_dialogues.append((future, session))
            self.dialogue_pending = None
            self.dialogue_generation = GenerationSession(self.owner.config.llm)
            self.emit("dialogue_generation_preempted")

    def _pump_sentence(self, now):
        stream, heard = self.reply_stream, self.stream_heard
        if stream is None:
            return
        if (
            self.stream_epoch != self.dialogue_epoch
            or not self.owner.enabled
            or not self._input_is_current(heard)
        ):
            self._discard_stream("preempted")
            return
        sentence = stream.peek()
        if sentence is None:
            return
        try:
            submitted = self.services.reply(
                sentence, now, input_context=heard.get("input"), wait_until_idle=True
            )
        except Exception as exc:
            self.owner.health["dialogue"] = {"state": "pending_output", "error": str(exc)[:200]}
            return
        if submitted:
            stream.acknowledge()
            self.emit(
                "dialogue_sentence_submitted",
                episode_id=heard["episode_id"],
                text=sentence,
                delivery="unverified",
            )

    def _submit_reply(self, now):
        if self.reply_pending is None:
            return
        value, heard, epoch, deadline = self.reply_pending
        if (
            epoch != self.dialogue_epoch
            or not self.owner.enabled
            or not self._input_is_current(heard)
        ):
            self.reply_pending = None
            self.emit(
                "dialogue_reply_discarded", episode_id=heard["episode_id"], reason="preempted"
            )
            self._discard_stream("preempted")
            return
        if now < deadline:
            self._pump_sentence(now)
        submitted = False
        error = None
        stream = self.reply_stream
        if stream is not None:
            submitted = stream.submitted == value.reply
        elif now < deadline:
            try:
                if heard.get("input") is not None and self.services.voice is not None:
                    submitted = self.services.reply(value.reply, now, input_context=heard["input"])
                else:
                    submitted = self.services.reply(value.reply, now)
            except Exception as exc:
                error = str(exc)[:200]
        if not submitted and now < deadline:
            self.owner.health["dialogue"] = {"state": "pending_output", "error": error}
            return
        self.reply_pending = None
        recorded = stream.submitted if stream is not None else value.reply
        if recorded:
            self.shared.reply(
                recorded,
                value.topic,
                value.remember_quotes,
                heard,
                bool(recorded) if stream is not None else submitted,
            )
        self.reply_stream = self.stream_heard = self.stream_epoch = None
        if getattr(value, "_inference", {}).get("adapter") == "pending_assessment_acknowledgement":
            from .shared_dialogue import DialogueSettings

            settings = getattr(self.owner.config, "dialogue", DialogueSettings())
            if settings.action_followup_s:
                self.assessment_followup = (heard, epoch, now + settings.action_followup_s)
        self.emit(
            "dialogue_reply",
            reply=recorded,
            episode_id=heard["episode_id"],
            generated_reply=value.reply,
            inference=getattr(value, "_inference", {}),
            submitted=submitted,
            output_state="submitted" if submitted else "expired",
            memory_revision=self.shared.revision,
            commitment_id=self.shared.commitment["id"] if self.shared.commitment else None,
        )
        self.owner.health["dialogue"] = {
            "state": "submitted" if submitted else "output_expired",
            "memory_revision": self.shared.revision,
        }


def run_purposes(owner):
    from .autonomous_services import Services

    services = Services(owner.config, owner.health, lambda: owner.world)
    runner = None
    previous = time.perf_counter()
    try:
        runner = PurposeRunner(owner, services)
        while not owner.stopped.wait(0.05):
            now = time.perf_counter()
            owner.world, events = services.poll(now, owner.enabled)
            active = (
                owner.enabled
                and owner.action_timing(now)["phase"] == "running"
                and owner.choice[2].skill not in ("WAIT", "SIT", "LIE")
            )
            # Moving alone is not a conversation. Otherwise autonomous walking
            # continually satisfies social desire without any speech occurring.
            observation = owner.health.get("exploration", {}).get("last") or {}
            seen = observation.get("timestamp", -float("inf"))
            owner.drives.advance(
                min(1, now - previous),
                interacting=owner.enabled and now < runner.listening_until,
                moving=active,
                observing=owner.enabled
                and 0 <= now - seen < 1.5
                and observation.get("outcome") == "changed_view",
            )
            previous = now
            for event in events:
                if not owner.enabled:
                    continue
                if event.kind in (
                    "speech_started",
                    "speech_active",
                    "partial_transcript",
                    "utterance",
                    "asr_no_speech",
                    "asr_error",
                    "audio_input_overflow",
                ):
                    runner.on_event(event, now)
                    continue
                if event.kind == "conversation_decision":
                    runner.emit(
                        "ignored_conversation_control", reason="speech_has_no_body_authority"
                    )
            for event in owner.external_events.drain():
                runner.on_event(event, now)
            runner.tick(now)
    except Exception as exc:
        owner.error = str(exc)
        owner.health["purpose"] = {"state": "failed", "error": str(exc)[:250]}
        owner.enabled = False
    finally:
        if runner:
            runner.close()
        services.close()
        owner.memory.save(owner.config.memory)
