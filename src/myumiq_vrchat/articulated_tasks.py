"""Explicit high-level task goals executed by one feedback-conditioned body actor."""

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .articulated_controller import ArticulatedController
from .body import BodyTarget, Controls, Frozen, Number, WorldState, qmul, rotate
from .body_facing import BodyFacing
from .capabilities import motion_capability, posture_capability
from .cli import outside_repo
from .motion_prior import MotionPlayback, MotionReference, heading, validate_motion
from .tracker_action import CONTRACT
from .tracker_policy import pose_error
from .whole_body import PARTS, state_target, target_from_vector, vector


class ArticulatedTasks(Frozen):
    version: int = Field(default=1, ge=1, le=1)
    actor: Path
    actor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rig_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_floor: Number
    goals: dict[str, BodyTarget] = Field(default_factory=dict, max_length=32)
    descriptions: dict[str, str] = Field(default_factory=dict, max_length=32)
    motions: dict[str, MotionReference] = Field(default_factory=dict, max_length=32)
    facing: bool = False
    condition_goals: bool = False
    motion_anchor: Literal["session", "current"] = "session"
    execution_mode: Literal["buffered", "feedback"] = "buffered"

    @model_validator(mode="after")
    def validate_goals(self):
        if not (self.goals or self.motions or self.facing or self.condition_goals):
            raise ValueError("configure at least one articulated goal adapter")
        if len(self.goals) + len(self.motions) > 32:
            raise ValueError("at most 32 configured posture and motion entries combined")
        if not all(posture_capability(name) for name in self.goals):
            raise ValueError("only named posture goals are currently implemented")
        if set(self.descriptions) - set(self.goals) or any(
            not value or len(value) > 80 for value in self.descriptions.values()
        ):
            raise ValueError(
                "posture descriptions must name configured goals and fit 80 characters"
            )
        if set(self.goals) & set(self.motions):
            raise ValueError("a posture must have one static or learned transition definition")
        if not all(motion_capability(name) for name in self.motions):
            raise ValueError("unsupported motion capability")
        for name, motion in self.motions.items():
            if (name == "WAVE") != (motion.hand is not None):
                raise ValueError("only a WAVE motion must declare its demonstrated hand")
        for pose in self.goals.values():
            if (
                not pose.is_full_body
                or pose.left.controls != Controls()
                or pose.right.controls != Controls()
            ):
                raise ValueError("articulated task goals require eleven poses and neutral inputs")
            if vector(pose)[:, 2].min() < self.reference_floor:
                raise ValueError("task goal is below its declared floor")
        return self


def anchored_goal(template, current):
    """Bind a complete posture goal at current tracking XY and pelvis heading."""

    angle = heading(current.pelvis) - heading(template.pelvis)
    rotation = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
    pivot = rotate(rotation, template.pelvis.position)
    offset = (current.pelvis.position[0] - pivot[0], current.pelvis.position[1] - pivot[1], 0.0)
    points = vector(template)
    for row in points:
        row[:3] = tuple(x + y for x, y in zip(rotate(rotation, tuple(row[:3])), offset))
        row[3:] = qmul(rotation, tuple(row[3:]))
    return target_from_vector(points)


@dataclass(frozen=True)
class ActionExecution:
    generation: int
    accepted_at: float
    preparation_deadline: float
    started_at: float | None = None
    deadline: float | None = None
    error: str | None = None
    cancelled: bool = False
    completed_at: float | None = None

    def snapshot(self, now):
        error = self.error
        if self.cancelled:
            phase = "cancelled"
        elif error:
            phase = "failed"
        elif self.completed_at is not None:
            phase = "completed"
        elif self.started_at is None:
            phase = "preparing" if now < self.preparation_deadline else "failed"
            if phase == "failed":
                error = "articulated preparation timed out"
        else:
            phase = "running" if now < self.deadline else "expired"
        return {
            "generation": self.generation,
            "accepted_at": self.accepted_at,
            "preparation_deadline": self.preparation_deadline,
            "started_at": self.started_at,
            "deadline": self.deadline,
            "phase": phase,
            "error": error,
            "completed_at": self.completed_at,
        }


class ArticulatedIntentMotor:
    def __init__(self, path):
        self.settings = ArticulatedTasks.model_validate_json(outside_repo(path).read_text("utf-8"))
        self.controller = ArticulatedController(
            outside_repo(self.settings.actor), reference_floor=self.settings.reference_floor
        )
        if (
            self.settings.actor_sha256 != self.controller.manifest["sha256"]
            or self.settings.rig_sha256
            != hashlib.sha256(self.controller.rig.model_dump_json().encode()).hexdigest()
        ):
            self.controller.close()
            raise ValueError("articulated task catalogue does not match its actor and rig")
        self.policy_id = "articulated:" + self.settings.actor_sha256[:16]
        if self.settings.execution_mode == "buffered":
            from .buffered_actor import BufferedActor

            self.controller = BufferedActor(self.controller)
        self.key = self.pose_goal = None
        self.active = False
        self.learning_metadata = None
        self.error = None
        self.execution = None
        self.terminal_pose = None
        self.completed_evidence = None
        self.motions = {}
        self.playback = None
        self.locomotion_state = None
        self.motion_origin_xy = None
        self.facing = None
        self.world = WorldState()
        self.settling = None
        from .body_conditions import ConditionPreparation

        self.conditions = ConditionPreparation()
        try:
            for name, reference in self.settings.motions.items():
                self.install_motion(name, reference.load(self.settings.reference_floor), reference)
        except Exception:
            self.controller.close()
            raise

    def install_motion(self, name, model, reference):
        if not motion_capability(name) or (name == "WAVE") != (reference.hand is not None):
            raise ValueError("unsupported motion capability or hand")
        validate_motion(model, self.settings.reference_floor)
        if posture_capability(name):
            from .motion_prior import FiniteImitation

            if not isinstance(model, FiniteImitation):
                raise ValueError("a posture requires a finite transition and held endpoint")
            if name in self.settings.goals:
                raise ValueError("a static posture definition cannot be replaced by a motion")
        # Replacing a catalogue entry never changes the active playback object.
        self.motions[name] = (model, reference)

    def supports(self, intent):
        if intent.skill == "BODY_GOAL":
            return self.settings.condition_goals and intent.body_goal is not None
        if intent.skill == "LOOK_AT":
            return self.settings.facing
        if intent.skill in self.settings.goals:
            return True
        motion = self.motions.get(intent.skill)
        return bool(motion and (motion[1].hand is None or intent.hand == motion[1].hand))

    def task_identity(self, intent):
        fitting = {
            "fitting_contract": "per_tracker_tolerance_objective_v2",
            "feedback_contract": "per_tracker_relative_confirmation_v4",
            "heading_contract": "forward_or_lateral_projection_v2",
        }
        if self.settings.execution_mode == "buffered":
            fitting["feedback_contract"] = "observed_buffered_trajectory_v1"
        if intent.body_goal is not None:
            fitting["condition_goal_sha256"] = hashlib.sha256(
                intent.body_goal.model_dump_json().encode()
            ).hexdigest()
        if self.controller.rig.joint_limits:
            fitting.update(
                fitting_contract="joint_envelope_tracker_objective_v3",
                joint_constraint_contract=self.controller.manifest["joint_constraint_contract"],
                rig_sha256=self.settings.rig_sha256,
            )
        if intent.skill == "LOOK_AT" and self.settings.facing:
            return fitting | {
                "reference_control": "whole_body_image_heading_v1",
                "duration_s": intent.duration_s,
            }
        from .exploration import NAVIGATION_SKILLS

        motion = self.motions.get(
            "WALK_IN_PLACE" if intent.skill in NAVIGATION_SKILLS else intent.skill
        )
        if motion:
            return fitting | {
                "reference_control": "confirmed_feedback_phase_v1",
                "motion_anchor": self.settings.motion_anchor,
                "motion_evaluation": "observed_sequence_v1",
                "motion_sha256": motion[1].sha256,
                "playback_rate": motion[1].playback_rate,
                "duration_s": intent.duration_s,
            }
        goal = self.settings.goals.get(intent.skill)
        return fitting | (
            {
                "posture_sha256": hashlib.sha256(goal.model_dump_json().encode()).hexdigest(),
                "duration_s": intent.duration_s,
            }
            if goal
            else {}
        )

    def configure_registry(self, registry):
        for name in (
            set(
                (
                    "WAVE",
                    "LOOK_AT",
                    "REACH",
                    "RETURN_TO_REST",
                    "CROUCH",
                    "SIT",
                    "LIE",
                    "STAND",
                    "WALK_IN_PLACE",
                )
            )
            | set(self.motions)
            | set(self.settings.goals)
        ):
            item = registry.get(name)
            item.available = name in self.settings.goals or name in self.motions
            if name == "LOOK_AT" and self.settings.facing:
                item.available = True
                item.target_sources = ("vision",)
            item.status = (
                "explicit_articulated_candidate"
                if item.available
                else "unavailable_in_articulated_trial"
            )
            item.method = "imitation" if motion_capability(name) else "articulated_policy"
            item.policy = self.policy_id if item.available else None
            if name in self.settings.goals:
                item.description = self.settings.descriptions.get(name, "")
            if name in self.motions:
                model, reference = self.motions[name]
                item.supported_hands = (reference.hand,) if reference.hand else None
                item.description = reference.description or model.clip
                item.duration_s = reference.execution_timeout_s or min(
                    20.0, math.ceil(model.duration_s / reference.playback_rate + 3)
                )

    def timing(self, choice, now):
        key, accepted, _, _ = choice
        execution = self.execution
        if execution is None or execution.generation != key:
            execution = ActionExecution(key, accepted, accepted + 8.0)
        return execution.snapshot(now)

    def hold(self):
        self.learning_metadata = None
        self.conditions.reset()
        if self.active:
            self.controller.end_goal()
            if self.execution is not None:
                self.execution = replace(self.execution, cancelled=True)
        self.active = False

    def step(self, body, choice, now, dt):
        self.learning_metadata = None
        key, accepted, intent, _ = choice
        current = state_target(body)
        if key != self.key:
            self.hold()
            self.controller.new_goal()
            self.key, self.error = key, None
            self.completed_evidence = None
            self.execution = ActionExecution(key, accepted, accepted + 8.0)
            self.pose_goal = self.terminal_pose = None
            self.settling = None
            self.playback = None
            self.facing = None
        if not self.supports(intent):
            self.error = "intent has no learned task goal"
            self.execution = replace(self.execution, error=self.error)
        elif self.pose_goal is None and self.error is None:
            try:
                if intent.skill == "BODY_GOAL":
                    self.pose_goal = current  # Hold while the slow goal completion runs.
                elif intent.skill == "LOOK_AT":
                    self.facing = BodyFacing(current, intent.target)
                    self.pose_goal = current
                elif intent.skill in self.motions:
                    model, reference = self.motions[intent.skill]
                    if self.motion_origin_xy is None:
                        self.motion_origin_xy = current.pelvis.position[:2]
                    self.playback = MotionPlayback(
                        model,
                        current,
                        reference.playback_rate,
                        origin_xy=self.motion_origin_xy
                        if self.settings.motion_anchor == "session"
                        else None,
                    )
                    self.pose_goal = self.playback.target()
                else:
                    self.pose_goal = anchored_goal(self.settings.goals[intent.skill], current)
            except ValueError as exc:
                self.pose_goal, self.error = None, str(exc)
                self.execution = replace(self.execution, error=self.error)
        timing = self.timing(choice, now)
        if timing["phase"] in ("failed", "cancelled", "expired", "completed"):
            return self._finish(current, timing["error"])
        self.active = True
        if self.conditions.retiring():
            return current
        if self.facing:
            available = self.facing.observe(current, self.world, now)
            if self.facing.error:
                return self._finish(current, self.facing.error)
            if not available:
                self.controller.end_goal()
                return current
            self.pose_goal = self.facing.pose
        self.controller.observe(body, now)
        if self.controller.error:
            return self._finish(current, self.controller.error)
        starting = self.execution.started_at is None
        if starting and not self.controller.ready:
            return self.controller.hold_target or current
        if intent.skill == "BODY_GOAL":
            try:
                completed = self.conditions.poll(
                    self.controller, current, intent.body_goal, self.world, now
                )
                if completed is None:
                    return current
                self.pose_goal = completed
            except (ValueError, RuntimeError) as exc:
                self.conditions.reset()
                return self._finish(current, str(exc))
        finite_complete = (
            self.playback is not None
            and self.playback.completed
            and self.playback.evidence()["endpoint_required"]
        )
        if (
            not starting
            and self.controller.ready
            and (
                intent.skill in self.settings.goals
                or intent.skill == "BODY_GOAL"
                or finite_complete
                or self.facing
            )
        ):
            evidence = self.goal_evidence(body, now, key)
            if evidence["success"] is True:
                stamp = min(body.signal_for(part).timestamp for part in PARTS)
                since, last, count = self.settling or (stamp, stamp, 0)
                if stamp - last > 0.5:
                    since, count = stamp, 0
                if stamp > last or count == 0:
                    count += 1
                self.settling = (since, stamp, count)
                if count >= 3 and stamp - since >= 0.15:
                    self.completed_evidence = {
                        **evidence,
                        "observed_at": stamp,
                        "settled_since": since,
                        "settled_samples": count,
                    }
                    self.execution = replace(self.execution, completed_at=now)
                    return self._finish(current, None)
                # Already within the task envelope: confirm the held observed
                # pose instead of issuing further increments that may drift out.
                return current
            self.settling = None
        if self.playback and self.controller.ready:
            signals = [body.signal_for(part) for part in PARTS]
            if all(s.valid and s.connected and 0 <= now - s.timestamp < 0.5 for s in signals):
                locomotion = getattr(self, "locomotion_state", None)
                self.playback.observe(
                    current,
                    min(s.timestamp for s in signals),
                    dt,
                    speed_scale=locomotion.gait_speed_scale if locomotion is not None else 1.0,
                )
                self.pose_goal = self.playback.target()
        previous = self.controller.previous.copy()
        if self.settings.execution_mode == "buffered":
            import copy

            reference = copy.copy(self.playback) if self.playback else None
            speed = self.locomotion_state.gait_speed_scale if self.locomotion_state else 1.0
            self.controller.reference = (
                (
                    lambda at: reference.target(
                        reference.phase
                        + max(0, at - now) * reference.rate * speed / reference.model.duration_s
                    )
                )
                if reference
                else None
            )
        deadline = now + intent.duration_s if starting else self.execution.deadline
        action_dt = min(dt, deadline - now)
        action, obs, rates = self.controller.step(
            body, self.pose_goal, action_dt, remaining_s=deadline - now
        )
        if self.controller.error:
            return self._finish(current, self.controller.error)
        if starting and getattr(self.controller, "started", False):
            self.execution = replace(self.execution, started_at=now, deadline=deadline)
        if obs is not None:
            if starting:
                self.execution = replace(self.execution, started_at=now, deadline=deadline)
            self.learning_metadata = {
                "motor_contract": CONTRACT,
                "rates": rates.tolist(),
                "previous_rates": previous.tolist(),
                "integration_dt": action_dt,
                "policy_observation": obs.tolist(),
                "pose_goal": self.pose_goal.model_dump(mode="json"),
                "policy_id": self.policy_id,
                "latent_action": self.controller.last_latent.tolist(),
                "deadline": deadline,
                "goal_owner": "autonomous",
                "intent_generation": key,
                "execution": self.execution.snapshot(now),
                "motion_reference": self.playback.evidence() if self.playback else None,
                "facing_reference": self.facing.evidence(self.world, now) if self.facing else None,
                "condition_goal": self.conditions.report,
                **self.controller.last_metadata,
            }
        return action

    def _finish(self, current, error):
        if error or self.conditions.pending:
            self.conditions.reset()
        if error:
            self.error = error
            self.execution = replace(self.execution, error=error)
        if self.active:
            self.controller.end_goal()
        self.active = False
        if self.terminal_pose is None:
            self.terminal_pose = current
        return self.terminal_pose

    def status(self):
        return {
            "policy_id": self.policy_id,
            "task_error": self.error,
            "available_tasks": sorted(
                set(self.settings.goals)
                | set(self.motions)
                | ({"LOOK_AT"} if self.settings.facing else set())
                | ({"BODY_GOAL"} if self.settings.condition_goals else set())
            ),
            "motion_reference": self.playback.evidence() if self.playback else None,
            "condition_goal": self.conditions.report,
            "goal_evidence": self.completed_evidence,
            **self.controller.status(),
        }

    def goal_evidence(self, body, now, generation):
        if self.key == generation and getattr(self, "completed_evidence", None) is not None:
            return dict(self.completed_evidence)
        evidence = {
            "success": None,
            "scope": "unavailable",
            "goal_owner": "articulated",
            "policy_id": self.policy_id,
            "intent_generation": generation,
        }
        if self.pose_goal is None or self.key != generation or body is None:
            return evidence
        signals = [body.signal_for(part) for part in PARTS]
        if not all(
            s.valid and s.pose is not None and 0 <= now - s.timestamp < 0.5 for s in signals
        ):
            return evidence
        sources = {s.source for s in signals}
        if sources not in ({"simulated"}, {"openvr_raw"}):
            return evidence
        error = pose_error(state_target(body), self.pose_goal)
        position = float(np.linalg.norm(error[:, :3], axis=1).max())
        angle = float(np.linalg.norm(error[:, 3:], axis=1).max())
        playback = getattr(self, "playback", None)
        motion = playback.evidence() if playback else None
        endpoint = position <= 0.12 and angle <= 0.35
        completed = (
            endpoint
            if not playback
            else playback.completed and (endpoint if motion["endpoint_required"] else True)
        )
        scope = "simulated_tracker" if sources == {"simulated"} else "device_tracker"
        result = {
            **evidence,
            "success": completed
            and (getattr(self, "error", None) or self.controller.error) is None,
            "scope": scope + ("_motion" if playback else "_goal"),
            "endpoint_within_tolerance": endpoint,
            "maximum_position_error_m": position,
            "maximum_rotation_error_rad": angle,
            "position_tolerance_m": 0.12,
            "rotation_tolerance_rad": 0.35,
            "actor_error": getattr(self, "error", None) or self.controller.error,
            "avatar_verified": False,
            "motion_reference": motion,
        }
        facing = getattr(self, "facing", None)
        conditions = getattr(self, "conditions", None)
        if conditions and conditions.resolved:
            measured = conditions.resolved.measure(state_target(body))
            try:
                conditions.resolved.validate_targets(self.world, now)
            except ValueError:
                measured["success"] = False
            result.update(
                condition_evidence=measured, success=result["success"] and measured["success"]
            )
        if facing:
            visual = facing.evidence(self.world, now)
            result.update(visual, success=visual["success"] if result["success"] else False)
        return result

    def close(self):
        self.conditions.reset()
        self.controller.close()
