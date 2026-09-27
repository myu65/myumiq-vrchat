"""Finite supervised body console; no driver setup or inferred avatar success."""

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .backends.osc import LiveConfig
from .backends.readback import OpenVRReadback, require_console
from .backends.supervisor import OutputSupervisor
from .body import (
    BodyGoal,
    BodyTarget,
    BodyTask,
    Controls,
    Frozen,
    Number,
    Pose,
    SignedUnit,
    WorldState,
    simulated_body,
)
from .cli import outside_repo
from .command_inbox import CommandInbox
from .loop_diagnostics import LoopDiagnostics, LoopStallTrace
from .postures import posture_target
from .replay import Observation, ReplayBuffer, WholeBodyTransition
from .replay_checkpoint import ReplayCheckpoint
from .status_writer import StatusWriter
from .whole_body import PARTS, PeriodicImitation, WholeBodyPolicy, bounded_step, state_target


class Command(Frozen):
    session: str
    issued: Number
    kind: Literal[
        "posture",
        "menu",
        "trigger",
        "fist",
        "open_hand",
        "aim",
        "view",
        "reference",
        "play",
        "stop",
        "autonomous",
        "manual",
        "utterance",
        "associate_speaker",
        "drive",
        "explore",
        "tracker_rates",
        "learned_pose",
        "body_goal",
    ]
    body_goal: BodyGoal | None = None
    pose_goal: BodyTarget | None = None
    goal_duration_s: Number | None = Field(default=None, ge=1, le=20)
    rates: tuple[SignedUnit, ...] | None = Field(default=None, min_length=66, max_length=66)
    direction: Literal["forward", "backward", "left", "right"] | None = None
    text: str | None = Field(default=None, min_length=1, max_length=2000)
    speaker_id: str | None = Field(default=None, min_length=1, max_length=80)
    target: str | None = Field(default=None, min_length=1, max_length=80)
    name: Literal["standing", "crouching", "sitting_floor", "lying"] | None = None
    hand: Literal["left", "right", "both"] | None = None
    pose: Pose | None = None
    duration_s: Number = Field(default=0.15, ge=0.05, le=0.5)

    @model_validator(mode="after")
    def arguments(self):
        if (self.kind == "body_goal") != (self.body_goal is not None):
            raise ValueError("body_goal requires conditions only")
        if self.body_goal is not None and (
            not self.body_goal.conditions or not 1 <= self.body_goal.duration_s <= 20
        ):
            raise ValueError(
                "body_goal requires explicit conditions and a duration of 1 to 20 seconds"
            )
        if (self.kind == "learned_pose") != (self.pose_goal is not None) or (
            self.kind == "learned_pose"
        ) != (self.goal_duration_s is not None):
            raise ValueError("learned_pose requires a full pose goal and finite duration")
        if self.pose_goal is not None and not self.pose_goal.is_full_body:
            raise ValueError("learned_pose requires all eleven poses")
        if self.pose_goal is not None and (
            self.pose_goal.left.controls != Controls()
            or self.pose_goal.right.controls != Controls()
        ):
            raise ValueError("pose goals cannot carry controller locomotion or buttons")
        if (self.kind == "tracker_rates") != (self.rates is not None):
            raise ValueError("tracker_rates requires exactly 66 normalized rates")
        if (self.kind == "drive") != (self.direction is not None):
            raise ValueError("drive requires exactly one direction")
        if (self.kind == "utterance") != (self.text is not None):
            raise ValueError("utterance requires text only")
        if self.kind == "associate_speaker" and self.speaker_id is None:
            raise ValueError("explicit speaker association requires an identifier")
        if self.kind not in ("utterance", "associate_speaker") and (self.speaker_id or self.target):
            raise ValueError("unexpected identity fields")
        if (self.kind == "posture") != (self.name is not None):
            raise ValueError("posture requires exactly one posture name")
        if (self.kind in ("aim", "view")) != (self.pose is not None):
            raise ValueError("aim/view requires exactly one pose")
        if self.kind in ("menu", "trigger", "fist", "open_hand", "aim"):
            if self.hand is None or (
                self.hand == "both" and self.kind not in ("trigger", "fist", "open_hand")
            ):
                raise ValueError("invalid hand for this command")
        elif self.hand is not None:
            raise ValueError("unexpected hand")
        return self

    def fresh(self, session: str, now: float) -> bool:
        return self.session == session and 0 <= now - self.issued <= 2


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    try:
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.002)
    finally:
        temporary.unlink(missing_ok=True)


def submit_body_goal(owner, goal, now):
    """Operator goal entry; admission is separate from achieved motion."""
    if not (
        owner
        and owner.enabled
        and owner.learned_motor
        and owner.learned_motor.settings.condition_goals
    ):
        return {"accepted": False, "reason": "active condition-enabled articulated motor required"}
    from .autonomous_body import Intent

    intent = Intent(skill="BODY_GOAL", body_goal=goal, duration_s=goal.duration_s)
    owner._choose(now, intent, "operator_body_goal")
    return {"accepted": True, "source": "operator_condition_goal", "completed": False}


def pulse_controls(kind: str) -> Controls:
    if kind in ("fist", "open_hand"):
        return Controls(curls=(1.0 if kind == "fist" else 0.0,) * 5)
    if kind == "menu":
        return Controls(buttons=tuple(i == 3 for i in range(18)))
    if kind == "trigger":
        return Controls(
            triggers=(1.0,) + (0.0,) * 8,
            trigger_clicks=(True,) + (False,) * 8,
            trigger_touches=(True,) + (False,) * 8,
            curls=(0.0, 1.0, 0.0, 0.0, 0.0),
        )
    raise ValueError("unsupported pulse")


def run(args):
    if not 1 <= args.duration <= 2100:
        raise ValueError("session duration must be 1..2100 seconds")
    config = None
    if args.live_config:
        require_console()
        config = LiveConfig.model_validate_json(
            outside_repo(args.live_config).read_text("utf-8-sig")
        )
        if not config.trackers or not args.hmd_serial:
            raise ValueError("eleven owned devices and HMD serial required")
    elif args.hmd_serial:
        raise ValueError("HMD identity requires live config")
    model = PeriodicImitation.model_validate_json(args.policy.read_text()) if args.policy else None
    tracker_actor = None
    articulated = None
    if getattr(args, "articulated_policy", None):
        if getattr(args, "tracker_policy", None) or getattr(args, "autonomous_config", None):
            raise ValueError(
                "articulated candidate trials require exclusive supervised actor ownership"
            )
        from .articulated_controller import ArticulatedController

        articulated = ArticulatedController(
            outside_repo(args.articulated_policy),
            reference_floor=getattr(args, "articulated_floor", None),
        )
        tracker_actor = articulated
    if getattr(args, "tracker_policy", None):
        from .tracker_policy import TrackerActor

        tracker_actor = TrackerActor(outside_repo(args.tracker_policy))
    reference = (
        BodyTarget.model_validate_json(args.reference_pose.read_text())
        if args.reference_pose
        else None
    )
    if reference is not None and not reference.is_full_body:
        raise ValueError("reference pose requires all eleven points")
    root = outside_repo(args.session)
    root.mkdir(parents=True, exist_ok=False)
    (root / "commands").mkdir()
    (root / "receipts").mkdir()
    token = uuid.uuid4().hex
    atomic_json(
        root / "session.json",
        {
            "session": token,
            "pid": os.getpid(),
            "live": config is not None,
            "duration_s": args.duration,
        },
    )
    rest = config.safe_target if config else posture_target("standing")
    current = desired = action = rest
    sensor = OpenVRReadback(config, args.hmd_serial) if config else None
    owner = OutputSupervisor(root / "output-events.jsonl", config)
    replay = ReplayBuffer(20000)
    consumed = set()
    policy = None
    pulse = None
    pulse_until = 0.0
    rate_lease = rate_pending = None
    learned_goal = None
    goal = BodyGoal(
        tasks=(BodyTask(id="standing", kind="posture", effectors=PARTS, posture="standing"),),
        duration_s=4.0,
    )
    started = previous_tick = time.perf_counter()
    last_record = last_status = 0.0
    error = None
    cleanup = []
    autonomous = None
    diagnostics = LoopDiagnostics()
    failure = None
    body = None
    status_writer = StatusWriter(root / "status.json")
    replay_checkpoint = ReplayCheckpoint(root / "experience.jsonl")
    command_inbox = CommandInbox(root, atomic_json)
    stall_trace = (
        LoopStallTrace(root / "stall-trace.log") if getattr(args, "trace_stalls", False) else None
    )
    try:
        if getattr(args, "autonomous_config", None):
            from .autonomous_body import AutonomousBody, AutonomousConfig

            autonomous = AutonomousBody(
                AutonomousConfig.model_validate_json(
                    outside_repo(args.autonomous_config).read_text("utf-8-sig")
                ),
                root,
                rest,
                model,
            )
            if autonomous.learned_motor:
                if tracker_actor is not None:
                    raise ValueError("only one learned body actor may own the console")
                articulated = tracker_actor = autonomous.learned_motor.controller
            autonomous.start()
        if sensor:
            sensor.start()
        owner.start()
        command_inbox.start()
        diagnostics.enter("loop_wait")
        while time.perf_counter() - started < args.duration:
            if stall_trace:
                stall_trace.tick()
            now = time.perf_counter()
            if command_inbox.stop_requested.is_set():
                break
            diagnostics.enter("readback")
            body = sensor.observe() if sensor else simulated_body(current, now)
            # Sensors stamp their sample during observe(). Motor freshness checks
            # must use a time after capture, never classify that sample as future.
            now = time.perf_counter()
            dt = min(0.1, max(0.0001, now - previous_tick))
            previous_tick = now
            diagnostics.enter("feedback_validation")
            if not owner.alive:
                raise RuntimeError("output supervisor stopped; see output-events.jsonl")
            valid = all(body.signal_for(p).valid for p in PARTS)
            if not valid:
                if now - started > 10:
                    raise RuntimeError("required device feedback unavailable")
                owner.publish(rest)
                time.sleep(0.01)
                continue
            current = state_target(body)
            if autonomous and autonomous.enabled:
                autonomous.observe_feedback(body, action)
            diagnostics.enter("commands")
            # One immutable file per command avoids Windows replace/read races.
            for path, encoded in command_inbox.poll():
                command = Command.model_validate_json(encoded)
                consumed.add(path.name)
                # A command can arrive after this tick began. Check freshness
                # at receipt, not against the earlier motor observation time.
                if not command.fresh(token, time.perf_counter()):
                    command_inbox.write(
                        root / "receipts" / path.name,
                        {"accepted": False, "reason": "expired or wrong session"},
                    )
                    continue
                if command.kind == "stop":
                    command_inbox.stop_requested.set()
                    command_inbox.write(root / "receipts" / path.name, {"accepted": True})
                    break
                if command.kind == "body_goal":
                    receipt = submit_body_goal(autonomous, command.body_goal, now)
                    command_inbox.write(root / "receipts" / path.name, receipt)
                    continue
                if command.kind in ("utterance", "associate_speaker", "explore"):
                    if not autonomous or not autonomous.config.purpose or not autonomous.enabled:
                        command_inbox.write(
                            root / "receipts" / path.name,
                            {"accepted": False, "reason": "active purpose mode required"},
                        )
                        continue
                    if command.target and not any(
                        o.name == command.target for o in autonomous.world.objects
                    ):
                        command_inbox.write(
                            root / "receipts" / path.name,
                            {"accepted": False, "reason": "target not observed"},
                        )
                        continue
                    if command.kind == "associate_speaker":
                        autonomous.speaker_binding = {
                            "id": command.speaker_id,
                            "target": command.target,
                            "expires": now + 300,
                        }
                    else:
                        from .events import RuntimeEvent

                        autonomous.external_events.publish(
                            RuntimeEvent(
                                "explore" if command.kind == "explore" else "utterance",
                                now,
                                target=command.target,
                                speaker_id=command.speaker_id,
                                text=command.text,
                                source="operator_text",
                            )
                        )
                    command_inbox.write(
                        root / "receipts" / path.name,
                        {"accepted": True, "source": "operator_association_or_text"},
                    )
                    continue
                if command.kind == "learned_pose" and tracker_actor is None:
                    command_inbox.write(
                        root / "receipts" / path.name,
                        {
                            "accepted": False,
                            "reason": "start with --tracker-policy or --articulated-policy",
                        },
                    )
                    continue
                if autonomous and command.kind != "autonomous":
                    autonomous.enable(False)
                if command.kind != "learned_pose":
                    learned_goal = None
                    if articulated is not None:
                        articulated.reset()
                    if tracker_actor is not None:
                        tracker_actor.previous.fill(0.0)
                rate_lease = None
                pulse = None
                if command.kind == "learned_pose":
                    if articulated is not None:
                        articulated.new_goal()
                    learned_goal = (command.pose_goal, now + command.goal_duration_s)
                    policy, desired = None, current
                    goal = BodyGoal(
                        tasks=(BodyTask(id="learned-pose", kind="posture", effectors=PARTS),),
                        duration_s=command.goal_duration_s,
                    )
                elif command.kind == "tracker_rates":
                    rate_lease = (command, now + command.duration_s)
                    policy, desired, pulse = None, current, None
                    goal = BodyGoal(
                        tasks=(BodyTask(id="tracker-rates", kind="posture", effectors=PARTS),),
                        duration_s=command.duration_s,
                    )
                elif command.kind == "autonomous":
                    if autonomous is None:
                        command_inbox.write(
                            root / "receipts" / path.name,
                            {"accepted": False, "reason": "autonomous config required"},
                        )
                        continue
                    autonomous.enable(True)
                    policy, desired = None, current
                elif command.kind == "manual":
                    policy, desired = None, current
                elif command.kind == "posture":
                    desired = posture_target(command.name)
                    policy = None
                    goal = BodyGoal(
                        tasks=(
                            BodyTask(
                                id=command.name,
                                kind="posture",
                                effectors=PARTS,
                                posture=command.name,
                            ),
                        ),
                        duration_s=4.0,
                    )
                elif command.kind == "aim":
                    desired = desired.model_copy(
                        update={
                            command.hand: getattr(desired, command.hand).model_copy(
                                update={"pose": command.pose}
                            )
                        }
                    )
                    policy = None
                elif command.kind == "view":
                    desired = desired.model_copy(update={"head": command.pose})
                    policy = None
                elif command.kind == "reference":
                    if reference is None:
                        raise ValueError("start with --reference-pose first")
                    desired, policy = reference, None
                elif command.kind == "play":
                    if model is None:
                        raise ValueError("start with --policy before requesting play")
                    policy = WholeBodyPolicy(model)
                    goal = BodyGoal(
                        tasks=(
                            BodyTask(
                                id="imitation",
                                kind="locomotion",
                                effectors=PARTS,
                                target=model.clip,
                            ),
                        ),
                        duration_s=20.0,
                    )
                else:
                    pulse, pulse_until = command, now + command.duration_s
                command_inbox.write(root / "last-command.json", command.model_dump(mode="json"))
                command_inbox.write(root / "receipts" / path.name, {"accepted": True})
            if articulated is not None and (
                articulated.feedback_pending or learned_goal and now < learned_goal[1]
            ):
                articulated.observe(body, now)
            if rate_pending is not None:
                old_observation, old_action, old_goal, metadata = rate_pending
                learning, reward = None, None
                defer_feedback = False
                if "policy_observation" in metadata:
                    import numpy as np

                    from .replay import TrackerLearningStep
                    from .tracker_policy import OBSERVATION_CONTRACT, observation, reward_components

                    pose_goal = BodyTarget.model_validate_json(json.dumps(metadata["pose_goal"]))
                    rates = np.asarray(metadata["rates"])
                    components = reward_components(
                        state_target(old_observation.body),
                        current,
                        pose_goal,
                        rates,
                        np.asarray(metadata["previous_rates"]),
                    )
                    following = (
                        articulated.replay_observation(
                            body, pose_goal, rates, metadata["integration_dt"], metadata
                        )
                        if "next_joint_state" in metadata
                        else observation(current, pose_goal, rates, metadata["integration_dt"])
                    )
                    if metadata.get("goal_owner") == "autonomous":
                        task_changed = (
                            autonomous is None
                            or not autonomous.enabled
                            or autonomous.choice[0] != metadata["intent_generation"]
                        )
                    else:
                        task_changed = (
                            learned_goal is None or learned_goal[1] != metadata["deadline"]
                        )
                    if following is not None:
                        if metadata.get("reference_floor") is not None:
                            heights = np.array(
                                [current.left_foot.position[2], current.right_foot.position[2]]
                            )
                            components["reference_floor_intrusion"] = -metadata.get(
                                "reference_floor_weight", 20.0
                            ) * float(
                                np.mean(
                                    np.maximum(0, metadata["reference_floor"] + 0.02 - heights)
                                    ** metadata.get("reference_floor_power", 2)
                                )
                            )
                        reward = sum(components.values())
                        learning = TrackerLearningStep(
                            rates=tuple(metadata["rates"]),
                            dt=metadata["integration_dt"],
                            policy_observation=tuple(metadata["policy_observation"]),
                            next_policy_observation=tuple(following),
                            observation_contract=metadata.get(
                                "observation_contract", OBSERVATION_CONTRACT
                            ),
                            reward_components=components,
                            reward_scope="tracker_geometry",
                            evidence_ids=(
                                f"{token}:{'readback' if config else 'simulated'}:{now:.9f}",
                            ),
                            terminated=False,
                            truncated=(
                                now >= metadata["deadline"]
                                or task_changed
                                or command_inbox.stop_requested.is_set()
                            ),
                        )
                    else:
                        defer_feedback = (
                            now < min(metadata.get("feedback_deadline", now), metadata["deadline"])
                            and articulated is not None
                            and articulated.feedback_pending
                            and not task_changed
                            and not command_inbox.stop_requested.is_set()
                        )
                        if not defer_feedback:
                            metadata["learning_unavailable"] = (
                                "issued target or predicted joints not confirmed by following feedback"
                            )
                if not defer_feedback:
                    replay.add(
                        WholeBodyTransition(
                            observation=old_observation,
                            next_observation=Observation(
                                timestamp=now,
                                body=body,
                                world=autonomous.world if autonomous else WorldState(),
                            ),
                            body_goal=old_goal,
                            action=old_action,
                            policy_id=metadata.get("policy_id", "supervised-tracker-rates"),
                            learning=learning,
                            reward=reward,
                            intent_metadata=metadata,
                            environment="vrchat" if config else "mock",
                            outcome="device_feedback" if config else "simulated",
                        ).model_dump_json()
                    )
                    rate_pending = None
            if command_inbox.stop_requested.is_set():
                break
            diagnostics.enter("motor")
            rate_metadata = None
            if learned_goal and now < learned_goal[1]:
                from .tracker_action import CONTRACT

                pose_goal, deadline = learned_goal
                action_dt = min(dt, deadline - now)
                previous_rates = tracker_actor.previous.copy()
                action, policy_observation, rates = tracker_actor.step(
                    body,
                    pose_goal,
                    action_dt,
                    **({"remaining_s": deadline - now} if articulated is not None else {}),
                )
                desired = action
                if policy_observation is not None:
                    rate_metadata = {
                        "motor_contract": CONTRACT,
                        "rates": rates.tolist(),
                        "integration_dt": action_dt,
                        "policy_observation": policy_observation.tolist(),
                        "pose_goal": pose_goal.model_dump(mode="json"),
                        "previous_rates": previous_rates.tolist(),
                        "policy_id": ("articulated:" if articulated else "tracker-sac:")
                        + tracker_actor.manifest["sha256"][:16],
                        "latent_action": getattr(tracker_actor, "last_latent", None).tolist()
                        if getattr(tracker_actor, "last_latent", None) is not None
                        else None,
                        "decoder_id": tracker_actor.manifest.get("decoder_id"),
                        "deadline": deadline,
                        "reward_scope": "tracker_geometry_not_avatar",
                        **getattr(tracker_actor, "last_metadata", {}),
                    }
            elif learned_goal:
                learned_goal = None
                action = desired = current
                tracker_actor.previous.fill(0.0)
                if articulated is not None:
                    articulated.end_goal()
            elif rate_lease and now < rate_lease[1]:
                from .tracker_action import CONTRACT, integrate_tracker_action

                command, until = rate_lease
                action_dt = min(dt, until - now)
                rows = tuple(command.rates[i : i + 6] for i in range(0, 66, 6))
                action = integrate_tracker_action(current, rows, action_dt)
                desired = action
                rate_metadata = {
                    "motor_contract": CONTRACT,
                    "rates": command.rates,
                    "integration_dt": action_dt,
                    "issued": command.issued,
                }
            elif rate_lease:
                rate_lease = None
                action = desired = current
            elif autonomous and autonomous.enabled:
                action = autonomous.step(body, now, dt)
                goal = autonomous.goal
                if autonomous.learning_metadata is not None:
                    rate_metadata = {**autonomous.intent_metadata, **autonomous.learning_metadata}
            elif policy and policy.elapsed + dt <= goal.duration_s:
                action = policy.step(body, goal, dt)
            else:
                if policy:
                    policy, desired = None, current
                action = bounded_step(current, desired, dt)
            if pulse and now < pulse_until:
                if pulse.kind == "drive":
                    from .exploration import drive_target

                    action = drive_target(action, pulse.direction)
                for hand in ("left", "right"):
                    if pulse.hand in (hand, "both"):
                        action = action.model_copy(
                            update={
                                hand: getattr(action, hand).model_copy(
                                    update={"controls": pulse_controls(pulse.kind)}
                                )
                            }
                        )
            diagnostics.enter("publish")
            owner.publish(action)
            current = action
            diagnostics.enter("replay")
            if rate_metadata is not None:
                if rate_pending is not None:
                    raise RuntimeError("new learned action before previous feedback was resolved")
                rate_pending = (
                    Observation(
                        timestamp=now,
                        body=body,
                        world=autonomous.world if autonomous else WorldState(),
                    ),
                    action,
                    goal,
                    rate_metadata,
                )
            elif now - last_record >= 0.1:
                after = sensor.observe() if sensor else simulated_body(action, time.perf_counter())
                replay.add(
                    WholeBodyTransition(
                        observation=Observation(
                            timestamp=now,
                            body=body,
                            world=autonomous.world if autonomous else WorldState(),
                        ),
                        next_observation=Observation(
                            timestamp=time.perf_counter(),
                            body=after,
                            world=autonomous.world if autonomous else WorldState(),
                        ),
                        body_goal=goal,
                        action=action,
                        policy_id=(
                            "autonomous"
                            if autonomous and autonomous.enabled
                            else "supervised-console"
                            if policy is None
                            else "imitation:" + model.clip
                        ),
                        intent_metadata=(
                            autonomous.intent_metadata
                            if autonomous and autonomous.enabled
                            else {"articulated_actor": articulated.status()}
                            if articulated
                            else {}
                        ),
                        environment="vrchat" if config else "mock",
                        outcome="device_feedback" if config else "simulated",
                    ).model_dump_json()
                )
                last_record = now
            replay_checkpoint.poll(replay, now)
            diagnostics.enter("status")
            if now - last_status >= 0.5:
                status_writer.submit(
                    {
                        "running": True,
                        "heartbeat": now,
                        "devices_valid": valid,
                        "body": body.model_dump(mode="json"),
                        "avatar_verified": False,
                        "commands_processed": len(consumed),
                        "autonomous": autonomous.status() if autonomous else None,
                        "articulated_actor": articulated.status() if articulated else None,
                        "loop_diagnostics": diagnostics.snapshot(),
                    }
                )
                last_status = now
            diagnostics.enter("loop_wait")
            time.sleep(0.01)
    except Exception as exc:
        error = repr(exc)
        failure = {
            "timestamp": time.perf_counter(),
            "wall_time": time.time(),
            "output_alive": owner.alive,
            "output_failed": owner.failed,
            "loop_diagnostics": diagnostics.snapshot(),
            "device_feedback": {
                part: body.signal_for(part).model_dump(mode="json") for part in PARTS
            }
            if body
            else None,
        }
        raise
    finally:
        if stall_trace:
            stall_trace.close()
        actor_report = articulated.status() if articulated else None
        for name, component in (
            ("output", owner),
            ("readback", sensor),
            ("autonomy", autonomous),
            ("actor", articulated),
            ("status_writer", status_writer),
            ("command_inbox", command_inbox),
            ("replay_checkpoint", replay_checkpoint),
        ):
            if component:
                atomic_json(
                    root / "cleanup-progress.json",
                    {"phase": name, "finished": False, "timestamp": time.perf_counter()},
                )
                try:
                    component.close()
                except Exception as exc:
                    cleanup.append(repr(exc))
        atomic_json(
            root / "cleanup-progress.json",
            {"phase": "save_replay", "finished": False, "timestamp": time.perf_counter()},
        )
        replay.save_state(root / "experience.jsonl")
        atomic_json(
            root / "result.json",
            {
                "error": error,
                "cleanup_errors": cleanup,
                "articulated_actor": actor_report,
                "failure": failure,
                "loop_diagnostics": diagnostics.snapshot(),
                "transitions": len(replay),
                "elapsed_s": time.perf_counter() - started,
                "avatar_verified": False,
                "autonomous": autonomous.status() if autonomous else None,
            },
        )
        atomic_json(root / "status.json", {"running": False, "heartbeat": time.perf_counter()})
        atomic_json(
            root / "cleanup-progress.json",
            {
                "phase": "complete",
                "finished": True,
                "errors": cleanup,
                "timestamp": time.perf_counter(),
            },
        )
        if cleanup:
            raise RuntimeError(f"cleanup failed: {cleanup}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    start = sub.add_parser("run")
    start.add_argument("--session", type=Path, required=True)
    start.add_argument("--live-config", type=Path)
    start.add_argument("--hmd-serial")
    start.add_argument("--policy", type=Path)
    start.add_argument(
        "--tracker-policy", type=Path, help="explicit candidate actor for learned_pose commands"
    )
    start.add_argument(
        "--articulated-policy",
        type=Path,
        help="explicit articulated candidate with background pose fitting",
    )
    start.add_argument(
        "--articulated-floor", type=float, help="declared floor Z in the calibrated tracking frame"
    )
    start.add_argument("--reference-pose", type=Path)
    start.add_argument("--duration", type=float, default=300)
    start.add_argument(
        "--trace-stalls",
        action="store_true",
        help="diagnose frame stalls without changing output deadlines",
    )
    start.add_argument("--autonomous-config", type=Path)
    for name in ("send", "status", "stop"):
        child = sub.add_parser(name)
        child.add_argument("--session", type=Path, required=True)
        if name == "send":
            child.add_argument("command", help="JSON command without session/issued")
    args = parser.parse_args()
    if args.operation == "run":
        return run(args)
    root = outside_repo(args.session)
    status = json.loads((root / "status.json").read_text())
    if args.operation == "status":
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    if args.operation == "stop":
        # A stale heartbeat must never prevent an idempotent stop request.
        (root / "stop.txt").touch()
        return
    if not status["running"] or not 0 <= time.perf_counter() - status["heartbeat"] < 2:
        raise RuntimeError("session is not running or its heartbeat is stale")
    identity = json.loads((root / "session.json").read_text())
    payload = json.loads(args.command)
    payload.update(session=identity["session"], issued=time.perf_counter())
    command = Command.model_validate_json(json.dumps(payload))
    name = f"{time.time_ns()}-{uuid.uuid4().hex}.json"
    atomic_json(root / "commands" / name, command.model_dump(mode="json"))
    deadline = time.perf_counter() + 2.5
    while time.perf_counter() < deadline:
        try:
            receipt = json.loads((root / "receipts" / name).read_text())
        except (FileNotFoundError, PermissionError):
            time.sleep(0.02)
            continue
        if not receipt["accepted"]:
            raise RuntimeError(receipt["reason"])
        print("Accepted: " + command.kind)
        return
    raise TimeoutError("command acknowledgement unavailable; inspect status before retrying input")


if __name__ == "__main__":
    main()
