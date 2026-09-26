"""PAMIQ interaction and persistence integration for a bounded avatar session."""

import json
import time
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path

from pamiq_core import Agent, Environment, FixedIntervalInteraction, LaunchConfig, launch

from .autonomy import AutonomousPlanner
from .backends.osc import LiveConfig
from .backends.readback import OpenVRReadback
from .backends.supervisor import OutputSupervisor
from .body import EXTRA_PARTS, ActuationTarget, WorldState, rest_target, simulated_body
from .calibration import ResidualPolicy
from .cognition import WAIT, Decision, Goal
from .conversation import LiveVoiceLoop
from .events import EventInbox
from .gaze import VisualGazePolicy
from .motor import MotionCommand, ProceduralMotor
from .reach_policy import LearnedReachPolicy
from .replay import Observation, ReplayBuffer, Transition, device_progress_reward
from .vision import LiveVisionLoop


class BodyEnvironment(Environment[Observation, ActuationTarget]):
    def __init__(
        self,
        supervisor: OutputSupervisor,
        world: WorldState,
        sensor: OpenVRReadback | None = None,
        rest: ActuationTarget | None = None,
        vision: LiveVisionLoop | None = None,
    ):
        self.supervisor, self.world, self.sensor, self.vision = supervisor, world, sensor, vision
        self.target = rest or rest_target()
        self.error = None
        self.started = 0.0
        self.frames = 0

    def setup(self):
        try:
            if self.sensor:
                self.sensor.start()
            self.supervisor.start()
            self.started = time.perf_counter()
        except Exception as exc:
            self.error = str(exc)
            raise

    def observe(self):
        try:
            now = time.perf_counter()
            if not self.supervisor.alive:
                raise RuntimeError("output supervisor stopped")
            body = self.sensor.observe() if self.sensor else simulated_body(self.target, now)
            if self.sensor and (
                not body.head.valid
                or (
                    now - self.started > 2
                    and not (
                        body.left.valid
                        and body.right.valid
                        and all(
                            getattr(body, p).valid
                            for p in EXTRA_PARTS
                            if getattr(self.target, p) is not None
                        )
                    )
                )
            ):
                raise RuntimeError("required device tracking unavailable")
            world = self.vision.world() if self.vision else self.world
            return Observation(timestamp=now, body=body, world=world)
        except Exception as exc:
            self.error = str(exc)
            raise

    def affect(self, action):
        try:
            self.supervisor.publish(action)
            # Only the explicit mock sensor treats this as simulated next state.
            self.target = action
            self.frames += 1
        except Exception as exc:
            self.error = str(exc)
            raise

    def on_paused(self):
        self.supervisor.close()

    def on_resumed(self):
        # Resume is a new process/lease, never implicit replay of an old action.
        raise RuntimeError("start a new session after pause; automatic live re-arm is unsupported")

    def teardown(self):
        try:
            self.supervisor.close()
        finally:
            if self.sensor:
                self.sensor.close()


class BodyAgent(Agent[Observation, ActuationTarget]):
    def __init__(
        self,
        decision: Decision | Future[Decision],
        hz: float,
        rest: ActuationTarget | None = None,
        planner: AutonomousPlanner | None = None,
        residual: ResidualPolicy | None = None,
        events: EventInbox | None = None,
        voice: LiveVoiceLoop | None = None,
        gaze_policy: VisualGazePolicy | None = None,
        reach_policy: LearnedReachPolicy | None = None,
    ):
        super().__init__()
        self.pending = decision if isinstance(decision, Future) else None
        self.planner = planner
        self.events = events
        self.voice = voice
        self.decision = Decision(goal=WAIT, source="fixed") if self.pending else decision
        self.policy = ProceduralMotor(rest, residual, gaze_policy, reach_policy)
        self.hz = hz
        self.goal_started = None
        self.previous = None
        self.error = None
        self.resolved = self.pending is None
        self.transitions = 0
        self.awaiting_conversation = False
        self.conversation_until = 0.0

    def on_data_collectors_attached(self):
        self.collector = self.get_data_collector("experience")

    def step(self, observation):
        try:
            now = observation.timestamp
            if self.voice is not None:
                self.voice.poll()
            speech_interrupted = False
            if self.events is not None:
                for event in self.events.drain():
                    if event.kind == "speech_started":
                        self.awaiting_conversation = True
                        self.pending = None
                        if self.planner is not None:
                            self.planner.interrupt()
                        goal = (
                            Goal(skill="LOOK_AT", target=event.target, duration_s=1.0)
                            if event.target is not None
                            else WAIT
                        )
                        goal.validate_world(observation.world)
                        self.decision = Decision(goal=goal, source="fixed")
                        self.goal_started = now
                        speech_interrupted = True
                    elif event.kind == "conversation_decision":
                        if event.decision is None:
                            raise ValueError("conversation event missing decision")
                        event.decision.goal.validate_world(observation.world)
                        self.decision = event.decision
                        self.goal_started = now
                        self.awaiting_conversation = False
                        self.conversation_until = now + event.decision.goal.duration_s
                        speech_interrupted = True
                        if self.planner is not None:
                            self.planner.memory.record(
                                event.decision, "conversation_reply_planned", 0.0
                            )
                            if event.speaker_id is not None and any(
                                item.name == event.speaker_id and item.kind == "player"
                                for item in observation.world.objects
                            ):
                                self.planner.memory.record_interaction(event.speaker_id)
            if self.previous is not None:
                before, decision, command, action = self.previous
                source = observation.body.head.source
                outcome = (
                    "simulated"
                    if source == "simulated"
                    else (
                        "device_feedback"
                        if all(
                            s.valid
                            for s in (
                                observation.body.head,
                                observation.body.left,
                                observation.body.right,
                            )
                        )
                        else "unobserved"
                    )
                )
                reward = (
                    device_progress_reward(before, action, observation)
                    if outcome == "device_feedback"
                    else None
                )
                if (
                    self.planner is not None
                    and reward is not None
                    and not self.awaiting_conversation
                    and now >= self.conversation_until
                ):
                    self.planner.observe_reward(reward)
                next_motor_observation = None
                if self.policy.last_motor_observation is not None:
                    try:
                        next_motor_observation = self.policy.reach_observation(
                            observation.body,
                            observation.world.locate(decision.goal.target),
                        )
                    except ValueError:
                        pass  # The full transition still records unavailable feedback.
                self.collector.collect(
                    Transition(
                        observation=before,
                        decision=decision,
                        command=command,
                        action=action,
                        next_observation=observation,
                        reward=reward,
                        reward_kind="device_progress" if reward is not None else None,
                        outcome=outcome,
                        environment="vrchat" if source == "openvr_raw" else "mock",
                        motor_observation=self.policy.last_motor_observation,
                        motor_action=self.policy.last_motor_action,
                        next_motor_observation=next_motor_observation,
                    ).model_dump_json()
                )
                self.transitions += 1
            if self.pending is not None and self.pending.done():
                self.decision = self.pending.result()  # Validation/HTTP failure stops explicitly.
                self.pending, self.resolved, self.goal_started = None, True, now
            conversation_owns_body = (
                self.awaiting_conversation
                or now < self.conversation_until
                or (self.voice is not None and self.voice.pipeline.output.speaking)
            )
            if self.planner is not None and not speech_interrupted and not conversation_owns_body:
                dt_planner = (
                    1 / self.hz
                    if self.previous is None
                    else min(1.0, now - self.previous[0].timestamp)
                )
                next_decision = self.planner.tick(now, observation.world, dt=dt_planner)
                if next_decision is not None:
                    self.decision, self.goal_started = next_decision, now
            elif self.planner is not None and conversation_owns_body:
                dt_planner = (
                    1 / self.hz
                    if self.previous is None
                    else min(1.0, now - self.previous[0].timestamp)
                )
                self.planner.drives.advance(max(0.0, dt_planner), interacting=True)
            if self.goal_started is None:
                self.goal_started = now
            elapsed = now - self.goal_started
            goal = self.decision.goal
            # Finite intention: hold observed pose and release inputs after expiry.
            command = MotionCommand(goal=goal, elapsed_s=elapsed)
            dt = 1 / self.hz if self.previous is None else now - self.previous[0].timestamp
            # A late Windows frame must not produce a large catch-up movement.
            # Wall time still expires the goal; the independent supervisor stops
            # a producer that exceeds its heartbeat/target deadlines.
            dt = min(dt, 0.1)
            action = self.policy.step(observation.body, command, observation.world, dt)
            self.previous = (observation, self.decision, command, action)
            return action
        except Exception as exc:
            self.error = str(exc)
            raise


def run_session(
    output: Path,
    world: WorldState,
    decision: Decision | Future[Decision],
    *,
    duration: float = 10.0,
    hz: float = 60.0,
    live: LiveConfig | None = None,
    hmd_serial: str | None = None,
    planner: AutonomousPlanner | None = None,
    residual: ResidualPolicy | None = None,
    events: EventInbox | None = None,
    voice: LiveVoiceLoop | None = None,
    vision: LiveVisionLoop | None = None,
    gaze_policy: VisualGazePolicy | None = None,
    reach_policy: LearnedReachPolicy | None = None,
) -> dict:
    if not 1 <= duration <= 3600 or not 20 <= hz <= 120:
        raise ValueError("duration must be 1..3600 seconds and rate 20..120 Hz")
    if live and not hmd_serial:
        raise ValueError("live readback needs the expected HMD serial")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    (output / "session.json").write_text(
        json.dumps(
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_start": started,
                "mode": "live" if live else "mock",
                "hz": hz,
                "duration_s": duration,
                "avatar_visual_confirmation": False,
            }
        ),
        encoding="utf-8",
    )
    rest = live.safe_target if live else rest_target()
    supervisor = OutputSupervisor(output / "output-events.jsonl", live)
    sensor = OpenVRReadback(live, hmd_serial) if live else None
    environment = BodyEnvironment(supervisor, world, sensor, rest, vision)
    agent = BodyAgent(
        decision, hz, rest, planner, residual, events, voice, gaze_policy, reach_policy
    )
    replay = ReplayBuffer(max_size=min(200000, int(duration * hz) + 1))
    error = None
    try:
        if vision is not None:
            vision.start()
        if voice is not None:
            voice.start()
            (output / "voice-ready.json").write_text(
                json.dumps({"monotonic_at": time.perf_counter()}), encoding="utf-8"
            )
        launch(
            FixedIntervalInteraction.with_sleep_adjustor(agent, environment, interval=1 / hz),
            models={},
            buffers={"experience": replay},
            trainers={},
            config=LaunchConfig(
                states_dir=output / "states", max_uptime=duration, web_api_address=None
            ),
        )
        if agent.error or environment.error or supervisor.failed:
            raise RuntimeError(
                agent.error or environment.error or "output supervisor failed/stopped"
            )
        if not agent.resolved:
            raise RuntimeError("session ended before an LLM intention was received")
        if not environment.frames:
            raise RuntimeError("no body frames were produced")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if voice is not None:
                voice.stop()
            if vision is not None:
                vision.stop()
            environment.teardown()
        except Exception as exc:
            error = f"{error + '; ' if error else ''}cleanup: {type(exc).__name__}: {exc}"
            raise
        finally:
            latest_world = environment.vision.world() if vision and not vision.error else world
            summary = {
                "frames_attempted": environment.frames,
                "transitions": agent.transitions,
                "elapsed_s": time.perf_counter() - started,
                "decision": agent.decision.model_dump(mode="json"),
                "error": error,
                "mode": "live" if live else "mock",
                "avatar_visual_confirmation": False,
                "autonomous_requests": planner.requests if planner else 0,
                "interaction_memory": planner.memory.recent if planner else [],
                "vision_frames": vision.frames if vision else 0,
                "world_objects_at_end": [
                    item.model_dump(mode="json") for item in latest_world.objects
                ],
                "voice_diagnostics": dict(voice.diagnostics) if voice else None,
                "voice_transcript": (voice.pipeline.last_transcript if voice is not None else None),
                "voice_decision": (
                    voice.pipeline.last_decision.model_dump(mode="json")
                    if voice is not None and voice.pipeline.last_decision is not None
                    else None
                ),
                "tts_error": (
                    str(voice.pipeline.output.error)
                    if voice is not None and voice.pipeline.output.error is not None
                    else None
                ),
            }
            (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
