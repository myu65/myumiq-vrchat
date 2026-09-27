"""Run mock/live body sessions, validate intents and inspect replay."""

import argparse
import json
import sys
import time
from pathlib import Path

from .autonomy import AutonomousPlanner, InteractionMemory
from .backends.osc import LiveConfig
from .body import BodyGoal, BodyState, BodyTarget, WorldState
from .calibration import ResidualPolicy
from .cognition import Decision, Goal, LLMConfig, decide_async, decide_conversation
from .conversation import ConversationPipeline, LiveVoiceLoop, VoiceConfig
from .events import EventInbox
from .gaze import VisualGazePolicy, train_visual_gaze
from .learning import train_residual_policy
from .reach_policy import LearnedReachPolicy
from .replay import summarize
from .runtime import run_session
from .vision import LiveVisionLoop, VisionConfig


def outside_repo(path: Path) -> Path:
    resolved = path.resolve()
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").exists() and (resolved == root or root in resolved.parents):
        raise ValueError("put machine configuration and run artifacts outside the repository")
    # Also enforce the boundary for a wheel install, whose source root is site-packages.
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValueError("put machine configuration and run artifacts outside a Git checkout")
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(prog="myumiq")
    subs = parser.add_subparsers(dest="command", required=True)
    from . import model_setup

    model_setup.add_commands(subs)
    run = subs.add_parser("run", help="PAMIQ body session (mock unless --live-config is supplied)")
    run.add_argument("--output", type=Path, required=True, help="new run directory outside repo")
    choice = run.add_mutually_exclusive_group(required=True)
    choice.add_argument(
        "--goal", help='JSON intent, e.g. {"skill":"WAVE","hand":"right","duration_s":3}'
    )
    choice.add_argument("--prompt", help="natural-language instruction for the local LLM")
    choice.add_argument("--autonomous", action="store_true", help="replan after each finite action")
    run.add_argument("--llm-config", type=Path)
    run.add_argument("--world", type=Path, help="explicit fixture/manual target geometry JSON")
    run.add_argument("--live-config", type=Path)
    run.add_argument("--hmd-serial")
    run.add_argument("--duration", type=float, default=10.0)
    run.add_argument("--hz", type=float, default=60.0)
    run.add_argument("--motor-policy", type=Path, help="promoted residual policy JSON")
    run.add_argument(
        "--gaze-policy", type=Path, help="supervised evaluation of a fitted visual LOOK_AT policy"
    )
    run.add_argument(
        "--reach-policy",
        type=Path,
        help="supervised evaluation of a Unity-trained right REACH actor",
    )
    run.add_argument("--voice-config", type=Path, help="live voice devices/models JSON")
    run.add_argument("--vision-config", type=Path, help="low-rate live visual detector JSON")
    run.add_argument("--memory-state", type=Path, help="persistent interaction memory JSON")
    run.add_argument("--prepared-file", type=Path, help="new model-readiness file outside repo")
    run.add_argument("--start-file", type=Path, help="wait for a new external start signal")
    replay = subs.add_parser("replay", help="validate and summarize saved JSONL transitions")
    replay.add_argument("path", type=Path)
    schema = subs.add_parser("schema", help="print a JSON schema for local configuration")
    schema.add_argument(
        "kind",
        choices=(
            "goal",
            "world",
            "llm",
            "live",
            "voice",
            "vision",
            "body-state",
            "body-goal",
            "body-target",
        ),
    )
    train = subs.add_parser("train-motor", help="fit a motor residual from real OpenVR replay")
    train.add_argument("replay", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--skill", choices=("WAVE", "REACH"), default="WAVE")
    train.add_argument("--hand", choices=("left", "right"), default="right")
    train_gaze = subs.add_parser(
        "train-gaze", help="fit visual inverse dynamics from settled VRChat replay"
    )
    train_gaze.add_argument("replay", type=Path)
    train_gaze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            return model_setup.execute(args)
        elif args.command == "schema":
            result = {
                "goal": Goal,
                "world": WorldState,
                "llm": LLMConfig,
                "live": LiveConfig,
                "voice": VoiceConfig,
                "vision": VisionConfig,
                "body-state": BodyState,
                "body-goal": BodyGoal,
                "body-target": BodyTarget,
            }[args.kind].model_json_schema()
        elif args.command == "replay":
            result = summarize(args.path)
        elif args.command == "train-motor":
            result = train_residual_policy(
                outside_repo(args.replay),
                outside_repo(args.output),
                skill=args.skill,
                hand=args.hand,
            )
        elif args.command == "train-gaze":
            result = train_visual_gaze(outside_repo(args.replay), outside_repo(args.output))
        else:
            output = outside_repo(args.output)
            if bool(args.prepared_file) != bool(args.start_file):
                raise ValueError("--prepared-file and --start-file must be supplied together")
            prepared = outside_repo(args.prepared_file) if args.prepared_file else None
            start_signal = outside_repo(args.start_file) if args.start_file else None
            if prepared and (prepared.exists() or start_signal.exists()):
                raise ValueError("startup handshake paths must be new for each run")
            world = (
                WorldState.model_validate_json(args.world.read_text(encoding="utf-8-sig"))
                if args.world
                else WorldState()
            )
            reach_policy = (
                LearnedReachPolicy(outside_repo(args.reach_policy)) if args.reach_policy else None
            )
            planner = None
            config = None
            if args.autonomous:
                if not args.llm_config:
                    raise ValueError("--autonomous requires --llm-config")
                config = LLMConfig.model_validate_json(
                    outside_repo(args.llm_config).read_text(encoding="utf-8-sig")
                )
                memory_path = outside_repo(args.memory_state) if args.memory_state else None
                planner = AutonomousPlanner(
                    config,
                    memory=InteractionMemory.load(memory_path) if memory_path else None,
                )
                decision = Decision(goal=Goal(skill="WAIT", duration_s=1.0), source="fixed")
            elif args.prompt:
                if not args.llm_config:
                    raise ValueError("--prompt requires --llm-config")
                config = LLMConfig.model_validate_json(
                    outside_repo(args.llm_config).read_text(encoding="utf-8-sig")
                )
                decision = decide_async(config, args.prompt, world)
            else:
                goal = Goal.model_validate_json(args.goal)
                goal.validate_world(world)
                decision = Decision(goal=goal, source="fixed")
            events = EventInbox() if args.voice_config else None
            vision = None
            if args.vision_config:
                vision_config = VisionConfig.model_validate_json(
                    outside_repo(args.vision_config).read_text(encoding="utf-8-sig")
                )
                from .service_adapters import make_detector

                if vision_config.model is not None:
                    outside_repo(vision_config.model)
                detector = make_detector(vision_config)
                vision = LiveVisionLoop(
                    detector,
                    title=vision_config.window_title,
                    hz=vision_config.hz,
                    capture_backend=vision_config.capture_backend,
                    fast_hz=vision_config.fast_hz,
                )
            voice = None
            if args.voice_config:
                if config is None:
                    raise ValueError(
                        "--voice-config requires --prompt or --autonomous with --llm-config"
                    )
                voice_config = VoiceConfig.model_validate_json(
                    outside_repo(args.voice_config).read_text(encoding="utf-8-sig")
                )
                from .service_adapters import make_asr, make_output, make_vad

                if voice_config.silero_model is not None:
                    outside_repo(voice_config.silero_model)
                output_device = make_output(voice_config)
                pipeline = ConversationPipeline(
                    make_asr(voice_config),
                    output_device,
                    events,
                    lambda transcript, current_world: decide_conversation(
                        config, transcript, current_world
                    ),
                    partial_transcripts=voice_config.partial_transcripts,
                    asr_queue_size=voice_config.asr_queue_size,
                    asr_queue_audio_s=voice_config.asr_queue_audio_s,
                )
                voice = LiveVoiceLoop(
                    voice_config.input_device,
                    voice_config.input_device_name,
                    make_vad(voice_config),
                    pipeline,
                    vision.world if vision else lambda: world,
                    loopback_speaker_name=voice_config.loopback_speaker_name,
                    input_gain=voice_config.input_gain,
                )
            if prepared:
                prepared.write_text(
                    json.dumps({"models_ready_at": time.perf_counter()}), encoding="utf-8"
                )
                deadline = time.monotonic() + 600
                while not start_signal.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("startup signal did not arrive within 600 seconds")
                    time.sleep(0.1)
            live = (
                LiveConfig.model_validate_json(
                    outside_repo(args.live_config).read_text(encoding="utf-8-sig")
                )
                if args.live_config
                else None
            )
            result = run_session(
                output,
                world,
                decision,
                duration=args.duration,
                hz=args.hz,
                live=live,
                hmd_serial=args.hmd_serial,
                planner=planner,
                residual=(
                    ResidualPolicy.model_validate_json(
                        outside_repo(args.motor_policy).read_text(encoding="utf-8-sig")
                    )
                    if args.motor_policy
                    else None
                ),
                events=events,
                voice=voice,
                vision=vision,
                gaze_policy=(
                    VisualGazePolicy.model_validate_json(
                        outside_repo(args.gaze_policy).read_text(encoding="utf-8-sig")
                    )
                    if args.gaze_policy
                    else None
                ),
                reach_policy=reach_policy,
            )
            if planner is not None and args.memory_state:
                planner.memory.save(outside_repo(args.memory_state))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
