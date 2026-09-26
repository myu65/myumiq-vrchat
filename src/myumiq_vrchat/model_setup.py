"""Expand portable role profiles into the existing autonomous configuration."""

import json
from importlib.resources import files
from pathlib import Path

from .autonomous_body import AutonomousConfig
from .body import Frozen
from .decision_runtime import DecisionSettings
from .generation_config import GenerationConfig
from .shared_dialogue import DialogueSettings


class ModelProfile(Frozen):
    speech: GenerationConfig
    thought: GenerationConfig
    decision: DecisionSettings
    dialogue: DialogueSettings = DialogueSettings()
    planner_use_image: bool = False


def load_profile(name):
    profiles = json.loads(files("myumiq_vrchat").joinpath("model_profiles.json").read_text())
    if name == "hybrid":
        profile = {**profiles["openrouter"], "speech": profiles["local"]["speech"]}
    else:
        profile = profiles[name]
    return ModelProfile.model_validate(profile)


def configure(profile, state_dir, base=None):
    """Replace role settings, preserving the base's state and embodiment settings."""
    from .cli import outside_repo

    state = outside_repo(state_dir)
    data = (
        json.loads(base.model_dump_json())
        if base
        else {
            "memory": str(state / "interaction.json"),
            "purpose": {
                "state": str(state / "purpose.json"),
                "memory": str(state / "memory.sqlite3"),
            },
        }
    )
    data["llm"] = profile.speech.model_dump(mode="json")
    data["decision"] = profile.decision.model_dump(mode="json")
    data["dialogue"] = profile.dialogue.model_dump(mode="json")
    purpose = data.get("purpose") or {"state": str(state / "purpose.json")}
    data["purpose"] = {
        **purpose,
        "planner": profile.thought.model_dump(mode="json"),
        "planner_use_image": profile.planner_use_image,
    }
    return AutonomousConfig.model_validate_json(json.dumps(data))


def add_commands(subparsers):
    parser = subparsers.add_parser("models", help="configure and check replaceable model roles")
    commands = parser.add_subparsers(dest="model_command", required=True)
    init = commands.add_parser("init", help="write a new configuration outside the repository")
    profiles = init.add_mutually_exclusive_group()
    profiles.add_argument(
        "--profile", choices=("local", "openrouter", "openrouter-vision", "hybrid"), default="local"
    )
    profiles.add_argument("--profile-file", type=Path, help="custom speech/thought/decision JSON")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--state-dir", type=Path)
    init.add_argument("--base", type=Path, help="preserve existing device/body/memory settings")
    check = commands.add_parser("check", help="call configured models using synthetic input only")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--output", type=Path, help="new JSON report outside the repository")


def execute(args):
    from .cli import outside_repo

    if args.model_command == "init":
        output = outside_repo(args.output)
        profile = (
            ModelProfile.model_validate_json(args.profile_file.read_text(encoding="utf-8-sig"))
            if args.profile_file
            else load_profile(args.profile)
        )
        base = (
            AutonomousConfig.model_validate_json(
                outside_repo(args.base).read_text(encoding="utf-8-sig")
            )
            if args.base
            else None
        )
        config = configure(profile, args.state_dir or output.parent / "state", base)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(config.model_dump_json(indent=2) + "\n")
        print(
            json.dumps(
                {
                    "config": str(output),
                    "next": "myumiq models check --config <config>",
                    "devices": "preserved from base"
                    if base
                    else "configure voice/vision/body separately",
                }
            )
        )
        return 0
    from .model_checks import check_models

    config = AutonomousConfig.model_validate_json(
        outside_repo(args.config).read_text(encoding="utf-8-sig")
    )
    output = outside_repo(args.output) if args.output else None
    if output and output.exists():
        raise FileExistsError("model check output already exists")
    result = check_models(config)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)
    return 0 if result["ok"] else 1
