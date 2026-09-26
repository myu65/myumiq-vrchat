import json

import httpx
import pytest

from myumiq_vrchat.cli import main
from myumiq_vrchat.generation import ChatGenerator
from myumiq_vrchat.model_checks import check_models
from myumiq_vrchat.model_setup import configure, load_profile


def test_profile_switch_preserves_state_devices_learning_and_body(tmp_path):
    base = configure(load_profile("local"), tmp_path / "original")
    data = json.loads(base.model_dump_json())
    data.update(
        voice=str(tmp_path / "voice.json"),
        vision=str(tmp_path / "vision.json"),
        articulated_tasks=str(tmp_path / "tasks.json"),
        retry_s=17,
    )
    data["purpose"]["retry_s"] = 23
    base = type(base).model_validate_json(json.dumps(data))
    for name in ("local", "openrouter", "openrouter-vision", "hybrid"):
        changed = configure(load_profile(name), tmp_path / "unused", base)
        assert changed.memory == base.memory
        assert changed.purpose.state == base.purpose.state
        assert changed.purpose.memory == base.purpose.memory
        assert changed.purpose.learning == base.purpose.learning
        assert changed.purpose.retry_s == 23
        for key in ("voice", "vision", "articulated_tasks", "exploration", "retry_s"):
            assert getattr(changed, key) == getattr(base, key)
        assert changed.llm.model != changed.purpose.planner.model
        assert "qwen2.5" not in changed.model_dump_json().lower()


def test_init_default_local_is_editable_and_never_overwrites(tmp_path, capsys):
    output = tmp_path / "config.json"
    args = ["models", "init", "--output", str(output)]
    assert main(args) == 0
    original = output.read_bytes()
    assert json.loads(original)["llm"]["adapter"] == "local_chat"
    assert not (tmp_path / "state").exists()
    assert main(args) == 1
    assert output.read_bytes() == original
    assert "FileExistsError" in capsys.readouterr().err


def test_custom_profile_accepts_adapter_and_model_without_allowlist(tmp_path):
    profile = load_profile("local")
    data = json.loads(profile.model_dump_json())
    data["speech"] = {
        "adapter": "extension",
        "model": "my-new-voice-model",
        "extension": {"name": "installed_generator", "options": {}},
    }
    custom = type(profile).model_validate(data)
    result = configure(custom, tmp_path)
    assert result.llm.extension.name == "installed_generator"
    data["speech"] = data["thought"]
    with pytest.raises(ValueError, match="separate model"):
        configure(type(profile).model_validate(data), tmp_path)


def test_checks_use_real_schemas_and_changed_turns_without_memory_files(tmp_path, monkeypatch):
    config = configure(load_profile("local"), tmp_path)
    calls = []

    def serve(request):
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        calls.append(name)
        if name == "shared_dialogue":
            token = "ひまわり" if "ひまわり" in body["messages"][-1]["content"] else "あさがお"
            value = {"reply": token, "topic": "合言葉", "remember_quotes": []}
        elif name == "purpose_plan":
            value = {
                "description": "少し待つ",
                "reason": "周囲に対象がない",
                "success_description": "待機を続ける",
                "steps": [{"capability": "WAIT"}],
            }
        else:
            value = {
                "candidate_id": "stand"
                if "今度は立って" in body["messages"][-1]["content"]
                else "crouch",
                "basis": "latest_utterance",
            }
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
            },
        )

    monkeypatch.setattr(
        "myumiq_vrchat.generation.make_generator",
        lambda cfg: ChatGenerator(cfg, transport=httpx.MockTransport(serve)),
    )
    report = check_models(config)
    assert report["ok"] and not report["vrchat_tested"] and not report["audio_tested"]
    assert calls == [
        "shared_dialogue",
        "shared_dialogue",
        "purpose_plan",
        "body_selection",
        "body_selection",
    ]
    assert list(tmp_path.iterdir()) == []


def test_failed_probe_does_not_hide_other_roles_or_echo_provider_secrets(tmp_path, monkeypatch):
    config = configure(load_profile("local"), tmp_path)

    def fail(*args):
        raise ValueError("credential-that-must-not-appear")

    monkeypatch.setattr("myumiq_vrchat.model_checks._speech", fail)
    monkeypatch.setattr("myumiq_vrchat.model_checks._thought", lambda *args: {"fixture": True})
    monkeypatch.setattr("myumiq_vrchat.model_checks._decision", lambda cfg: {"fixture": True})
    report = check_models(config)
    assert not report["ok"]
    assert [role["ok"] for role in report["roles"]] == [False, True, True]
    assert "credential-that-must-not-appear" not in json.dumps(report)


def test_missing_api_key_is_configuration_failure_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = configure(load_profile("openrouter"), tmp_path)
    report = check_models(config)
    assert not report["ok"]
    assert all(role["error"] == "missing_credentials" for role in report["roles"])
