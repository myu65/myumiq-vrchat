import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import TypeAdapter

from myumiq_vrchat.adapters import AdapterSpec, load_adapter
from myumiq_vrchat.autonomous_body import AutonomousConfig
from myumiq_vrchat.decision import BackendConfig, Candidate, DecisionInput
from myumiq_vrchat.decision_remote import JevScorer
from myumiq_vrchat.decision_selection import LocalSelection
from myumiq_vrchat.generation import ChatGenerator, GenerationResult, request_json
from myumiq_vrchat.generation_config import GenerationConfig, LLMConfig, OpenRouterConfig


def remote(**updates):
    return OpenRouterConfig.model_validate(
        {
            "adapter": "openrouter_chat",
            "model": "google/gemma-4-26b-a4b-it",
            "route": {"provider": "darkbloom"},
            **updates,
        }
    )


def test_profiles_keep_local_security_and_allow_distinct_models_on_one_router(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        LLMConfig(base_url="https://openrouter.ai/api/v1", model="model")
    config = AutonomousConfig.model_validate_json(
        json.dumps(
            {
                "llm": remote(model="vendor/speech").model_dump(mode="json"),
                "memory": str(tmp_path / "legacy.json"),
                "purpose": {
                    "state": str(tmp_path / "purpose.json"),
                    "planner": remote().model_dump(mode="json"),
                },
                "decision": {"selection": {"llm": remote().model_dump(mode="json")}},
            }
        )
    )
    assert config.purpose.planner.adapter == "openrouter_chat"
    with pytest.raises(ValueError, match="separate model"):
        AutonomousConfig.model_validate_json(
            config.model_copy(update={"llm": remote()}).model_dump_json()
        )
    for key, value in [("base_url", "https://elsewhere.invalid"), ("models", ["x/y"])]:
        with pytest.raises(ValueError):
            remote(**{key: value})


def test_remote_payload_routing_budget_metadata_and_selection(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    seen = []

    def handler(request):
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer secret-test-key"
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "google/gemma-4-26b-a4b-it",
                "provider": "Darkbloom",
                "id": "fixture",
                "usage": {"cost": 0.00001, "prompt_tokens": 100},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"candidate_id":"hold","basis":"none"}'},
                    }
                ],
            },
        )

    generator = ChatGenerator(remote(max_tokens=256), transport=httpx.MockTransport(handler))
    monkeypatch.setattr("myumiq_vrchat.generation.make_generator", lambda _: generator)
    request = DecisionInput(
        captured_at=10, candidates=(Candidate(id="hold", description="Hold pose"),)
    )
    result = LocalSelection(remote(max_tokens=256)).choose(request)
    assert result.candidate_id == "hold" and "scores" not in result.model_dump()
    assert result.metadata["provider"] == "Darkbloom"
    assert result.metadata["usage"]["cost"] == 0.00001
    payload = seen[0]
    assert payload["provider"] == {
        "only": ["darkbloom"],
        "order": ["darkbloom"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {"prompt": 1.0, "completion": 2.0},
    }
    assert payload["reasoning"] == {"enabled": False}
    assert payload["max_tokens"] == 256
    assert payload["response_format"]["json_schema"]["strict"]
    assert len(seen) == 1
    assert generator.client.is_closed


@pytest.mark.parametrize(
    "status,reply",
    [
        (503, {"error": "secret-test-key private transcript"}),
        (200, {"error": {"message": "secret-test-key private transcript"}}),
        (200, {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}),
        (200, {"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]}),
        (200, {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]}),
    ],
)
def test_failure_is_bounded_sanitized_and_never_falls_back(monkeypatch, status, reply):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=reply)

    generator = ChatGenerator(remote(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises((ValueError, RuntimeError)) as error:
            generator.request([], {}, "test", 10)
        assert "secret-test-key" not in str(error.value) and "transcript" not in str(error.value)
        assert len(calls) == 1
    finally:
        generator.close()


def test_missing_key_and_oversized_input_do_not_make_network_requests(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    generator = ChatGenerator(
        remote(), transport=httpx.MockTransport(lambda _: pytest.fail("network"))
    )
    try:
        with pytest.raises(RuntimeError, match="missing API key"):
            generator.request([], {}, "test", 10)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        with pytest.raises(ValueError, match="size limit"):
            generator.request([{"role": "user", "content": "a" * 140000}], {}, "test", 10)
    finally:
        generator.close()


def test_wrong_returned_model_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    generator = ChatGenerator(
        remote(),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "model": "unexpected/model",
                    "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                },
            )
        ),
    )
    try:
        with pytest.raises(ValueError, match="unexpected model"):
            generator.request([], {}, "test", 10)
    finally:
        generator.close()


def test_jev_has_its_own_protocol_and_default_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    config = BackendConfig(backend="jev_openrouter", expected_model="typesafe/jev-1.13-20260917")

    def handler(request):
        assert request.url.path == "/api/alpha/decisions"
        payload = json.loads(request.content)
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["provider"]["only"] == ["typesafe"]
        assert "messages" not in payload
        assert all(q["type"] == "score" for q in payload["questions"].values())
        return httpx.Response(
            200,
            json={
                "model": config.expected_model,
                "provider": "TypeSafe",
                "answers": {"wait": {"type": "score", "score": 1.6}},
                "usage": {"cost": 0.00001},
            },
        )

    scorer = JevScorer(config, transport=httpx.MockTransport(handler))
    try:
        result = scorer.score(
            DecisionInput(captured_at=0, candidates=(Candidate(id="wait", description="Wait"),))
        )
        assert result.scores[0].score == 0.8 and result.metadata["provider"] == "TypeSafe"
        assert result.model == config.expected_model
    finally:
        scorer.close()


def test_installed_generation_extension_and_unknown_port(monkeypatch):
    calls, closed = [], []

    def request(*args):
        calls.append(args)
        return GenerationResult({"candidate_id": None}, 0.1, {"provider": "fixture"})

    def entries(*, group, name):
        if (group, name) != ("myumiq_vrchat.generation", "fixture"):
            return ()

        def factory(options):
            assert options["model"] == "separate" and options["timeout_s"] == 20
            return SimpleNamespace(request=request, close=lambda: closed.append(True))

        return (SimpleNamespace(load=lambda: factory),)

    monkeypatch.setattr("myumiq_vrchat.adapters.entry_points", entries)
    config = TypeAdapter(GenerationConfig).validate_python(
        {"adapter": "extension", "model": "separate", "extension": {"name": "fixture"}}
    )
    result = request_json(config, [], {}, "test", 10)
    assert result.value == {"candidate_id": None} and calls and closed == [True]
    with pytest.raises(ValueError, match="installed scorer"):
        load_adapter("scorer", AdapterSpec(name="fixture"), methods=("score",))
