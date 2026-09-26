import json

import httpx
import pytest

from myumiq_vrchat.body import WorldState
from myumiq_vrchat.cognition import LLMConfig, decide, decide_conversation


def install_response(monkeypatch, content, finish="stop", status=200):
    original = httpx.Client
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            status, json={"choices": [{"finish_reason": finish, "message": {"content": content}}]}
        )

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    return requests


def test_http_request_asks_only_for_high_level_intent(monkeypatch):
    requests = install_response(monkeypatch, '{"skill":"WAVE","duration_s":3,"hand":"right"}')
    result = decide(
        LLMConfig(base_url="http://localhost:1234/v1", model="local"), "右手を振って", WorldState()
    )
    assert result.goal.skill == "WAVE"
    assert result.source == "local_llm"
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[0]["max_tokens"] == 160
    variants = requests[0]["response_format"]["json_schema"]["schema"]["anyOf"]
    assert all("pose" not in variant["properties"] for variant in variants)
    assert {variant["properties"]["skill"]["const"] for variant in variants} == {
        "WAIT",
        "WAVE",
        "RETURN_TO_REST",
    }


def test_conversation_returns_reply_and_same_validated_intent(monkeypatch):
    requests = install_response(
        monkeypatch,
        '{"reply":"こんにちは","goal":{"skill":"WAVE","duration_s":1,"hand":"right","target":null}}',
    )
    reply, decision = decide_conversation(
        LLMConfig(base_url="http://localhost:1234/v1", model="local"), "こんにちは", WorldState()
    )
    assert reply == "こんにちは" and decision.goal.skill == "WAVE"
    schema = requests[0]["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"reply", "goal"}


@pytest.mark.parametrize("invalid_reply", ["OK", "明白了", "了解", "ー"])
def test_conversation_retries_invalid_language_once(monkeypatch, invalid_reply):
    from myumiq_vrchat import cognition

    calls = []

    def request(config, messages, *args):
        calls.append((config.timeout_s, list(messages)))
        return {
            "reply": invalid_reply if len(calls) == 1 else "はい、分かりました。",
            "goal": {"skill": "WAVE", "hand": "right", "duration_s": 1},
        }, 0.1

    monkeypatch.setattr(cognition, "_request", request)
    reply, decision = decide_conversation(
        LLMConfig(base_url="http://localhost:1234/v1", model="local"), "右手を振って", WorldState()
    )
    assert reply == "はい、分かりました。" and decision.goal.skill == "WAVE"
    assert len(calls) == 2 and calls[1][0] <= calls[0][0]
    assert len(calls[1][1]) == len(calls[0][1]) + 2
    assert calls[1][1][-2]["role"] == "assistant"
    assert calls[1][1][-1]["role"] == "user"


def test_conversation_does_not_retry_forever(monkeypatch):
    requests = install_response(
        monkeypatch, '{"reply":"OK","goal":{"skill":"WAIT","duration_s":1}}'
    )
    with pytest.raises(ValueError, match="Japanese"):
        decide_conversation(
            LLMConfig(base_url="http://localhost:1234/v1", model="local"),
            "こんにちは",
            WorldState(),
        )
    assert len(requests) == 2


@pytest.mark.parametrize(
    "content,finish,status",
    [
        ('{"skill":"WAIT","duration_s":1}', "length", 200),
        ('{"skill":"WAVE","duration_s":1,"hand":"right","joystick":1}', "stop", 200),
        ('{"skill":"LOOK_AT","duration_s":1,"target":"invented"}', "stop", 200),
        ('```json\n{"skill":"WAIT","duration_s":1}\n```', "stop", 200),
        ('{"skill":"WAIT","duration_s":1}', "stop", 503),
    ],
)
def test_http_or_schema_failure_never_becomes_an_intention(monkeypatch, content, finish, status):
    install_response(monkeypatch, content, finish, status)
    with pytest.raises((ValueError, RuntimeError)):
        decide(LLMConfig(base_url="http://localhost:1234/v1", model="local"), "test", WorldState())
