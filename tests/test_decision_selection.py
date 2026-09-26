import json

import pytest

from myumiq_vrchat.cognition import LLMConfig
from myumiq_vrchat.decision import Candidate, DecisionInput
from myumiq_vrchat.decision_selection import LocalSelection


def request():
    return DecisionInput(
        captured_at=10.0,
        state={
            "conversation": [{"role": "user", "text": "立って"}],
            "body": {"pelvis_height": 0.4},
            "last_outcome": {"success": False},
        },
        candidates=(
            Candidate(id="keep", description="姿勢を保つ"),
            Candidate(id="stand", description="立ち上がる", intent={"skill": "STAND"}),
        ),
    )


def test_unavailable_semantic_requirement_cannot_be_replaced_by_another_skill(monkeypatch):
    observed = request().model_copy(
        update={
            "state": {
                "capabilities": [
                    {"name": "FOLLOW", "available": False},
                    {"name": "STAND", "available": True},
                ]
            }
        }
    )
    monkeypatch.setattr(
        "myumiq_vrchat.decision_selection._request",
        lambda *args: (
            {
                "requested_capabilities": ["FOLLOW"],
                "request_status": "action",
                "request_reason": "ついていく",
                "candidate_id": "stand",
                "basis": "latest_utterance",
            },
            0.1,
        ),
    )
    chosen = LocalSelection(
        LLMConfig(base_url="http://localhost:1", model="test"), understand_requests=True
    ).choose(observed)
    assert chosen.request_status == "unsupported" and chosen.candidate_id is None


def test_stop_assessment_cannot_start_a_different_motion(monkeypatch):
    observed = request().model_copy(
        update={"state": {"capabilities": [{"name": "WAIT", "available": True}]}}
    )
    monkeypatch.setattr(
        "myumiq_vrchat.decision_selection._request",
        lambda *args: (
            {
                "requested_capabilities": ["WAIT"],
                "request_status": "stop",
                "request_reason": "停止する",
                "candidate_id": "stand",
                "basis": "latest_utterance",
            },
            0.1,
        ),
    )
    with pytest.raises(ValueError, match="conflicts"):
        LocalSelection(
            LLMConfig(base_url="http://localhost:1", model="test"), understand_requests=True
        ).choose(observed)


@pytest.mark.parametrize("selected", ["stand", None])
def test_selection_preserves_context_abstention_identity_and_has_no_fake_scores(
    monkeypatch, selected
):
    calls = []

    def choose(config, messages, schema, name, budget):
        calls.append(config.model)
        payload = json.loads(messages[-1]["content"])
        assert payload["state"] == request().state
        assert schema["properties"]["candidate_id"]["enum"] == ["keep", "stand", None]
        return {"candidate_id": selected, "basis": "latest_utterance" if selected else "none"}, 0.2

    monkeypatch.setattr("myumiq_vrchat.decision_selection._request", choose)
    selector = LocalSelection(LLMConfig(base_url="http://127.0.0.1:1234/v1", model="separate"))
    result = selector.choose(request())
    chosen = result.select(request(), 10.5, 2.0)
    assert (chosen.id if chosen else None) == selected
    assert not hasattr(result, "scores") and calls == ["separate"]
    with pytest.raises(ValueError, match="expired"):
        result.select(request(), 13.0, 2.0)
    with pytest.raises(ValueError, match="identity"):
        result.select(
            request().model_copy(update={"candidates": request().candidates[::-1]}), 10.5, 2.0
        )


@pytest.mark.parametrize(
    "value", [{"candidate_id": "invented"}, {"candidate_id": "stand", "speech": "done"}]
)
def test_unvalidated_selection_has_no_fallback_or_execution(monkeypatch, value):
    monkeypatch.setattr("myumiq_vrchat.decision_selection._request", lambda *args: (value, 0.1))
    selector = LocalSelection(LLMConfig(base_url="http://127.0.0.1:1234/v1", model="separate"))
    with pytest.raises(ValueError, match="selection must"):
        selector.choose(request())


def test_selection_does_not_claim_image_input_or_accept_external_endpoint(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "myumiq_vrchat.decision_selection._request",
        lambda *args: (calls.append(args) or {"candidate_id": None, "basis": "none"}, 0.1),
    )
    config = LLMConfig(base_url="http://localhost:1234/v1", model="separate")
    image_request = request().model_copy(update={"image_base64": "eA=="})
    with pytest.raises(ValueError, match="state-only"):
        LocalSelection(config).choose(image_request)
    assert not calls
    result = LocalSelection(config, allow_state_only=True).choose(image_request)
    assert not result.image_used and result.metadata["image_unsupported"]
    assert "eA==" not in json.dumps(calls[0][1])
    with pytest.raises(ValueError, match="loopback"):
        LLMConfig(base_url="https://example.com/v1", model="separate")


def test_image_selection_preserves_snapshot_and_explicit_missing_image_fallback(monkeypatch):
    calls = []

    def respond(*args):
        calls.append(args)
        value = {"candidate_id": None, "basis": "none"}
        if "visible_target_ids" in args[2]["required"]:
            value["visible_target_ids"] = []
            value["scene"] = "An empty room."
        return value, 0.1

    monkeypatch.setattr("myumiq_vrchat.decision_selection._request", respond)
    config = LLMConfig(base_url="http://localhost:1234/v1", model="vision-model")
    selector = LocalSelection(config, use_image=True)
    image_request = request().model_copy(
        update={"image_base64": "eA==", "image_mime": "image/jpeg"}
    )
    result = selector.choose(image_request)
    content = calls[-1][1][-1]["content"]
    assert json.loads(content[0]["text"])["state"] == request().state
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,eA=="
    assert result.image_used and result.captured_at == image_request.captured_at
    assert result.metadata["condition"] == "image_and_state"
    with pytest.raises(ValueError, match="requires a fresh image"):
        selector.choose(request())
    assert len(calls) == 1
    result = LocalSelection(config, use_image=True, allow_state_only=True).choose(request())
    assert not result.image_used and isinstance(calls[-1][1][-1]["content"], str)


@pytest.mark.parametrize("visible", [[], ["invented"], ["person", "person"], "person"])
def test_image_target_selection_requires_supplied_visible_identity(monkeypatch, visible):
    monkeypatch.setattr(
        "myumiq_vrchat.decision_selection._request",
        lambda *args: (
            {
                "scene": "An empty room.",
                "candidate_id": "look",
                "basis": "observation",
                "visible_target_ids": visible,
            },
            0.1,
        ),
    )
    config = LLMConfig(base_url="http://localhost:1234/v1", model="vision-model")
    image_request = request().model_copy(
        update={
            "image_base64": "eA==",
            "candidates": (
                Candidate(
                    id="look",
                    description="Look at person",
                    intent={"skill": "LOOK_AT", "target": "person"},
                ),
            ),
        }
    )
    with pytest.raises(ValueError, match="ground"):
        LocalSelection(config, use_image=True).choose(image_request)


def test_completed_request_is_not_reclassified_when_current_capability_is_unavailable(monkeypatch):
    observed = request().model_copy(
        update={
            "state": {
                "latest_user_utterance": "まっすぐ少し進んで",
                "capabilities": [{"name": "MOVE_FORWARD", "available": False}],
                "fulfilled_request_capabilities": ["MOVE_FORWARD"],
            }
        }
    )
    monkeypatch.setattr(
        "myumiq_vrchat.decision_selection._request",
        lambda *args: (
            {
                "requested_capabilities": ["MOVE_FORWARD"],
                "request_status": "unsupported",
                "request_reason": "現在の能力が使えない",
                "candidate_id": None,
                "basis": "none",
            },
            0.1,
        ),
    )
    result = LocalSelection(
        LLMConfig(base_url="http://localhost:1", model="test"), understand_requests=True
    ).choose(observed)
    assert result.request_status == "action" and result.candidate_id is None
