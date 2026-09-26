import json

import httpx
import pytest

from myumiq_vrchat.autonomous_body import Intent
from myumiq_vrchat.body import WorldObject, WorldState
from myumiq_vrchat.decision import (
    BackendConfig,
    Candidate,
    CandidateScore,
    DecisionInput,
    ScoreResult,
)
from myumiq_vrchat.decision_remote import HttpScorer, JevScorer
from myumiq_vrchat.decision_runtime import DecisionSettings, propose


def request():
    return DecisionInput(
        captured_at=10,
        candidates=(Candidate(id="a", description="Sit"), Candidate(id="b", description="Wait")),
    )


@pytest.mark.parametrize(
    "backend,endpoint",
    [
        ("jev", "https://api.typesafe.ai/v1/systemone"),
        ("jev_openrouter", "https://openrouter.ai/api/alpha/decisions"),
    ],
)
def test_independent_questions_and_order_preservation(monkeypatch, backend, endpoint):
    monkeypatch.setenv("TEST_KEY", "test-only")
    payloads = []

    def handle(req):
        assert str(req.url) == endpoint
        payload = json.loads(req.content)
        payloads.append(payload)
        assert "image_base64" not in payload["state"]
        return httpx.Response(
            200,
            json={
                "model": "test",
                "answers": {
                    k: {"type": "score", "score": 2 if k == "a" else 0, "confidence": 0.8}
                    for k in reversed(payload["questions"])
                },
            },
        )

    scorer = JevScorer(
        BackendConfig(backend=backend, api_key_env="TEST_KEY"),
        transport=httpx.MockTransport(handle),
    )
    first = request()
    second = first.model_copy(update={"candidates": tuple(reversed(first.candidates))})
    for value in (first, second):
        result = scorer.score(value)
        assert [s.id for s in result.scores] == [c.id for c in value.candidates]
        assert result.select(value, 11, 2).id == "a"
        assert not result.image_used
    assert payloads[0]["questions"] == payloads[1]["questions"]
    assert all(q["type"] == "score" for q in payloads[0]["questions"].values())
    scorer.close()


def test_images_require_explicit_text_only_consent():
    scorer = JevScorer(BackendConfig(backend="jev"))
    with pytest.raises(ValueError, match="text only"):
        scorer.score(request().model_copy(update={"image_base64": "eA=="}))
    scorer.close()


def test_freshness_identity_and_stable_ties():
    req = request()
    result = ScoreResult(
        backend="test",
        model="test",
        captured_at=10,
        image_used=False,
        semantics="relevance_logit",
        scores=(CandidateScore(id="a", score=0), CandidateScore(id="b", score=0)),
    )
    assert result.select(req, 10, 2).id == "a"
    for now in (9, 13):
        with pytest.raises(ValueError):
            result.select(req, now, 2)
    with pytest.raises(ValueError):
        result.model_copy(update={"scores": tuple(reversed(result.scores))}).validate_request(req)
    with pytest.raises(ValueError):
        CandidateScore(id="a", score=float("nan"))
    with pytest.raises(ValueError):
        DecisionInput(captured_at=1, candidates=(req.candidates[0], req.candidates[0]))


def test_runtime_proposals_respect_capabilities_and_vision_reach():
    world = WorldState(objects=(WorldObject(name="person", position=(1, 0, 1), source="vision"),))
    candidates = propose(Intent(skill="REACH", hand="right", target="person"), world, False)
    skills = [c.intent["skill"] for c in candidates]
    assert "REACH" not in skills and "WALK_IN_PLACE" not in skills
    assert "LOOK_AT" in skills
    assert len(skills) == len(set(skills))
    with pytest.raises(ValueError, match="isolated"):
        DecisionSettings(backend=BackendConfig(backend="qwen"))
    with pytest.raises(ValueError, match="loopback"):
        HttpScorer(BackendConfig(backend="http", endpoint="https://example.com"))


def test_runtime_result_is_validated_and_expired_requests_never_call_backend(monkeypatch):
    from myumiq_vrchat import decision_runtime as runtime
    from myumiq_vrchat.autonomy import Drives

    calls = []

    class Scorer:
        def score(self, req):
            calls.append(req)
            return ScoreResult(
                backend="test",
                model="test",
                captured_at=req.captured_at,
                image_used=True,
                semantics="relevance_logit",
                scores=tuple(
                    CandidateScore(id=c.id, score=1 if c.intent["skill"] == "SIT" else 0)
                    for c in req.candidates
                ),
            )

        def close(self):
            pass

    monkeypatch.setattr(runtime, "make_scorer", lambda cfg: Scorer())
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: 11.0)
    settings = DecisionSettings(backend=BackendConfig(backend="http"))
    intent, report = runtime.rerank(
        settings, Intent(skill="WAIT"), (WorldState(), "eA==", 10.0), Drives(), [], {}, False
    )
    assert intent.skill == "SIT" and report["image_used"]
    assert len(calls) == 1
    with pytest.raises(ValueError, match="expired"):
        runtime.rerank(
            settings, Intent(skill="WAIT"), (WorldState(), "eA==", 0.0), Drives(), [], {}, False
        )
    assert len(calls) == 1


def test_autonomous_worker_uses_fresh_body_and_records_decision(monkeypatch, tmp_path):
    import time

    from myumiq_vrchat import autonomous_body as module
    from myumiq_vrchat.autonomous_services import Services
    from myumiq_vrchat.body import simulated_body
    from myumiq_vrchat.cognition import LLMConfig
    from myumiq_vrchat.postures import posture_target

    class Vision:
        def decision_snapshot(self):
            return WorldState(), "eA==", time.perf_counter()

        def stop(self):
            pass

    def poll(self, now, enabled=True):
        self.vision = Vision()
        return WorldState(), []

    def rerank(settings, primary, snapshot, drives, recent, body, walking):
        assert body["timestamp"] == 123.0
        return Intent(skill="SIT"), {"backend": "test", "captured_at": snapshot[2]}

    monkeypatch.setattr(Services, "poll", poll)
    monkeypatch.setattr(module, "request_intent", lambda *args: Intent(skill="WAIT"))
    monkeypatch.setattr(module, "rerank", rerank)
    cfg = module.AutonomousConfig(
        llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
        memory=tmp_path / "memory.json",
        decision=DecisionSettings(backend=BackendConfig(backend="http")),
    )
    rest = posture_target("standing")
    agent = module.AutonomousBody(cfg, tmp_path, rest, None)
    agent.snapshot = simulated_body(rest, 123.0)
    agent.enable(True)
    agent.start()
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and "+decision:" not in agent.choice[3]:
            time.sleep(0.02)
        assert agent.choice[3] == "local_llm+decision:test"
        assert agent.choice[2].skill == "SIT"
        assert agent.health["decision"]["state"] == "running"
    finally:
        agent.close()
