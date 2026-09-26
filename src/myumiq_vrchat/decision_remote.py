"""Documented TypeSafe API and local inference transport; no guessed image API."""

import os
import time
from urllib.parse import urlsplit

import httpx

from .decision import CandidateScore, ScoreResult
from .generation import read_json
from .generation_config import ProviderRoute


class JevScorer:
    def __init__(self, config, transport=None):
        self.config = config
        self.client = httpx.Client(
            timeout=config.timeout_s, trust_env=False, follow_redirects=False, transport=transport
        )

    def close(self):
        self.client.close()

    def score(self, request):
        if request.image_base64 is not None and not self.config.allow_state_only:
            raise ValueError(
                "official Jev currently supports text only; explicit allow_state_only required"
            )
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise RuntimeError(f"missing API key environment variable: {self.config.api_key_env}")
        # One independent rubric question per candidate, not a Choice over actions.
        questions = {
            c.id: {
                "type": "score",
                "instructions": {
                    "task": request.instruction,
                    "candidate_action": c.description,
                    "intent": c.intent,
                },
                "criteria": [
                    "Contradicts the observed situation or cannot be executed now",
                    "Plausible but not useful for the current goal",
                    "Appropriate and useful for the current situation",
                ],
            }
            for c in request.candidates
        }
        started = time.perf_counter()
        endpoint = (
            "https://openrouter.ai/api/alpha/decisions"
            if self.config.backend == "jev_openrouter"
            else "https://api.typesafe.ai/v1/systemone"
        )
        payload = {
            "model": self.config.jev_model,
            "state": {"world": request.world.model_dump(mode="json"), "state": request.state},
            "questions": questions,
        }
        route = self.config.route or ProviderRoute(provider="typesafe")
        if self.config.backend == "jev_openrouter":
            payload["provider"] = route.payload()
        data, _ = read_json(
            self.client,
            endpoint,
            payload,
            self.config.timeout_s,
            headers={"Authorization": "Bearer " + key},
            label="official Jev",
        )
        if self.config.expected_model and data.get("model") != self.config.expected_model:
            raise ValueError("Jev returned an unexpected model revision")
        answers = data["answers"]
        if set(answers) != set(questions):
            raise ValueError("Jev response candidate IDs do not match")
        scores = []
        for c in request.candidates:
            answer = answers[c.id]
            if answer["type"] != "score" or not 0 <= answer["score"] <= 2:
                raise ValueError("invalid Jev rubric response")
            scores.append(
                CandidateScore(
                    id=c.id, score=answer["score"] / 2, confidence=answer.get("confidence")
                )
            )
        return ScoreResult(
            backend=self.config.backend,
            model=data["model"],
            scores=tuple(scores),
            captured_at=request.captured_at,
            image_used=False,
            semantics="rubric_score_0_1",
            timings_ms={"end_to_end": (time.perf_counter() - started) * 1000},
            metadata={
                "condition": "state_only",
                "image_unsupported": True,
                "requested_model": self.config.jev_model,
                "requested_provider": route.provider
                if self.config.backend == "jev_openrouter"
                else None,
                "provider": data.get("provider"),
                "request_id": data.get("id"),
                "usage": data.get("usage", {}),
            },
        ).validate_request(request)


class HttpScorer:
    def __init__(self, config):
        parsed = urlsplit(config.endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("decision service must be a credential-free loopback HTTP endpoint")
        self.url = config.endpoint.rstrip("/") + "/score"
        self.client = httpx.Client(
            timeout=config.timeout_s, trust_env=False, follow_redirects=False
        )

    def close(self):
        self.client.close()

    def score(self, request):
        started = time.perf_counter()
        response = self.client.post(
            self.url,
            content=request.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        result = ScoreResult.model_validate_json(response.content).validate_request(request)
        timings = dict(
            result.timings_ms, transport_end_to_end=(time.perf_counter() - started) * 1000
        )
        return result.model_copy(update={"timings_ms": timings})
