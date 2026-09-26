"""Independent candidate scoring contract, with explicit modality/score semantics."""

import base64
import math
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from .adapters import AdapterSpec, load_adapter
from .body import Frozen, Number, WorldState
from .generation_config import ProviderRoute


class Candidate(Frozen):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = Field(min_length=1, max_length=1000)
    intent: dict = Field(default_factory=dict)


class DecisionInput(Frozen):
    world: WorldState = WorldState()
    state: dict = Field(default_factory=dict)
    candidates: tuple[Candidate, ...] = Field(min_length=1, max_length=64)
    image_base64: str | None = Field(default=None, max_length=12_000_000)
    image_mime: Literal["image/png", "image/jpeg"] = "image/png"
    captured_at: Number
    instruction: str = Field(
        default="Evaluate how appropriate this action is now, using the image and state.",
        min_length=1,
        max_length=2000,
    )

    @model_validator(mode="after")
    def unique_ids(self):
        if len({c.id for c in self.candidates}) != len(self.candidates):
            raise ValueError("candidate IDs must be unique")
        if self.image_base64 is not None:
            try:
                data = base64.b64decode(self.image_base64, validate=True)
            except Exception as exc:
                raise ValueError("invalid image base64") from exc
            if not data:
                raise ValueError("empty image")
        return self

    @classmethod
    def with_image(cls, path: Path, **kwargs):
        return cls(image_base64=base64.b64encode(path.read_bytes()).decode(), **kwargs)


class CandidateScore(Frozen):
    id: str
    score: Number
    confidence: Number | None = Field(default=None, ge=0, le=1)


class ScoreResult(Frozen):
    backend: str
    model: str
    scores: tuple[CandidateScore, ...]
    captured_at: Number
    image_used: bool
    semantics: Literal["relevance_logit", "rubric_score_0_1"]
    timings_ms: dict[str, float] = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    def validate_request(self, request: DecisionInput):
        if (
            [s.id for s in self.scores] != [c.id for c in request.candidates]
            or self.captured_at != request.captured_at
            or any(not math.isfinite(s.score) for s in self.scores)
        ):
            raise ValueError("scorer did not preserve candidate identity/snapshot")
        return self

    def select(self, request: DecisionInput, now: float, max_age_s: float) -> Candidate:
        self.validate_request(request)
        if not 0 <= now - self.captured_at <= max_age_s:
            raise ValueError("decision snapshot expired or is from the future")
        # Stable ID tie-break, never candidate input position.
        best = min(self.scores, key=lambda s: (-s.score, s.id))
        return next(c for c in request.candidates if c.id == best.id)


class Scorer(Protocol):
    def score(self, request: DecisionInput) -> ScoreResult: ...


class BackendConfig(Frozen):
    backend: Literal["qwen", "nemotron", "jev", "jev_openrouter", "http", "extension"]
    extension: AdapterSpec | None = None
    model_path: Path | None = None
    device: str = "cuda:0"
    expected_device_name: str | None = None
    batch_size: int = Field(default=8, ge=1, le=64)
    max_pixels: int = Field(default=131072, ge=4096, le=1843200)
    max_tiles: int = Field(default=1, ge=1, le=6)
    max_length: int = Field(default=2048, ge=128, le=10240)
    endpoint: str = "http://127.0.0.1:18520"
    api_key_env: str = "TYPESAFE_API_KEY"
    jev_model: str = "jev-latest"
    route: ProviderRoute | None = None
    expected_model: str | None = None
    allow_state_only: bool = False
    timeout_s: Number = Field(default=15, gt=0, le=120)

    @model_validator(mode="before")
    @classmethod
    def router_defaults(cls, value):
        if isinstance(value, dict) and value.get("backend") == "jev_openrouter":
            return {"api_key_env": "OPENROUTER_API_KEY", "jev_model": "typesafe/jev-1.13", **value}
        return value

    @model_validator(mode="after")
    def adapter_contract(self):
        if (self.backend == "extension") != (self.extension is not None):
            raise ValueError("extension scorer requires exactly one extension configuration")
        if self.route is not None and self.backend != "jev_openrouter":
            raise ValueError("provider route is only supported by jev_openrouter")
        return self


def make_scorer(config: BackendConfig) -> Scorer:
    if config.backend == "extension":
        return load_adapter("scorer", config.extension, methods=("score", "close"))
    if config.backend in ("qwen", "nemotron"):
        from .decision_local import LocalReranker

        return LocalReranker(config)
    from .decision_remote import HttpScorer, JevScorer

    return JevScorer(config) if config.backend in ("jev", "jev_openrouter") else HttpScorer(config)
