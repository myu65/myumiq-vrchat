"""Fixed per-role generation profiles; local URLs never implicitly become remote."""

from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .adapters import AdapterSpec
from .body import Frozen, Number


class LLMConfig(Frozen):
    adapter: Literal["local_chat"] = "local_chat"
    base_url: str
    model: str = Field(min_length=1)
    model_identity: str | None = Field(default=None, min_length=1)
    timeout_s: Number = Field(default=30.0, gt=0, le=120)
    structured_output: bool = True
    max_tokens: int | None = Field(default=None, ge=1, le=8192)

    @model_validator(mode="after")
    def local_only(self) -> Self:
        parts = urlsplit(self.base_url)
        if (
            parts.scheme not in ("http", "https")
            or parts.hostname not in ("127.0.0.1", "localhost", "::1")
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError(
                "this local adapter requires a loopback HTTP endpoint without credentials"
            )
        return self


class ProviderRoute(Frozen):
    # One full endpoint slug pins routing, including quantization where available.
    provider: str = Field(min_length=1, max_length=120)
    max_prompt_price: Number = Field(default=1.0, gt=0, le=100)
    max_completion_price: Number = Field(default=2.0, gt=0, le=100)

    def payload(self, *, structured=False):
        return {
            "only": [self.provider],
            "order": [self.provider],
            "allow_fallbacks": False,
            "require_parameters": structured,
            "max_price": {"prompt": self.max_prompt_price, "completion": self.max_completion_price},
        }


class ReasoningConfig(Frozen):
    enabled: bool = False
    effort: Literal["minimal", "low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def consistent(self):
        if not self.enabled and self.effort is not None:
            raise ValueError("reasoning effort requires enabled=true")
        return self


class OpenRouterConfig(Frozen):
    adapter: Literal["openrouter_chat"]
    model: str = Field(min_length=1, pattern=r"^[\w.-]+/[\w./-]+$")
    model_identity: str | None = Field(default=None, min_length=1)
    # Optional exact returned revision, when the provider exposes a dated identity.
    expected_model: str | None = Field(default=None, min_length=1)
    route: ProviderRoute
    api_key_env: str = Field(default="OPENROUTER_API_KEY", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    timeout_s: Number = Field(default=20.0, gt=0, le=120)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    reasoning: ReasoningConfig = ReasoningConfig()


class ExtensionGenerationConfig(Frozen):
    adapter: Literal["extension"]
    extension: AdapterSpec
    model: str = Field(min_length=1)
    model_identity: str | None = Field(default=None, min_length=1)
    timeout_s: Number = Field(default=20.0, gt=0, le=120)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


GenerationConfig = LLMConfig | OpenRouterConfig | ExtensionGenerationConfig


def require_separate_models(speech: GenerationConfig, thought: GenerationConfig):
    a, b = speech.model_identity or speech.model, thought.model_identity or thought.model
    if a == b:
        raise ValueError("body reasoning requires separate model weights from speech")
    if isinstance(speech, LLMConfig) and isinstance(thought, LLMConfig):
        # All accepted hosts are loopback aliases. Same port denotes the same server.
        def port(config):
            url = urlsplit(config.base_url)
            return url.port or (443 if url.scheme == "https" else 80)

        if port(speech) == port(thought):
            raise ValueError("body reasoning requires a separate local endpoint from speech")
