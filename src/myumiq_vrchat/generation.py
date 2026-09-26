"""Bounded structured generation; provider protocols stay outside cognitive roles."""

import json
import math
import os
import time
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Protocol

import httpx

from .adapters import load_adapter
from .generation_config import ExtensionGenerationConfig, GenerationConfig, OpenRouterConfig
from .generation_stream import ChatStream, GenerationCancelled


@dataclass(frozen=True)
class GenerationResult:
    value: dict
    elapsed_s: float
    metadata: dict = field(default_factory=dict)

    def __iter__(self):
        # Existing callers unpack (value, latency); telemetry stays out of model JSON.
        yield self.value
        yield self.elapsed_s


class JsonGenerator(Protocol):
    def request(
        self, messages: list[dict], schema: dict, name: str, max_tokens: int
    ) -> GenerationResult: ...
    def close(self) -> None: ...


def read_json(client, endpoint, body, timeout_s, *, headers=None, label="model"):
    encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > 131072:
        raise ValueError(f"{label} input exceeded the size limit")
    started = time.perf_counter()
    try:
        with client.stream(
            "POST",
            endpoint,
            content=encoded,
            headers={"Content-Type": "application/json", **(headers or {})},
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"{label} returned HTTP {response.status_code}")
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > 65536 or time.perf_counter() - started > timeout_s:
                    raise ValueError(f"{label} response exceeded the size/time limit")
        return json.loads(data), time.perf_counter() - started
    except httpx.HTTPError as exc:
        # Provider bodies and headers can contain credentials or private context.
        raise RuntimeError(f"{label} transport failed: {type(exc).__name__}") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError(f"{label} returned invalid JSON") from None


class ChatGenerator:
    def __init__(self, config, *, transport=None):
        self.config = config
        self.stream = ChatStream(transport)
        self.client = httpx.Client(
            timeout=config.timeout_s,
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(
                max_connections=2, max_keepalive_connections=1, keepalive_expiry=120.0
            ),
            **({"transport": transport} if transport is not None else {}),
        )

    def close(self):
        self.client.close()

    def cancel(self):
        self.stream.cancel()

    def _payload(self, messages, schema, name, max_tokens):
        cfg = self.config
        remote = isinstance(cfg, OpenRouterConfig)
        body = {
            "model": cfg.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": cfg.max_tokens or max_tokens,
        }
        if remote or cfg.structured_output:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            }
        headers = {}
        if remote:
            key = os.environ.get(cfg.api_key_env)
            if not key:
                raise RuntimeError(f"missing API key environment variable: {cfg.api_key_env}")
            headers["Authorization"] = "Bearer " + key
            body["provider"] = cfg.route.payload(structured=True)
            body["reasoning"] = cfg.reasoning.model_dump(exclude_none=True)
            endpoint = "https://openrouter.ai/api/v1/chat/completions"
        else:
            endpoint = cfg.base_url.rstrip("/") + "/chat/completions"
        return endpoint, headers, body

    def request_stream(self, messages, schema, name, max_tokens, on_text):
        cfg = self.config
        remote = isinstance(cfg, OpenRouterConfig)
        endpoint, headers, body = self._payload(messages, schema, name, max_tokens)
        value, elapsed, metadata = self.stream.request(
            endpoint,
            headers,
            body,
            cfg.timeout_s,
            (cfg.expected_model or cfg.model) if remote else None,
            on_text,
        )
        return GenerationResult(
            value,
            elapsed,
            {
                "adapter": cfg.adapter,
                "requested_model": cfg.model,
                "requested_provider": cfg.route.provider if remote else None,
                "streaming": True,
                **metadata,
            },
        )

    def request(self, messages, schema, name, max_tokens):
        cfg = self.config
        remote = isinstance(cfg, OpenRouterConfig)
        endpoint, headers, body = self._payload(messages, schema, name, max_tokens)
        result, elapsed = read_json(
            self.client, endpoint, body, cfg.timeout_s, headers=headers, label=cfg.adapter
        )
        if "error" in result:
            raise RuntimeError(f"{cfg.adapter} returned a model error")
        try:
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("LLM did not finish a complete decision")
            value = json.loads(choice["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            raise ValueError("LLM did not return a JSON object") from None
        if not isinstance(value, dict):
            raise ValueError("LLM must return a JSON object")
        if remote and result.get("model") != (cfg.expected_model or cfg.model):
            raise ValueError("OpenRouter returned an unexpected model revision")
        return GenerationResult(
            value,
            elapsed,
            {
                "adapter": cfg.adapter,
                "requested_model": cfg.model,
                "model": result.get("model", cfg.model),
                "provider": result.get("provider"),
                "requested_provider": cfg.route.provider if remote else None,
                "usage": result.get("usage", {}),
                "request_id": result.get("id"),
            },
        )


def make_generator(config: GenerationConfig) -> JsonGenerator:
    if isinstance(config, ExtensionGenerationConfig):
        spec = config.extension.model_copy(
            update={
                "options": {
                    **config.extension.options,
                    "model": config.model,
                    "timeout_s": config.timeout_s,
                }
            }
        )
        return load_adapter("generation", spec, methods=("request", "close"))
    return ChatGenerator(config)


def _invoke(generator, config, messages, schema, name, max_tokens, on_text=None):
    started = time.perf_counter()
    stream = getattr(generator, "request_stream", None)
    result = (
        stream(messages, schema, name, config.max_tokens or max_tokens, on_text)
        if on_text is not None and callable(stream)
        else generator.request(messages, schema, name, config.max_tokens or max_tokens)
    )
    if not isinstance(result, GenerationResult) or not isinstance(result.value, dict):
        raise TypeError("generation adapter must return GenerationResult with a JSON object")
    if (
        not math.isfinite(result.elapsed_s)
        or result.elapsed_s < 0
        or result.elapsed_s > config.timeout_s
        or time.perf_counter() - started > config.timeout_s
    ):
        raise ValueError("generation adapter exceeded the request deadline")
    return result


class GenerationSession:
    """One role's bounded worker; lazy construction, reuse and deferred cleanup."""

    def __init__(self, config):
        self.config, self.generator = config, None
        self.guard = Lock()
        self.active = self.closed = False
        self.cancelled = Event()

    def request(self, messages, schema, name, max_tokens):
        return self._request(messages, schema, name, max_tokens)

    def request_stream(self, messages, schema, name, max_tokens, on_text):
        return self._request(messages, schema, name, max_tokens, on_text)

    def _request(self, messages, schema, name, max_tokens, on_text=None):
        with self.guard:
            if self.closed or self.active:
                raise RuntimeError("generation session closed or request already active")
            self.active = True
        try:
            if self.generator is None:
                self.generator = make_generator(self.config)
            if self.cancelled.is_set():
                raise GenerationCancelled("generation cancelled")
            result = _invoke(
                self.generator, self.config, messages, schema, name, max_tokens, on_text
            )
            if self.cancelled.is_set():
                raise GenerationCancelled("generation cancelled")
            return result
        finally:
            with self.guard:
                self.active = False
                dispose = self.generator if self.closed else None
                if dispose is not None:
                    self.generator = None
            if dispose is not None:
                dispose.close()

    def cancel(self):
        """Cancel cooperative transports without waiting on their network worker."""
        self.cancelled.set()
        cancel = getattr(self.generator, "cancel", None)
        if callable(cancel):
            cancel()

    def close(self):
        with self.guard:
            self.closed = True
            dispose = self.generator if not self.active else None
            if dispose is not None:
                self.generator = None
        if dispose is not None:
            dispose.close()


def request_json(config, messages, schema, name, max_tokens):
    generator = make_generator(config)
    try:
        return _invoke(generator, config, messages, schema, name, max_tokens)
    finally:
        generator.close()
