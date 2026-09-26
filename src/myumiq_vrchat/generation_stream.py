"""Cancellable SSE transport for a role's existing background generation worker."""

import asyncio
import json
import time
from threading import Lock

import httpx


class GenerationCancelled(RuntimeError):
    pass


class ChatStream:
    def __init__(self, transport=None):
        self.transport = transport
        self.guard = Lock()
        self.cancelled = False
        self.loop = self.task = None

    def cancel(self):
        with self.guard:
            self.cancelled = True
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.task.cancel)

    def request(self, endpoint, headers, payload, timeout_s, expected_model, on_text):
        async def run():
            with self.guard:
                if self.cancelled:
                    raise GenerationCancelled("generation cancelled")
                self.loop, self.task = asyncio.get_running_loop(), asyncio.current_task()
            try:
                async with asyncio.timeout(timeout_s):
                    return await self._read(
                        endpoint, headers, payload, timeout_s, expected_model, on_text
                    )
            except asyncio.CancelledError:
                raise GenerationCancelled("generation cancelled") from None
            except (httpx.HTTPError, TimeoutError) as exc:
                raise RuntimeError(f"chat stream transport failed: {type(exc).__name__}") from None
            finally:
                with self.guard:
                    self.loop = self.task = None

        return asyncio.run(run())

    async def _read(self, endpoint, headers, payload, timeout_s, expected_model, on_text):
        encoded = json.dumps(
            {**payload, "stream": True, "stream_options": {"include_usage": True}},
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > 131072:
            raise ValueError("chat stream input exceeded the size limit")
        started = time.perf_counter()
        content, pending, received = "", b"", 0
        metadata, finish, model_verified = {}, None, expected_model is None
        async with httpx.AsyncClient(
            timeout=timeout_s,
            trust_env=False,
            follow_redirects=False,
            **({"transport": self.transport} if self.transport is not None else {}),
        ) as client:
            async with client.stream(
                "POST",
                endpoint,
                content=encoded,
                headers={"Content-Type": "application/json", **headers},
            ) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"chat stream returned HTTP {response.status_code}")
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    if received > 65536:
                        raise ValueError("chat stream response exceeded the size limit")
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            continue
                        try:
                            packet = json.loads(data)
                        except (ValueError, UnicodeDecodeError):
                            raise ValueError("chat stream returned invalid JSON") from None
                        if not isinstance(packet, dict) or "error" in packet:
                            raise ValueError("chat stream returned a model error")
                        if packet.get("model") is not None:
                            if expected_model is not None and packet["model"] != expected_model:
                                raise ValueError("OpenRouter returned an unexpected model revision")
                            model_verified = True
                        for key in ("model", "provider", "usage", "id"):
                            if packet.get(key) is not None:
                                metadata["request_id" if key == "id" else key] = packet[key]
                        choices = packet.get("choices", [])
                        if not choices:
                            continue  # Usage-only final event.
                        choice = choices[0]
                        delta = choice.get("delta", {}).get("content") or ""
                        if not isinstance(delta, str):
                            raise ValueError("chat stream content must be text")
                        if delta:
                            if not model_verified:
                                raise ValueError("OpenRouter stream did not identify its model")
                            if finish is not None:
                                raise ValueError("chat stream content after completion")
                            content += delta
                            on_text(delta)
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
        if finish != "stop":
            raise ValueError("LLM did not finish a complete decision")
        try:
            value = json.loads(content)
        except (ValueError, UnicodeDecodeError):
            raise ValueError("LLM did not return a JSON object") from None
        if not isinstance(value, dict):
            raise ValueError("LLM must return a JSON object")
        return value, time.perf_counter() - started, metadata
