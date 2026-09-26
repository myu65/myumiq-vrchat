import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import httpx
import pytest

from myumiq_vrchat.generation import ChatGenerator, GenerationSession
from myumiq_vrchat.generation_config import LLMConfig, OpenRouterConfig
from myumiq_vrchat.generation_stream import GenerationCancelled


def event(delta="", finish=None, model="speech"):
    return (
        "data: "
        + json.dumps(
            {"model": model, "choices": [{"delta": {"content": delta}, "finish_reason": finish}]}
        )
        + "\n\n"
    ).encode()


def test_sse_payload_metadata_and_sentence_before_finish():
    seen = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event('{"reply":"こんにちは。')
            assert seen == ['{"reply":"こんにちは。']
            yield event('", "topic":"挨拶"}', "stop")
            yield b"data: [DONE]\n\n"

    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] and body["model"] == "speech"
        return httpx.Response(200, stream=Stream())

    generator = ChatGenerator(
        LLMConfig(base_url="http://localhost:1", model="speech"),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = generator.request_stream([], {}, "speech", 100, seen.append)
        assert result.value["reply"] == "こんにちは。"
        assert result.metadata["streaming"] and result.metadata["model"] == "speech"
    finally:
        generator.close()


@pytest.mark.parametrize("before_headers", [True, False])
def test_cancellation_closes_stalled_network_without_waiting_for_timeout(
    monkeypatch, before_headers
):
    entered, closed = Event(), Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await asyncio.sleep(30)
            yield event("{}", "stop")

        async def aclose(self):
            closed.set()

    async def handler(request):
        if before_headers:
            entered.set()
            await asyncio.sleep(30)
        return httpx.Response(200, stream=Stream())

    monkeypatch.setattr(
        "myumiq_vrchat.generation.make_generator",
        lambda cfg: ChatGenerator(cfg, transport=httpx.MockTransport(handler)),
    )
    session = GenerationSession(LLMConfig(base_url="http://localhost:1", model="speech"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(session.request_stream, [], {}, "speech", 100, lambda _: None)
        try:
            assert entered.wait(2)
        finally:
            session.cancel()
            session.close()
        with pytest.raises(GenerationCancelled):
            future.result(timeout=2)
    if not before_headers:
        assert closed.is_set()


def test_remote_revision_is_checked_before_any_text_delivery(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    generator = ChatGenerator(
        OpenRouterConfig(
            adapter="openrouter_chat", model="vendor/speech", route={"provider": "fixture"}
        ),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, content=event('{"reply":"こんにちは。"}', "stop", model="other")
            )
        ),
    )
    seen = []
    try:
        with pytest.raises(ValueError, match="unexpected model"):
            generator.request_stream([], {}, "speech", 100, seen.append)
        assert not seen
    finally:
        generator.close()


def test_truncated_stream_never_counts_as_a_complete_reply():
    generator = ChatGenerator(
        LLMConfig(base_url="http://localhost:1", model="speech"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=event('{"reply":'))),
    )
    try:
        with pytest.raises(ValueError, match="complete decision"):
            generator.request_stream([], {}, "speech", 100, lambda _: None)
    finally:
        generator.close()
