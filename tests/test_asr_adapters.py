"""Bounded local ASR requests and explicitly selected inference devices."""

import base64
import io
import json
import sys
import wave
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from myumiq_vrchat.asr_adapters import LocalAudioChatASR, LocalAudioChatSettings
from myumiq_vrchat.conversation import VoiceConfig
from myumiq_vrchat.service_adapters import make_asr


def recognizer(monkeypatch, handler, **settings):
    client = httpx.Client
    monkeypatch.setattr(
        "myumiq_vrchat.asr_adapters.httpx.Client",
        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
    )
    return LocalAudioChatASR(
        LocalAudioChatSettings(
            base_url="http://127.0.0.1:18531/v1", model="qwen3-asr-0.6b", **settings
        )
    )


def response(text="language Japanese<asr_text>今度は立って。", finish="stop"):
    return httpx.Response(
        200, json={"choices": [{"message": {"content": text}, "finish_reason": finish}]}
    )


def test_audio_request_is_independent_pcm_and_cold_timeout_is_separate(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        return response()

    asr = recognizer(monkeypatch, handle, timeout_s=3.0, warmup_timeout_s=60.0)
    asr.warmup()
    for _ in range(2):
        assert asr.transcribe((0.0, 0.5, -2.0), 16000) == "今度は立って。"
    assert [r.extensions["timeout"]["read"] for r in requests] == [60.0, 3.0, 3.0]
    payload = json.loads(requests[-1].content)
    assert payload["model"] == "qwen3-asr-0.6b" and not payload["cache_prompt"]
    assert len(payload["messages"]) == 1
    content = payload["messages"][0]["content"][0]
    assert content["type"] == "input_audio"
    with wave.open(io.BytesIO(base64.b64decode(content["input_audio"]["data"]))) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        assert np.frombuffer(wav.readframes(3), dtype="<i2").tolist() == [0, 16383, -32767]
    assert asr.metadata["native_streaming"] is False


@pytest.mark.parametrize(
    ("text", "finish", "error"),
    [
        ("途中だけ", "length", "truncated"),
        ("unsupported format", "stop", "boundary"),
        (None, "stop", "no text"),
        ("<asr_text>" + "あ" * 8001, "stop", "too large"),
    ],
    ids=["truncated", "missing-boundary", "no-text", "oversized-text"],
)
def test_partial_or_malformed_responses_are_not_final_inputs(monkeypatch, text, finish, error):
    asr = recognizer(monkeypatch, lambda _: response(text, finish))
    with pytest.raises(ValueError, match=error):
        asr.transcribe((0.1,), 16000)


@pytest.mark.parametrize(
    "url",
    [
        "https://remote.example/v1",
        "http://user:secret@localhost/v1",
        "http://localhost/v1?key=secret",
    ],
)
def test_local_asr_cannot_implicitly_send_audio_remotely(url):
    with pytest.raises(ValueError, match="loopback"):
        LocalAudioChatSettings(base_url=url, model="asr")


def test_size_and_total_deadline_are_enforced(monkeypatch):
    asr = recognizer(monkeypatch, lambda _: httpx.Response(200, content=b"x" * 131073))
    with pytest.raises(ValueError, match="too large"):
        asr.transcribe((0.1,), 16000)
    clock = iter([0.0, 4.0])
    monkeypatch.setattr(
        "myumiq_vrchat.asr_adapters.time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    with pytest.raises(TimeoutError, match="deadline"):
        asr.transcribe((0.1,), 16000)


@pytest.mark.parametrize(
    ("audio", "rate"),
    [
        ((0.0,), 48000),
        ((), 16000),
        ((float("nan"),), 16000),
        (((0.0, 0.0),), 16000),
        ((0.0,) * 16001, 16000),
    ],
)
def test_invalid_audio_is_rejected_before_request(monkeypatch, audio, rate):
    asr = recognizer(monkeypatch, lambda _: pytest.fail("invalid audio sent"), max_audio_s=1.0)
    with pytest.raises(ValueError):
        asr.transcribe(audio, rate)


def test_whisper_device_is_configurable_and_receives_pcm_without_temporary_files(monkeypatch):
    observed = {}

    class Whisper:
        def __init__(self, model, **kwargs):
            observed.update(model=model, **kwargs)

        def transcribe(self, samples, **kwargs):
            assert isinstance(samples, np.ndarray) and samples.dtype == np.float32
            observed.update(samples=samples.tolist(), decoding=kwargs)
            return iter([SimpleNamespace(text=" 日本語")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Whisper))
    cfg = VoiceConfig.model_validate_json(
        json.dumps(
            {
                "input_device": 0,
                "input_device_name": "fixture",
                "output_device": 1,
                "output_device_name": "fixture out",
                "vad_adapter": {"name": "fixture"},
                "asr_adapter": {
                    "name": "faster_whisper",
                    "options": {
                        "model": "local/small",
                        "device": "cuda",
                        "device_index": 1,
                        "compute_type": "float16",
                        "cpu_threads": 2,
                    },
                },
            }
        )
    )
    asr = make_asr(cfg)
    assert asr.transcribe((2.0, -2.0, 0.0), 16000) == "日本語"
    assert observed["device"] == "cuda" and observed["device_index"] == 1
    assert observed["cpu_threads"] == 2 and observed["compute_type"] == "float16"
    assert observed["samples"] == [1.0, -1.0, 0.0]
    assert observed["decoding"]["condition_on_previous_text"] is False


@pytest.mark.parametrize("text", ["language Japanese<asr_text>手を振って。", "手を振って。", ""])
def test_fixed_language_prefill_accepts_full_or_suffix_without_history(monkeypatch, text):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return response(text)

    asr = recognizer(monkeypatch, handle, qwen3_language="Japanese", context="日本語で会話中。")
    for _ in range(2):
        assert asr.transcribe((0.1,), 16000) == ("手を振って。" if text else "")
    assert requests[0] == requests[1]
    assert [m["role"] for m in requests[0]["messages"]] == ["system", "user", "assistant"]
    assert requests[0]["messages"][-1]["content"] == "language Japanese<asr_text>"
    assert asr.metadata["language"] == "Japanese"


@pytest.mark.parametrize("echo", ["full", "vocabulary", None])
def test_conditioning_prompt_cannot_become_a_heard_user_command(monkeypatch, echo):
    context = "VRChatで日本語の会話をしています。用語: アバター、しゃがむ、立つ、手を振る、足踏み、踊る、うなずく。"
    text = (
        context
        if echo == "full"
        else context.split("用語: ")[1]
        if echo
        else "アバター、しゃがんで。"
    )
    asr = recognizer(
        monkeypatch, lambda _: response(text), context=context, qwen3_language="Japanese"
    )
    if echo:
        with pytest.raises(ValueError, match="conditioning prompt"):
            asr.transcribe((0.1,) * 160, 16000)
        asr.warmup()  # Warmup only proves model readiness; its text is never an input event.
    else:
        assert asr.transcribe((0.1,) * 160, 16000) == text


def test_language_prefill_is_specific_and_context_is_bounded():
    for options in (
        {"qwen3_language": "Japanese", "text_format": "text"},
        {"qwen3_language": "Japanese<asr_text>"},
        {"context": "a" * 2049},
    ):
        with pytest.raises(ValueError):
            LocalAudioChatSettings(base_url="http://localhost:18531/v1", model="asr", **options)


def test_digital_silence_is_not_sent_as_speech_but_warmup_still_loads_the_model(monkeypatch):
    requests = []
    asr = recognizer(monkeypatch, lambda req: requests.append(req) or response())
    assert asr.transcribe((0.0,) * 16000, 16000) == ""
    assert not requests
    asr.warmup()
    assert len(requests) == 1
