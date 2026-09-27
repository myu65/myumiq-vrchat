"""Configurable finalized-audio recognizers; native input streaming is separate."""

import base64
import io
import json
import time
import wave
from typing import Literal

import httpx
from pydantic import Field, model_validator

from .body import Frozen, Number
from .generation_config import LLMConfig


class FasterWhisperSettings(Frozen):
    model: str = Field(min_length=1)
    device: Literal["cpu", "cuda"] = "cpu"
    device_index: int = Field(default=0, ge=0)
    compute_type: str = "int8"
    cpu_threads: int = Field(default=2, ge=1, le=32)
    language: str = Field(default="ja", min_length=2, max_length=20)


class LocalAudioChatSettings(Frozen):
    base_url: str
    model: str = Field(min_length=1)
    text_format: Literal["qwen3_asr", "text"] = "qwen3_asr"
    timeout_s: Number = Field(default=3.0, gt=0, le=30)
    warmup_timeout_s: Number = Field(default=60.0, gt=0, le=120)
    max_tokens: int = Field(default=512, ge=16, le=2048)
    max_audio_s: Number = Field(default=31.0, gt=0, le=120)
    context: str = Field(default="", max_length=2048)
    qwen3_language: str | None = Field(
        default=None, min_length=2, max_length=40, pattern=r"^[A-Za-z]+(?: [A-Za-z]+)*$"
    )

    @model_validator(mode="after")
    def local_endpoint(self):
        LLMConfig(base_url=self.base_url, model=self.model)
        if self.qwen3_language is not None and self.text_format != "qwen3_asr":
            raise ValueError("Qwen3 language prefill requires qwen3_asr text format")
        return self


class LocalAudioChatASR:
    """Audio-chat compatible service, pinned by configuration with no fallback."""

    def __init__(self, config: LocalAudioChatSettings):
        self.config = config
        self.metadata = {
            "backend": "local_audio_chat",
            "model": config.model,
            "base_url": config.base_url,
            "native_streaming": False,
            "language": config.qwen3_language or "auto",
        }

    def transcribe(self, audio: tuple[float, ...], sample_rate: int) -> str:
        return self._recognize(audio, sample_rate, self.config.timeout_s)

    def warmup(self):
        # No user audio is captured before runtime readiness. Silence is only a
        # kernel/model warmup and is never published as an input utterance.
        self._recognize((0.0,) * 16000, 16000, self.config.warmup_timeout_s, warmup=True)

    def _recognize(self, audio, sample_rate, timeout_s, *, warmup=False):
        import numpy as np

        if sample_rate != 16000 or not audio or len(audio) / sample_rate > self.config.max_audio_s:
            raise ValueError("ASR requires bounded non-empty 16 kHz mono audio")
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("ASR audio must be finite and mono")
        if not warmup and not np.any(samples):
            return ""  # Digitally silent PCM contains no words, whatever the decoder predicts.
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        messages = []
        if self.config.context:
            messages.append({"role": "system", "content": self.config.context})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(stream.getvalue()).decode("ascii"),
                            "format": "wav",
                        },
                    }
                ],
            }
        )
        if self.config.qwen3_language:
            messages.append(
                {"role": "assistant", "content": f"language {self.config.qwen3_language}<asr_text>"}
            )
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "cache_prompt": False,
            "messages": messages,
        }
        deadline = time.monotonic() + timeout_s
        with httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=min(2.0, timeout_s)),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            with client.stream(
                "POST", self.config.base_url.rstrip("/") + "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    if time.monotonic() > deadline:
                        raise TimeoutError("ASR response exceeded total deadline")
                    body.extend(chunk)
                    if len(body) > 131072:
                        raise ValueError("ASR response too large")
        data = json.loads(body)
        choice = data["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("ASR did not finish; refusing a truncated final transcript")
        text = choice["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("ASR response has no text")
        if self.config.text_format == "qwen3_asr":
            if "<asr_text>" in text:
                text = text.split("<asr_text>", 1)[1]
            elif self.config.qwen3_language is None:
                raise ValueError("Qwen3-ASR response is missing its transcript boundary")
        text = text.strip()
        # Conditioning text is a vocabulary hint, not captured audio. Models can
        # copy it on background noise; do not turn that into a user command.
        # A long contiguous copied passage is suspect too, but individual hinted
        # vocabulary words remain valid recognition results.
        hint, transcript = " ".join(self.config.context.split()), " ".join(text.split())
        if (
            not warmup
            and hint
            and transcript
            and (transcript == hint or len(transcript) >= 16 and transcript in hint)
        ):
            raise ValueError("ASR echoed its conditioning prompt; transcript is unverified")
        if len(text) > 8000:
            raise ValueError("ASR transcript too large")
        return text


def builtin_asr(spec):
    if spec.name == "local_audio_chat":
        return LocalAudioChatASR(
            LocalAudioChatSettings.model_validate_json(json.dumps(spec.options))
        )
    if spec.name == "faster_whisper":
        from .audio import FasterWhisperASR

        settings = FasterWhisperSettings.model_validate_json(json.dumps(spec.options))
        return FasterWhisperASR(**settings.model_dump())
    raise ValueError(f"unknown built-in ASR {spec.name}")
