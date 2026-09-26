"""Streaming speech events and barge-in contracts independent of model vendors."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AudioFrame:
    samples: tuple[float, ...]
    sample_rate: int
    timestamp: float


@dataclass(frozen=True)
class SpeechEvent:
    kind: str
    timestamp: float
    audio: tuple[float, ...] = ()


class StreamingASR(Protocol):
    def transcribe(self, audio: tuple[float, ...], sample_rate: int) -> str: ...


class FasterWhisperASR:
    """Local utterance ASR fallback; VAD remains responsible for streaming onset."""

    def __init__(
        self,
        model: str,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        device_index: int = 0,
        cpu_threads: int = 2,
        language: str = "ja",
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("install the optional faster-whisper dependency") from exc
        self._model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            device_index=device_index,
            cpu_threads=cpu_threads,
        )
        self.language = language
        self.metadata = {
            "backend": "faster_whisper",
            "model": model,
            "device": device,
            "device_index": device_index,
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
        }

    def transcribe(self, audio: tuple[float, ...], sample_rate: int) -> str:
        if sample_rate != 16000 or not audio:
            raise ValueError("ASR requires non-empty 16 kHz mono audio")
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("ASR audio must be finite and mono")
        segments, _ = self._model.transcribe(
            np.clip(samples, -1, 1),
            language=self.language,
            beam_size=1,
            vad_filter=False,
            temperature=0,
            condition_on_previous_text=False,
        )
        return "".join(segment.text for segment in segments).strip()

    def warmup(self):
        self.transcribe((0.0,) * 16000, 16000)


class SpeechOutput(Protocol):
    @property
    def speaking(self) -> bool: ...
    def stop(self) -> None: ...


class EnergyVAD:
    """Deterministic fallback/event gate; production can substitute Silero VAD."""

    def __init__(self, *, threshold: float = 0.02, release_frames: int = 4):
        self.threshold, self.release_frames = threshold, release_frames
        self.active, self.silent = False, 0
        self.buffer: list[float] = []

    def accept(self, frame: AudioFrame) -> list[SpeechEvent]:
        if frame.sample_rate <= 0 or not frame.samples:
            raise ValueError("invalid audio frame")
        rms = math.sqrt(sum(x * x for x in frame.samples) / len(frame.samples))
        events = []
        if rms >= self.threshold:
            self.silent = 0
            self.buffer.extend(frame.samples)
            if not self.active:
                self.active = True
                events.append(SpeechEvent("speech_started", frame.timestamp))
        elif self.active:
            self.silent += 1
            self.buffer.extend(frame.samples)
            if self.silent >= self.release_frames:
                self.active, self.silent = False, 0
                events.append(SpeechEvent("speech_ended", frame.timestamp, tuple(self.buffer)))
                self.buffer.clear()
        return events


class SileroVAD:
    """Silero v6 streaming adapter. Import/model loading is explicit and local."""

    def __init__(self, model_path: Path, *, threshold: float = 0.5, release_frames: int = 4):
        if not 0 < threshold < 1 or release_frames < 1:
            raise ValueError("invalid Silero VAD parameters")
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("install numpy and onnxruntime for Silero VAD") from exc
        if not model_path.is_file():
            raise ValueError("Silero ONNX model path does not exist")
        self._numpy = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self.threshold, self.release_frames = threshold, release_frames
        self.active, self.silent = False, 0
        self.buffer: list[float] = []
        self._active_frames = 0

    def accept(self, frame: AudioFrame) -> list[SpeechEvent]:
        if frame.sample_rate != 16000 or len(frame.samples) != 512:
            raise ValueError("Silero VAD requires 512 mono samples at 16 kHz")
        audio = self._numpy.asarray(frame.samples, dtype=self._numpy.float32)[None, :]
        # Match Silero's streaming ONNX wrapper: each 512-sample frame needs
        # the preceding 64 samples in addition to the recurrent state.
        model_input = self._numpy.concatenate((self._context, audio), axis=1)
        probability, self._state = self._session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": self._numpy.asarray(16000, dtype="int64"),
            },
        )
        self._context = audio[:, -64:].copy()
        probability = float(probability[0, 0])
        events = []
        self.buffer.extend(frame.samples)
        if probability >= self.threshold:
            self.silent = 0
            if not self.active:
                self.active = True
                events.append(SpeechEvent("speech_started", frame.timestamp))
        elif self.active:
            self.silent += 1
            if self.silent >= self.release_frames:
                self.active, self.silent = False, 0
                events.append(SpeechEvent("speech_ended", frame.timestamp, tuple(self.buffer)))
                self.buffer.clear()
        elif len(self.buffer) > 16000:
            del self.buffer[:-16000]
        if self.active:
            self._active_frames += 1
            # Bounded provisional windows. Final recognition retains up to 30s.
            if self._active_frames % 8 == 0:
                events.append(
                    SpeechEvent("speech_active", frame.timestamp, tuple(self.buffer[-128000:]))
                )
            if len(self.buffer) >= 480000:
                events.append(SpeechEvent("speech_ended", frame.timestamp, tuple(self.buffer)))
                self.buffer.clear()
                self.active, self.silent = False, 0
        else:
            self._active_frames = 0
        return events


class BargeInCoordinator:
    def __init__(self, output: SpeechOutput):
        self.output = output
        self.interruptions = 0

    def on_event(self, event: SpeechEvent) -> bool:
        if event.kind == "speech_started" and self.output.speaking:
            self.output.stop()
            self.interruptions += 1
            return True
        return False
