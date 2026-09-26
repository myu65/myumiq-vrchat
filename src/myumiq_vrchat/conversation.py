"""Non-blocking VAD/ASR/cognition/TTS orchestration."""

import time
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, RLock, Thread
from typing import Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adapters import AdapterSpec
from .audio import AudioFrame, BargeInCoordinator, SpeechEvent, StreamingASR
from .body import WorldState
from .cognition import Decision
from .events import EventInbox, RuntimeEvent


class ReplyOutput(Protocol):
    @property
    def speaking(self) -> bool: ...
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...


class VoiceActivityDetector(Protocol):
    def accept(self, frame: AudioFrame) -> list[SpeechEvent]: ...


class VoiceConfig(BaseModel):
    """Machine-local voice devices and models; device names guard against index drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    input_device: int | None = Field(default=None, ge=0)
    input_device_name: str | None = Field(default=None, min_length=1, max_length=200)
    loopback_speaker_name: str | None = Field(default=None, min_length=1, max_length=200)
    output_device: int = Field(ge=0)
    output_device_name: str = Field(min_length=1, max_length=200)
    silero_model: Path | None = None
    whisper_model: str | None = Field(default=None, min_length=1, max_length=500)
    asr_adapter: AdapterSpec | None = None
    vad_adapter: AdapterSpec | None = None
    output_adapter: AdapterSpec | None = None
    sapi_voice: str = Field(default="Microsoft Haruka Desktop", min_length=1, max_length=100)
    streaming_tts_endpoint: str | None = None
    tts_leading_silence_max_s: float = Field(default=0.0, ge=0, le=0.75)
    vad_threshold: float = Field(default=0.5, gt=0, lt=1)
    vad_release_frames: int = Field(default=12, ge=1, le=100)
    input_gain: float = Field(default=1.0, ge=0.1, le=16)
    partial_transcripts: bool = False
    asr_queue_size: int = Field(default=8, ge=1, le=64)
    asr_queue_audio_s: float = Field(default=60, ge=1, le=300)

    @model_validator(mode="after")
    def one_input_route(self):
        if self.asr_adapter is None and self.whisper_model is None:
            raise ValueError("choose an ASR adapter or whisper_model")
        if self.vad_adapter is None and self.silero_model is None:
            raise ValueError("choose a VAD adapter or silero_model")
        if self.output_adapter and self.streaming_tts_endpoint:
            raise ValueError("choose one speech output adapter")
        device_route = self.input_device is not None or self.input_device_name is not None
        if device_route != (self.input_device is not None and self.input_device_name is not None):
            raise ValueError("input device index and name must be specified together")
        if device_route == (self.loopback_speaker_name is not None):
            raise ValueError("select exactly one input device or loopback speaker")
        return self


@dataclass(frozen=True)
class CapturedUtterance:
    event: SpeechEvent
    world: WorldState
    sample_rate: int
    sequence: int
    reply_generation: int
    queued_at: float

    @property
    def duration_s(self) -> float:
        return len(self.event.audio) / self.sample_rate


@dataclass(frozen=True)
class RecognizedUtterance:
    text: str
    response: tuple[str, Decision] | None = None


class ConversationPipeline:
    """Speech onset is synchronous; ASR and response generation run off motor thread."""

    def __init__(
        self,
        asr: StreamingASR,
        output: ReplyOutput,
        events: EventInbox,
        responder: Callable[[str, WorldState], tuple[str, Decision]] | None,
        *,
        partial_transcripts: bool = True,
        asr_queue_size: int = 8,
        asr_queue_audio_s: float = 60,
    ):
        if asr_queue_size < 1 or asr_queue_audio_s <= 0:
            raise ValueError("ASR backlog limits must be positive")
        self.asr, self.output, self.events, self.responder = asr, output, events, responder
        self.partial_transcripts = partial_transcripts
        self._asr_diagnostics = {}
        self.barge_in = BargeInCoordinator(output)
        self.pending: Future[RecognizedUtterance] | None = None
        self.last_transcript: str | None = None
        self.last_decision: Decision | None = None
        self._lock = RLock()
        self._reply_generation = 0
        self._input_session = uuid.uuid4().hex
        self._input_sequence = 0
        self._input_open = False
        self._closed = False
        self._pending_input: CapturedUtterance | None = None
        self._queued: deque[CapturedUtterance] = deque()
        self._queue_size, self._queue_audio_s = asr_queue_size, asr_queue_audio_s
        self._rejected_inputs = 0
        self._overflow_notice: RuntimeEvent | None = None
        self._asr_lock = Lock()
        self._partial_busy = False
        self._last_partial_at = -float("inf")

    @property
    def asr_diagnostics(self):
        # Snapshot only; no model call or waiting for inference on the executive.
        with self._lock:
            pending = self._pending_input
            return {
                **self._asr_diagnostics,
                "runtime": getattr(self.asr, "metadata", {}),
                "pending_input_sequence": pending.sequence if pending else None,
                "pending_age_s": time.perf_counter() - pending.queued_at if pending else 0.0,
                "waiting_inputs": len(self._queued),
                "buffered_audio_s": sum(item.duration_s for item in self._queued)
                + (pending.duration_s if pending else 0.0),
                "rejected_inputs": self._rejected_inputs,
            }

    def accept(self, event: SpeechEvent, world: WorldState, sample_rate: int = 16000) -> None:
        with self._lock:
            self._accept(event, world, sample_rate)

    def _accept(self, event: SpeechEvent, world: WorldState, sample_rate: int) -> None:
        if self._closed:
            return
        if event.kind == "speech_started":
            self._begin_input()
            players = [item for item in world.objects if item.kind == "player"]
            target = (
                min(players, key=lambda item: sum(value * value for value in item.position)).name
                if players
                else None
            )
            self.events.publish(self._input_event("speech_started", event.timestamp, target=target))
            self.barge_in.on_event(event)
            return
        if event.kind == "speech_active":
            if not self._input_open:
                self._begin_input()
            self.events.publish(self._input_event("speech_active", event.timestamp))
            if (
                self.partial_transcripts
                and event.audio
                and not self._partial_busy
                and self.pending is None
                and event.timestamp - self._last_partial_at >= 0.8
            ):
                self._partial_busy = True
                self._last_partial_at = event.timestamp
                generation = self._reply_generation
                sequence = self._input_sequence

                def partial():
                    try:
                        with self._asr_lock:
                            text = self.asr.transcribe(event.audio, sample_rate)
                        with self._lock:
                            if (
                                not self._closed
                                and generation == self._reply_generation
                                and self.pending is None
                                and text
                            ):
                                self.events.publish(
                                    self._input_event(
                                        "partial_transcript",
                                        event.timestamp,
                                        sequence=sequence,
                                        text=text,
                                        source="asr_chunked_provisional",
                                    )
                                )
                    except Exception as exc:
                        with self._lock:
                            if not self._closed and generation == self._reply_generation:
                                self.events.publish(
                                    self._input_event(
                                        "partial_asr_error",
                                        event.timestamp,
                                        sequence=sequence,
                                        text=str(exc)[:200],
                                        source="asr",
                                    )
                                )
                    finally:
                        with self._lock:
                            self._partial_busy = False

                Thread(target=partial, name="myumiq-partial-asr", daemon=True).start()
            return
        if event.kind != "speech_ended" or not event.audio:
            return
        if not self._input_open:
            self._begin_input()
        self._input_open = False
        captured = CapturedUtterance(
            event,
            world,
            sample_rate,
            self._input_sequence,
            self._reply_generation,
            time.perf_counter(),
        )
        self.events.publish(self._input_event("speech_ended", event.timestamp))
        buffered_s = sum(item.duration_s for item in self._queued)
        if self._pending_input is not None:
            buffered_s += self._pending_input.duration_s
        if (
            self.pending is not None and len(self._queued) >= self._queue_size
        ) or buffered_s + captured.duration_s > self._queue_audio_s:
            # Keep already accepted inputs. Coalesce the overload notice, not
            # their audio; record the cumulative loss even if the inbox is full.
            self._rejected_inputs += 1
            self._overflow_notice = self._input_event(
                "audio_input_overflow",
                event.timestamp,
                text=f"ASR backlog full; rejected_inputs={self._rejected_inputs}",
                source="asr",
            )
            return
        if self.pending is not None:
            self._queued.append(captured)
            return
        self._start_recognition(captured)

    def _begin_input(self) -> None:
        self._reply_generation += 1
        self._input_sequence += 1
        self._input_open = True

    def input_is_current(self, session: str, sequence: int) -> bool:
        """Check capture revision even when the onset event is still in transit."""
        with self._lock:
            return (
                not self._closed
                and session == self._input_session
                and sequence == self._input_sequence
            )

    def submit_reply(
        self, text: str, input_context: dict | None = None, *, wait_until_idle: bool = False
    ) -> bool:
        """Serialize freshness + nonblocking output submission with barge-in."""
        with self._lock:
            if self._closed or self._input_open:
                return False
            if wait_until_idle and self.output.speaking:
                return False
            if input_context is not None and not self.input_is_current(
                input_context["session"], input_context["sequence"]
            ):
                return False
            self.output.speak(text)
            return True

    def _input_event(self, kind, timestamp, *, sequence=None, **kwargs):
        sequence = self._input_sequence if sequence is None else sequence
        return RuntimeEvent(
            kind,
            timestamp,
            input_session=self._input_session,
            input_sequence=sequence,
            utterance_id=f"{self._input_session}:{sequence}",
            **kwargs,
        )

    def _start_recognition(self, captured: CapturedUtterance) -> None:
        future: Future[RecognizedUtterance] = Future()
        self.pending = future
        self._pending_input = captured
        event, world, sample_rate = captured.event, captured.world, captured.sample_rate

        def work():
            try:
                with self._asr_lock:
                    recognition_started = time.perf_counter()
                    transcript = self.asr.transcribe(event.audio, sample_rate)
                finished = time.perf_counter()
                with self._lock:
                    if self._closed:
                        future.cancel()
                        return
                    self._asr_diagnostics = {
                        "runtime": getattr(self.asr, "metadata", {}),
                        "audio_duration_s": len(event.audio) / sample_rate,
                        "queue_wait_s": recognition_started - captured.queued_at,
                        "inference_s": finished - recognition_started,
                        "final_latency_s": finished - event.timestamp,
                        "partial_transcripts": self.partial_transcripts,
                        "input_sequence": captured.sequence,
                        "rejected_inputs": self._rejected_inputs,
                    }
                    if not transcript.strip():
                        # VAD can fire on ambient audio. An empty recognition is
                        # not a failed device/model and must not tear down capture.
                        future.set_result(RecognizedUtterance(""))
                        return
                    self.last_transcript = transcript
                    reply_current = captured.reply_generation == self._reply_generation
                if self.responder is None or not reply_current:
                    future.set_result(RecognizedUtterance(transcript))
                    return
                reply, decision = self.responder(transcript, world)
                if not reply or len(reply) > 300:
                    raise ValueError("invalid spoken reply")
                decision.goal.validate_world(world)
                future.set_result(RecognizedUtterance(transcript, (reply, decision)))
            except Exception as exc:
                future.set_exception(exc)

        Thread(target=work, name="myumiq-conversation", daemon=True).start()

    def poll(self) -> Decision | None:
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._poll()
        finally:
            self._lock.release()

    def _poll(self) -> Decision | None:
        if self._closed:
            return None
        if self._overflow_notice and self.events.publish(self._overflow_notice, reliable=True):
            self._overflow_notice = None
        if self.pending is None or not self.pending.done():
            return None
        future, captured = self.pending, self._pending_input
        error = future.exception()
        result = None if error else future.result()
        current = captured.reply_generation == self._reply_generation
        # Production shared-memory mode commits every final input. Reply
        # freshness is checked separately by the executive using its identity.
        if self.responder is None or (result and not result.text and current):
            kind = "asr_error" if error else "utterance" if result.text else "asr_no_speech"
            text = str(error)[:200] if error else result.text or None
            finalized = self._input_event(
                kind,
                captured.event.timestamp,
                sequence=captured.sequence,
                text=text,
                source="asr",
                audio_start_at=captured.event.timestamp - captured.duration_s,
                audio_end_at=captured.event.timestamp,
            )
            if not self.events.publish(finalized, reliable=True):
                return None  # Keep the completed result until the consumer drains.
        self.pending = self._pending_input = None
        if self._queued:
            self._start_recognition(self._queued.popleft())
        if error:
            self._asr_diagnostics["last_error"] = str(error)[:200]
            if self.responder is not None and current:
                raise error
            return None
        if not current or result.response is None:
            return None
        reply, decision = result.response
        self.last_decision = decision
        self.output.speak(reply)
        self.events.publish(RuntimeEvent("conversation_decision", 0.0, decision=decision))
        return decision

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._reply_generation += 1
            self._queued.clear()
            self.pending = None
            self._pending_input = None
            self._overflow_notice = None
        self.output.stop()


class LiveVoiceLoop:
    """Capture VR audio continuously and emit 512-sample 16 kHz VAD frames."""

    def __init__(
        self,
        input_device: int | None,
        expected_device_name: str | None,
        vad: VoiceActivityDetector,
        pipeline: ConversationPipeline,
        world: Callable[[], WorldState],
        *,
        loopback_speaker_name: str | None = None,
        input_gain: float = 1.0,
    ):
        device_route = input_device is not None or expected_device_name is not None
        if device_route != (input_device is not None and bool(expected_device_name)):
            raise ValueError("invalid voice input configuration")
        if device_route == (loopback_speaker_name is not None) or not 0.1 <= input_gain <= 16:
            raise ValueError("invalid voice input configuration")
        self.input_device = input_device
        self.expected_device_name = (
            expected_device_name.casefold() if expected_device_name else None
        )
        self.loopback_speaker_name = loopback_speaker_name
        self.input_gain = input_gain
        self.vad, self.pipeline, self.world = vad, pipeline, world
        self._queue: Queue[tuple[object, float]] = Queue(maxsize=64)
        self._running = False
        self._thread: Thread | None = None
        self._capture_thread: Thread | None = None
        self._stream = None
        self._ready = Event()
        self._pending = None
        self.error: Exception | None = None
        self.diagnostics = {
            "blocks": 0,
            "vad_frames": 0,
            "peak": 0.0,
            "last_block_peak": 0.0,
            "last_block_rms": 0.0,
            "capture_gap_s": None,
            "capture_queue_wait_s": 0.0,
            "max_capture_queue_wait_s": 0.0,
            "first_audio_at": None,
            "last_audio_at": None,
            "speech_started": 0,
            "speech_ended": 0,
        }

    def start(self) -> None:
        if self.loopback_speaker_name is not None:
            self._running = True
            self._thread = Thread(
                target=self._work, args=(48000,), name="myumiq-voice", daemon=True
            )
            self._thread.start()
            self._capture_thread = Thread(
                target=self._work_loopback, name="myumiq-voice-loopback", daemon=True
            )
            self._capture_thread.start()
            if not self._ready.wait(timeout=5):
                self.stop()
                raise RuntimeError(f"voice loopback failed to start: {self.error}")
            if self.error is not None:
                self.stop()
                raise RuntimeError(f"voice loopback failed to start: {self.error}")
            return
        import sounddevice as sd

        device = sd.query_devices(self.input_device)
        name = str(device["name"])
        if self.expected_device_name not in name.casefold():
            raise RuntimeError(
                f"refusing voice input: device {self.input_device} is {name!r}, "
                f"expected {self.expected_device_name!r}"
            )
        native_rate = int(device["default_samplerate"])

        def callback(data, frames, timing, status):
            del frames, timing
            if status:
                self.error = RuntimeError(str(status))
            try:
                self._queue.put_nowait((data[:, 0].copy(), time.perf_counter()))
            except Exception:
                self.error = RuntimeError("voice capture queue overflow")

        self._running = True
        self._stream = sd.InputStream(
            device=self.input_device,
            channels=1,
            samplerate=native_rate,
            blocksize=max(1, round(native_rate * 0.032)),
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        self._thread = Thread(
            target=self._work, args=(native_rate,), name="myumiq-voice", daemon=True
        )
        self._thread.start()

    def _work_loopback(self) -> None:
        com_initialized = False
        try:
            import ctypes

            import soundcard as sc

            # COM initialization is per thread, even when soundcard was already
            # imported by another capture/playback component in this process.
            # Import first: soundcard initializes COM itself on its first import
            # and incorrectly rejects S_FALSE if we initialize it beforehand.
            ole32 = ctypes.OleDLL("ole32")
            ole32.CoInitializeEx(None, 0)
            com_initialized = True

            matches = [
                speaker
                for speaker in sc.all_speakers()
                if self.loopback_speaker_name.casefold() in speaker.name.casefold()
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one loopback speaker {self.loopback_speaker_name!r}, found {len(matches)}"
                )
            microphone = sc.get_microphone(matches[0].id, include_loopback=True)
            # SoundCard/WASAPI has a documented single-channel capture defect.
            # Capture native channels first, then mix in software; keep right-side
            # speakers audible too. The device buffer exceeds each read block.
            with microphone.recorder(samplerate=48000, channels=None, blocksize=3072) as recorder:
                self._ready.set()
                while self._running:
                    samples = recorder.record(numframes=1536).mean(axis=1)
                    try:
                        self._queue.put_nowait((samples, time.perf_counter()))
                    except Full as exc:
                        raise RuntimeError("voice capture queue overflow") from exc
        except Exception as exc:
            self.error = exc
            self._running = False
            self._ready.set()
        finally:
            if com_initialized:
                ole32.CoUninitialize()

    def _work(self, native_rate: int) -> None:
        try:
            import numpy as np

            self._pending = np.empty(0, dtype=np.float32)
            while self._running:
                try:
                    samples, timestamp = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                self._feed(samples, native_rate, timestamp)
        except Exception as exc:
            self.error = exc
            self._running = False

    def _feed(self, samples, native_rate: int, timestamp: float) -> None:
        import numpy as np

        source = np.clip(np.asarray(samples, dtype=np.float32) * self.input_gain, -1, 1)
        self.diagnostics["blocks"] += 1
        previous = self.diagnostics["last_audio_at"]
        self.diagnostics["capture_gap_s"] = (
            max(0.0, timestamp - previous) if previous is not None else None
        )
        waiting = max(0.0, time.perf_counter() - timestamp)
        self.diagnostics["capture_queue_wait_s"] = waiting
        self.diagnostics["max_capture_queue_wait_s"] = max(
            self.diagnostics["max_capture_queue_wait_s"], waiting
        )
        self.diagnostics["last_audio_at"] = timestamp
        if self.diagnostics["first_audio_at"] is None:
            self.diagnostics["first_audio_at"] = timestamp
        peak = float(np.max(np.abs(source))) if source.size else 0.0
        self.diagnostics["last_block_peak"] = peak
        self.diagnostics["last_block_rms"] = (
            float(np.sqrt(np.mean(source * source))) if source.size else 0.0
        )
        self.diagnostics["peak"] = max(self.diagnostics["peak"], peak)
        if native_rate != 16000:
            count = max(1, round(len(source) * 16000 / native_rate))
            source = np.interp(
                np.arange(count) * native_rate / 16000,
                np.arange(len(source)),
                source,
            ).astype(np.float32)
        if self._pending is None:
            self._pending = np.empty(0, dtype=np.float32)
        self._pending = np.concatenate((self._pending, source))
        while len(self._pending) >= 512:
            frame, self._pending = self._pending[:512], self._pending[512:]
            self.diagnostics["vad_frames"] += 1
            for event in self.vad.accept(
                AudioFrame(tuple(float(x) for x in frame), 16000, timestamp)
            ):
                if event.kind in ("speech_started", "speech_ended"):
                    self.diagnostics[event.kind] += 1
                self.pipeline.accept(event, self.world())

    def poll(self) -> Decision | None:
        if self.error is not None:
            raise RuntimeError(f"voice loop failed: {self.error}")
        output_error = getattr(self.pipeline.output, "error", None)
        if output_error is not None:
            raise RuntimeError(f"speech output failed: {output_error}")
        return self.pipeline.poll()

    def stop(self) -> None:
        self._running = False
        try:
            self.pipeline.close()
        finally:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            if self._thread is not None:
                self._thread.join(timeout=1)
            if self._capture_thread is not None:
                self._capture_thread.join(timeout=1)
            close_output = getattr(self.pipeline.output, "close", None)
            if callable(close_output):
                close_output()
