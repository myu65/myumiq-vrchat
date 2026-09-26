import time

import pytest
import sounddevice

from myumiq_vrchat.audio import SpeechEvent
from myumiq_vrchat.body import WorldObject, WorldState
from myumiq_vrchat.cognition import Decision, Goal
from myumiq_vrchat.conversation import ConversationPipeline, LiveVoiceLoop, VoiceConfig
from myumiq_vrchat.events import EventInbox
from myumiq_vrchat.tts import SapiSpeechOutput


@pytest.mark.parametrize(
    "result, releases, plays", [(0, 1, True), (1, 1, True), (-2147417850, 0, True), (-1, 0, False)]
)
def test_audio_worker_com_ownership_and_failure(monkeypatch, result, releases, plays):
    import ctypes
    from types import SimpleNamespace

    import myumiq_vrchat.tts as module

    calls = []
    api = SimpleNamespace(
        CoInitializeEx=lambda *_: result, CoUninitialize=lambda: calls.append("release")
    )
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda _: api, raising=False)
    output = SapiSpeechOutput(12, expected_device_name="CABLE Input")

    def fail_playback(_, cancel):
        assert cancel is output._cancel
        calls.append("play")
        raise RuntimeError("playback failed")

    monkeypatch.setattr(output, "_synthesize_and_play", fail_playback)
    output._work("test", output._cancel)
    assert calls.count("release") == releases
    assert ("play" in calls) == plays
    assert output.error is not None


def test_conversation_barges_in_then_runs_asr_and_reply_off_thread():
    class ASR:
        def transcribe(self, audio, sample_rate):
            assert sample_rate == 16000 and audio
            return "こんにちは"

    class Output:
        speaking = True
        stopped = False
        spoken = []

        def stop(self):
            self.stopped, self.speaking = True, False

        def speak(self, text):
            self.spoken.append(text)

    output, events = Output(), EventInbox()
    pipeline = ConversationPipeline(
        ASR(),
        output,
        events,
        lambda text, world: (
            "こんにちは",
            Decision(goal=Goal(skill="WAVE", hand="right", duration_s=1), source="local_llm"),
        ),
    )
    pipeline.accept(SpeechEvent("speech_started", 1), WorldState())
    assert output.stopped and events.drain()[0].kind == "speech_started"
    pipeline.accept(SpeechEvent("speech_ended", 2, (0.1,) * 512), WorldState())
    for _ in range(100):
        decision = pipeline.poll()
        if decision:
            break
        time.sleep(0.001)
    assert decision.goal.skill == "WAVE"
    assert output.spoken == ["こんにちは"]
    delivered = events.drain()
    assert delivered[0].kind == "speech_ended"
    assert delivered[1].kind == "conversation_decision"
    assert delivered[1].decision == decision


def test_speech_onset_targets_nearest_visible_player():
    class ASR:
        def transcribe(self, audio, sample_rate):
            return ""

    class Output:
        speaking = False

        def stop(self):
            pass

        def speak(self, text):
            pass

    events = EventInbox()
    pipeline = ConversationPipeline(ASR(), Output(), events, lambda text, world: None)
    world = WorldState(
        objects=(
            WorldObject(name="far", position=(4.0, 0.0, 1.6), source="vision", kind="player"),
            WorldObject(name="near", position=(1.0, 0.0, 1.6), source="vision", kind="player"),
        )
    )
    pipeline.accept(SpeechEvent("speech_started", 1), world)
    assert events.drain()[0].target == "near"


def test_live_voice_refuses_device_index_drift(monkeypatch):
    monkeypatch.setattr(
        sounddevice,
        "query_devices",
        lambda index: {"name": "Laptop microphone", "default_samplerate": 48000},
    )
    loop = LiveVoiceLoop(3, "VRChat capture", object(), object(), WorldState)
    with pytest.raises(RuntimeError, match="refusing voice input"):
        loop.start()


def test_tts_refuses_device_index_drift_before_synthesis(monkeypatch):
    monkeypatch.setattr(
        sounddevice,
        "query_devices",
        lambda index: {"name": "Speakers", "default_samplerate": 48000},
    )
    output = SapiSpeechOutput(2, expected_device_name="CABLE Input")
    with pytest.raises(RuntimeError, match="refusing TTS route"):
        output._synthesize_and_play("こんにちは", output._cancel)


def test_voice_config_requires_exactly_one_capture_route(tmp_path):
    common = {
        "output_device": 1,
        "output_device_name": "CABLE Input",
        "silero_model": tmp_path / "silero.onnx",
        "whisper_model": "small",
    }
    assert VoiceConfig(**common, loopback_speaker_name="FS2434").input_device is None
    assert VoiceConfig(**common, input_device=2, input_device_name="Mic").input_device == 2
    with pytest.raises(ValueError, match="exactly one"):
        VoiceConfig(**common)
    with pytest.raises(ValueError, match="exactly one"):
        VoiceConfig(
            **common,
            input_device=2,
            input_device_name="Mic",
            loopback_speaker_name="FS2434",
        )


def test_voice_diagnostics_distinguish_silence_and_speech():
    import numpy as np

    class Detector:
        def accept(self, frame):
            return (
                [SpeechEvent("speech_started", frame.timestamp)] if max(frame.samples) > 0 else []
            )

    class Pipeline:
        def accept(self, event, world):
            pass

    loop = LiveVoiceLoop(0, "test", Detector(), Pipeline(), WorldState, input_gain=2)
    loop._feed(np.zeros(1536, dtype=np.float32), 48000, 10.0)
    assert loop.diagnostics["vad_frames"] == 1
    assert loop.diagnostics["peak"] == 0
    assert loop.diagnostics["speech_started"] == 0
    loop._feed(np.full(1536, 0.25, dtype=np.float32), 48000, 11.0)
    assert loop.diagnostics["blocks"] == 2
    assert loop.diagnostics["vad_frames"] == 2
    assert loop.diagnostics["peak"] == 0.5
    assert loop.diagnostics["speech_started"] == 1
    assert loop.diagnostics["first_audio_at"] == 10.0
    assert loop.diagnostics["last_audio_at"] == 11.0


def test_loopback_initializes_and_releases_thread_com_on_failure(monkeypatch):
    import ctypes
    import sys
    from types import SimpleNamespace

    calls = []
    com = SimpleNamespace(
        CoInitializeEx=lambda *args: calls.append("initialize"),
        CoUninitialize=lambda: calls.append("release"),
    )
    monkeypatch.setattr(ctypes, "OleDLL", lambda name: com, raising=False)
    monkeypatch.setitem(sys.modules, "soundcard", SimpleNamespace(all_speakers=lambda: []))
    loop = LiveVoiceLoop(
        None, None, object(), object(), WorldState, loopback_speaker_name="missing"
    )
    loop._work_loopback()
    assert calls == ["initialize", "release"]
    assert isinstance(loop.error, RuntimeError)
    assert loop._ready.is_set()


def test_loopback_captures_native_channels_before_mono_mix(monkeypatch):
    import ctypes
    import sys
    from types import SimpleNamespace

    import numpy as np

    com = SimpleNamespace(CoInitializeEx=lambda *args: None, CoUninitialize=lambda: None)
    monkeypatch.setattr(ctypes, "OleDLL", lambda name: com, raising=False)
    loop = LiveVoiceLoop(
        None, None, object(), object(), WorldState, loopback_speaker_name="test output"
    )
    loop._running = True

    class Recorder:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def record(self, numframes):
            loop._running = False
            # Preserve right-only speech and the sample count/timebase.
            return np.tile([0.0, 0.6], (numframes, 1)).astype(np.float32)

    def recorder(**kwargs):
        assert kwargs["channels"] is None
        assert kwargs["blocksize"] > 1536
        return Recorder()

    monkeypatch.setitem(
        sys.modules,
        "soundcard",
        SimpleNamespace(
            all_speakers=lambda: [SimpleNamespace(id="fixture", name="test output")],
            get_microphone=lambda *args, **kwargs: SimpleNamespace(recorder=recorder),
        ),
    )
    loop._work_loopback()
    assert loop.error is None
    samples, _ = loop._queue.get_nowait()
    assert samples.shape == (1536,)
    np.testing.assert_allclose(samples, 0.3)


def test_barge_in_discards_pending_reply_and_handles_latest_utterance():
    from threading import Event

    entered, release = Event(), Event()

    class ASR:
        def transcribe(self, audio, sample_rate):
            return "最初" if audio[0] == 1 else "最新"

    class Output:
        speaking = False
        spoken = []

        def stop(self):
            pass

        def speak(self, text):
            self.spoken.append(text)

    def respond(text, world):
        if text == "最初":
            entered.set()
            assert release.wait(2)
        return text, Decision(goal=Goal(skill="WAIT", duration_s=1), source="fixed")

    output = Output()
    pipeline = ConversationPipeline(ASR(), output, EventInbox(), respond)
    pipeline.accept(SpeechEvent("speech_started", 1), WorldState())
    pipeline.accept(SpeechEvent("speech_ended", 2, (1.0,)), WorldState())
    assert entered.wait(2)
    old = pipeline.pending
    pipeline.accept(SpeechEvent("speech_started", 3), WorldState())
    pipeline.accept(SpeechEvent("speech_ended", 4, (2.0,)), WorldState())
    release.set()
    old.result(timeout=2)
    assert pipeline.poll() is None
    pipeline.pending.result(timeout=2)
    assert pipeline.poll() is not None
    assert output.spoken == ["最新"]
