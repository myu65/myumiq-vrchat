import ctypes
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from myumiq_vrchat.audio import SpeechEvent
from myumiq_vrchat.body import WorldState
from myumiq_vrchat.conversation import ConversationPipeline, LiveVoiceLoop
from myumiq_vrchat.events import EventInbox


def test_capture_health_distinguishes_silence_and_vad_queue_delay(monkeypatch):
    loop = LiveVoiceLoop(
        1, "input", SimpleNamespace(accept=lambda frame: []), SimpleNamespace(), WorldState
    )
    monkeypatch.setattr("myumiq_vrchat.conversation.time.perf_counter", lambda: 10.2)
    loop._feed(np.ones(512, dtype=np.float32) * 0.5, 16000, 10.0)
    assert loop.diagnostics["last_block_peak"] == 0.5
    assert loop.diagnostics["last_block_rms"] == 0.5
    assert loop.diagnostics["capture_queue_wait_s"] == pytest.approx(0.2)
    assert loop.diagnostics["capture_gap_s"] is None
    monkeypatch.setattr("myumiq_vrchat.conversation.time.perf_counter", lambda: 10.533)
    loop._feed(np.zeros(512, dtype=np.float32), 16000, 10.532)
    assert loop.diagnostics["peak"] == 0.5
    assert loop.diagnostics["last_block_peak"] == 0
    assert loop.diagnostics["last_block_rms"] == 0
    assert loop.diagnostics["capture_queue_wait_s"] == pytest.approx(0.001)
    assert loop.diagnostics["max_capture_queue_wait_s"] == pytest.approx(0.2)
    assert loop.diagnostics["capture_gap_s"] == pytest.approx(0.532)


def test_onset_and_active_do_not_wait_for_partial_asr_and_stale_partial_is_dropped():
    entered, release = threading.Event(), threading.Event()

    class ASR:
        def transcribe(self, audio, sample_rate):
            entered.set()
            assert release.wait(2)
            return "古い途中の文"

    class Output:
        speaking = True
        stopped = False

        def stop(self):
            self.stopped = True

    output, events = Output(), EventInbox()
    pipeline = ConversationPipeline(ASR(), output, events, None)
    pipeline.accept(SpeechEvent("speech_started", 1), WorldState())
    pipeline.accept(SpeechEvent("speech_active", 2, (0.1,) * 512), WorldState())
    assert entered.wait(1)
    pipeline.accept(SpeechEvent("speech_started", 3), WorldState())
    assert output.stopped
    assert [e.kind for e in events.drain()] == ["speech_started", "speech_active", "speech_started"]
    release.set()
    end = time.monotonic() + 2
    while pipeline._partial_busy and time.monotonic() < end:
        time.sleep(0.001)
    assert not pipeline._partial_busy
    assert not events.drain()


def test_final_only_asr_preserves_immediate_onset_and_avoids_partial_work():
    calls = []
    output = SimpleNamespace(speaking=True, stop=lambda: calls.append("stop"))

    def recognize(audio, sample_rate):
        calls.append("asr")
        return "立って"

    events = EventInbox()
    pipeline = ConversationPipeline(
        SimpleNamespace(transcribe=recognize), output, events, None, partial_transcripts=False
    )
    now = time.perf_counter()
    pipeline.accept(SpeechEvent("speech_started", now), WorldState())
    pipeline.accept(SpeechEvent("speech_active", now, (0.1,) * 512), WorldState())
    assert calls == ["stop"]
    assert [e.kind for e in events.drain()] == ["speech_started", "speech_active"]
    pipeline.accept(SpeechEvent("speech_ended", now, (0.1,) * 512), WorldState())
    pipeline.pending.result(timeout=2)
    pipeline.poll()
    assert calls == ["stop", "asr"]
    assert events.drain()[-1].text == "立って"
    assert pipeline.asr_diagnostics["partial_transcripts"] is False
    assert pipeline.asr_diagnostics["audio_duration_s"] == 512 / 16000
    assert pipeline.asr_diagnostics["queue_wait_s"] >= 0


def test_partial_is_provisional_and_final_still_publishes_utterance():
    class ASR:
        def transcribe(self, audio, sample_rate):
            return "こんにちは"

    class Output:
        speaking = False

        def stop(self):
            pass

    events = EventInbox()
    pipeline = ConversationPipeline(ASR(), Output(), events, None)
    pipeline.accept(SpeechEvent("speech_active", 1, (0.1,) * 512), WorldState())
    end = time.monotonic() + 2
    while pipeline._partial_busy and time.monotonic() < end:
        time.sleep(0.001)
    provisional = events.drain()
    assert provisional[-1].kind == "partial_transcript"
    assert provisional[-1].source == "asr_chunked_provisional"
    pipeline.accept(SpeechEvent("speech_ended", 2, (0.1,) * 512), WorldState())
    pipeline.pending.result(timeout=2)
    pipeline.poll()
    assert [e.kind for e in events.drain()] == ["speech_ended", "utterance"]


def test_loopback_keeps_capturing_while_vad_worker_is_busy(monkeypatch):
    entered, release, captured = threading.Event(), threading.Event(), threading.Event()
    reads = []

    class Recorder:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def record(self, numframes):
            if reads:
                assert entered.wait(2)
            reads.append(1)
            if len(reads) == 4:
                captured.set()
                assert release.wait(2)
                loop._running = False
            return np.ones((numframes, 1), dtype=np.float32) * 0.1

    microphone = SimpleNamespace(recorder=lambda **kwargs: Recorder())
    monkeypatch.setitem(
        sys.modules,
        "soundcard",
        SimpleNamespace(
            all_speakers=lambda: [SimpleNamespace(name="test-output", id="device")],
            get_microphone=lambda *args, **kwargs: microphone,
        ),
    )
    monkeypatch.setattr(
        ctypes,
        "OleDLL",
        lambda *args: SimpleNamespace(CoInitializeEx=lambda *args: 0, CoUninitialize=lambda: None),
        raising=False,
    )

    class VAD:
        def accept(self, frame):
            entered.set()
            assert release.wait(2)
            return []

    closed = []
    pipeline = SimpleNamespace(
        close=lambda: None, output=SimpleNamespace(close=lambda: closed.append(True))
    )
    loop = LiveVoiceLoop(
        None, None, VAD(), pipeline, WorldState, loopback_speaker_name="test-output"
    )
    try:
        loop.start()
        assert entered.wait(1)
        assert captured.wait(1), "device capture waited for VAD/event processing"
        assert loop._queue.qsize() == 2 and loop.error is None
    finally:
        release.set()
        loop.stop()
    assert closed == [True]
    assert not loop._capture_thread.is_alive() and not loop._thread.is_alive()


def test_empty_asr_keeps_pipeline_available_for_the_next_utterance():
    answers = iter(["   ", "今度は立って"])
    asr = SimpleNamespace(transcribe=lambda *args: next(answers))
    output = SimpleNamespace(speaking=False, stop=lambda: None)
    events = EventInbox()
    pipeline = ConversationPipeline(asr, output, events, None)
    for index in range(2):
        pipeline.accept(SpeechEvent("speech_started", index * 2), WorldState())
        pipeline.accept(SpeechEvent("speech_ended", index * 2 + 1, (0.1,) * 512), WorldState())
        pipeline.pending.result(timeout=2)
        assert pipeline.poll() is None
        observed = events.drain()
        if index == 0:
            assert [e.kind for e in observed] == ["speech_started", "speech_ended", "asr_no_speech"]
            assert pipeline.last_transcript is None
        else:
            assert observed[-1].kind == "utterance" and observed[-1].text == "今度は立って"


def test_empty_asr_does_not_invoke_inline_dialogue():
    calls = []
    events = EventInbox()
    pipeline = ConversationPipeline(
        SimpleNamespace(transcribe=lambda *args: ""),
        SimpleNamespace(speaking=False, stop=lambda: None),
        events,
        lambda *args: calls.append(args),
    )
    pipeline.accept(SpeechEvent("speech_ended", 1, (0.1,) * 512), WorldState())
    pipeline.pending.result(timeout=2)
    pipeline.poll()
    assert not calls and events.drain()[-1].kind == "asr_no_speech"


@pytest.mark.parametrize("kind", ["speech_active", "speech_ended"])
@pytest.mark.parametrize("fails", [False, True])
def test_voice_shutdown_discards_late_recognition_and_rejects_new_input(kind, fails):
    entered, release = threading.Event(), threading.Event()
    calls, stops = [], []

    def transcribe(*args):
        calls.append(args)
        entered.set()
        assert release.wait(2)
        if fails:
            raise RuntimeError("old session recognition failed")
        return "前の音声"

    events = EventInbox()
    output = SimpleNamespace(speaking=False, stop=lambda: stops.append(True))
    pipeline = ConversationPipeline(SimpleNamespace(transcribe=transcribe), output, events, None)
    loop = LiveVoiceLoop(0, "fixture", object(), pipeline, WorldState)
    pipeline.accept(SpeechEvent(kind, 1, (0.1,) * 512), WorldState())
    try:
        assert entered.wait(1)
        pending = pipeline.pending
        events.drain()
        loop.stop()
        loop.stop()
        pipeline.accept(SpeechEvent("speech_started", 2), WorldState())
        pipeline.accept(SpeechEvent("speech_ended", 3, (0.1,) * 512), WorldState())
    finally:
        release.set()
    end = time.monotonic() + 2
    while (pipeline._partial_busy or (pending and not pending.done())) and time.monotonic() < end:
        time.sleep(0.001)
    assert not pipeline._partial_busy and (pending is None or pending.done())
    assert pipeline.poll() is None and pipeline.pending is None
    assert not events.drain() and len(calls) == 1 and stops == [True]


def test_voice_shutdown_discards_a_completed_reply_before_playback():
    from myumiq_vrchat.cognition import Decision, Goal

    spoken = []
    output = SimpleNamespace(speaking=False, speak=spoken.append, stop=lambda: None)
    events = EventInbox()
    pipeline = ConversationPipeline(
        SimpleNamespace(transcribe=lambda *args: "前の依頼"),
        output,
        events,
        lambda *args: ("前の返答", Decision(goal=Goal(skill="WAIT", duration_s=1), source="fixed")),
    )
    pipeline.accept(SpeechEvent("speech_ended", 1, (0.1,) * 512), WorldState())
    pipeline.pending.result(timeout=2)
    events.drain()
    pipeline.close()
    assert pipeline.poll() is None and not spoken and not events.drain()
