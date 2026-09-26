"""Input retention and obsolete-reply prevention under slow inference/backpressure."""

from threading import Event
from types import SimpleNamespace

import pytest

from myumiq_vrchat.audio import SpeechEvent
from myumiq_vrchat.body import WorldState
from myumiq_vrchat.conversation import ConversationPipeline
from myumiq_vrchat.events import EventInbox, RuntimeEvent


def feed(pipeline, sequence):
    pipeline.accept(SpeechEvent("speech_started", sequence * 2), WorldState())
    pipeline.accept(
        SpeechEvent("speech_ended", sequence * 2 + 1, (float(sequence),) * 512), WorldState()
    )


def test_later_onsets_preserve_all_completed_audio_in_capture_order():
    entered, release = Event(), Event()
    recognized, stops = [], []

    def transcribe(audio, rate):
        if audio[0] == 1:
            entered.set()
            assert release.wait(2)
        recognized.append(int(audio[0]))
        return {1: "しゃがんで", 2: "今度は立って", 3: "手も振って"}[int(audio[0])]

    events = EventInbox()
    pipeline = ConversationPipeline(
        SimpleNamespace(transcribe=transcribe),
        SimpleNamespace(speaking=True, stop=lambda: stops.append(True)),
        events,
        None,
    )
    try:
        feed(pipeline, 1)
        assert entered.wait(1)
        feed(pipeline, 2)
        feed(pipeline, 3)
        backlog = pipeline.asr_diagnostics
        assert backlog["waiting_inputs"] == 2 and backlog["pending_input_sequence"] == 1
        assert backlog["buffered_audio_s"] == pytest.approx(3 * 512 / 16000)
        assert backlog["pending_age_s"] >= 0
    finally:
        release.set()
    observed = events.drain()
    for _ in range(3):
        pipeline.pending.result(timeout=2)
        pipeline.poll()
        observed.extend(events.drain())
    finals = [event for event in observed if event.kind == "utterance"]
    assert recognized == [1, 2, 3] and len(stops) == 3
    assert [event.text for event in finals] == ["しゃがんで", "今度は立って", "手も振って"]
    assert [event.input_sequence for event in finals] == [1, 2, 3]
    assert len({event.utterance_id for event in finals}) == 3
    assert len({event.input_session for event in finals}) == 1
    assert [event.audio_end_at for event in finals] == [3, 5, 7]
    assert finals[0].audio_start_at == pytest.approx(3 - 512 / 16000)
    assert not pipeline.input_is_current(finals[0].input_session, 1)
    assert pipeline.input_is_current(finals[2].input_session, 3)
    assert pipeline.pending is None
    pipeline.close()


def test_reliable_finals_survive_provisional_traffic_and_backpressure_without_duplicates():
    events = EventInbox(max_size=2)
    first = RuntimeEvent("utterance", 1, text="既に確定した発話")
    second = RuntimeEvent("utterance", 2, text="もう一つの発話")
    assert events.publish(first, reliable=True)
    assert events.publish(second, reliable=True)
    assert not events.publish(RuntimeEvent("partial_transcript", 3, text="途中"))
    pipeline = ConversationPipeline(
        SimpleNamespace(transcribe=lambda *args: "次の発話"),
        SimpleNamespace(speaking=False, stop=lambda: None),
        events,
        None,
    )
    feed(pipeline, 1)
    completed = pipeline.pending
    completed.result(timeout=2)
    pipeline.poll()
    pipeline.poll()
    assert pipeline.pending is completed
    assert events.drain() == [first, second]
    pipeline.poll()
    pipeline.poll()
    assert [event.text for event in events.drain()] == ["次の発話"]
    assert pipeline.pending is None
    pipeline.close()


def test_reliable_input_replaces_provisional_event_without_reordering_retained_events():
    events = EventInbox(max_size=2)
    events.publish(RuntimeEvent("partial_transcript", 1))
    events.publish(RuntimeEvent("utterance", 2, text="一"), reliable=True)
    assert events.publish(RuntimeEvent("utterance", 3, text="二"), reliable=True)
    assert [event.text for event in events.drain()] == ["一", "二"]


def test_overload_is_reported_and_keeps_previously_accepted_audio():
    entered, release = Event(), Event()
    recognized = []

    def transcribe(audio, rate):
        entered.set()
        assert release.wait(2)
        recognized.append(int(audio[0]))
        return str(int(audio[0]))

    events = EventInbox()
    pipeline = ConversationPipeline(
        SimpleNamespace(transcribe=transcribe),
        SimpleNamespace(speaking=False, stop=lambda: None),
        events,
        None,
        asr_queue_size=1,
    )
    try:
        feed(pipeline, 1)
        assert entered.wait(1)
        feed(pipeline, 2)
        feed(pipeline, 3)
        assert pipeline.asr_diagnostics["rejected_inputs"] == 1
    finally:
        release.set()
    observed = events.drain()
    for _ in range(2):
        pipeline.pending.result(timeout=2)
        pipeline.poll()
        observed.extend(events.drain())
    assert recognized == [1, 2] and pipeline.pending is None
    assert [event.text for event in observed if event.kind == "utterance"] == ["1", "2"]
    overload = next(event for event in observed if event.kind == "audio_input_overflow")
    assert overload.input_sequence == 3 and "rejected_inputs=1" in overload.text
    assert pipeline.asr_diagnostics["rejected_inputs"] == 1
    pipeline.close()


def test_audio_duration_budget_rejects_oversized_input_explicitly():
    events = EventInbox()
    pipeline = ConversationPipeline(
        SimpleNamespace(transcribe=lambda *args: pytest.fail("over-budget ASR started")),
        SimpleNamespace(speaking=False, stop=lambda: None),
        events,
        None,
        asr_queue_audio_s=1,
    )
    pipeline.accept(SpeechEvent("speech_ended", 2, (0.1,) * 20), WorldState(), sample_rate=10)
    pipeline.poll()
    assert pipeline.pending is None
    assert events.drain()[-1].kind == "audio_input_overflow"
    pipeline.close()


def test_output_submission_rechecks_capture_after_executive_freshness_check():
    from myumiq_vrchat.autonomous_services import Services

    spoken = []
    events = EventInbox()
    pipeline = ConversationPipeline(
        SimpleNamespace(transcribe=lambda *args: "前の質問"),
        SimpleNamespace(speaking=False, stop=lambda: None, speak=spoken.append),
        events,
        None,
    )
    services = Services(SimpleNamespace(), {}, WorldState)
    services.voice = SimpleNamespace(pipeline=pipeline, vad=SimpleNamespace(active=False))
    feed(pipeline, 1)
    pipeline.pending.result(timeout=2)
    pipeline.poll()
    final = events.drain()[-1]
    captured = {"session": final.input_session, "sequence": final.input_sequence}
    assert pipeline.input_is_current(captured["session"], captured["sequence"])
    assert services.reply("最初の返答", 1, input_context=captured)
    # A whole newer capture arrives after the executive check and before submit.
    # VAD is already quiet again, so checking VAD alone would play the old reply.
    feed(pipeline, 2)
    assert not services.reply("古い返答の再生", 2, input_context=captured)
    assert spoken == ["最初の返答"]
    pipeline.close()


def test_failed_recognition_does_not_erase_the_next_captured_input():
    entered, release = Event(), Event()

    def transcribe(audio, rate):
        if audio[0] == 1:
            entered.set()
            assert release.wait(2)
            raise RuntimeError("recognition failed for first input")
        return "次は聞こえた"

    events = EventInbox()
    pipeline = ConversationPipeline(
        SimpleNamespace(transcribe=transcribe),
        SimpleNamespace(speaking=False, stop=lambda: None),
        events,
        None,
    )
    try:
        feed(pipeline, 1)
        assert entered.wait(1)
        feed(pipeline, 2)
    finally:
        release.set()
    assert pipeline.pending.exception(timeout=2)
    pipeline.poll()
    pipeline.pending.result(timeout=2)
    pipeline.poll()
    finals = [event for event in events.drain() if event.kind in ("utterance", "asr_error")]
    assert [event.kind for event in finals] == ["asr_error", "utterance"]
    assert [event.input_sequence for event in finals] == [1, 2]
    assert finals[1].text == "次は聞こえた"
    pipeline.close()
