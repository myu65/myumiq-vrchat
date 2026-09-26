import time

import numpy as np

from myumiq_vrchat.streaming_tts import _LeadingSilence, _PlaybackBuffer


def test_slow_producer_rebuffers_without_dropping_or_repeating_pcm():
    buffer = _PlaybackBuffer(20, time.perf_counter())
    output = np.empty((4, 1), dtype=np.float32)
    buffer.append(np.arange(1, 11, dtype=np.float32))
    delivered = []
    for _ in range(3):
        assert not buffer.read(output, False)
        delivered.extend(output[:, 0])
    assert delivered == list(range(1, 11)) + [0, 0]
    assert buffer.diagnostics["underflows"] == 1
    first = buffer.diagnostics["first_audio_s"]
    # A tiny arriving chunk must wait; otherwise each arrival chops the voice.
    buffer.append(np.array([11, 12], dtype=np.float32))
    for _ in range(5):
        assert not buffer.read(output, False)
        np.testing.assert_array_equal(output, 0)
    assert buffer.queued == 2 and buffer.diagnostics["underflows"] == 1
    buffer.append(np.arange(13, 21, dtype=np.float32))
    resumed = []
    for _ in range(3):
        drained = buffer.read(output, True)
        resumed.extend(output[:, 0])
    assert drained and resumed == list(range(11, 21)) + [0, 0]
    assert buffer.diagnostics["first_audio_s"] == first


def test_short_final_packet_drains_and_empty_synthesis_never_claims_audio():
    buffer = _PlaybackBuffer(48000, time.perf_counter())
    output = np.empty((4, 1), dtype=np.float32)
    assert buffer.read(output, True)
    assert buffer.diagnostics["first_audio_s"] is None
    buffer.append(np.array([0.1, 0.2], dtype=np.float32))
    assert buffer.read(output, True)
    np.testing.assert_allclose(output[:, 0], [0.1, 0.2, 0, 0])
    assert buffer.diagnostics["underflows"] == 0


def test_trim_preserves_preroll_and_all_speech_across_chunk_boundaries():
    onset = _LeadingSilence(1000, 0.6)
    silence = np.zeros(400, dtype=np.float32)
    speech = np.linspace(0.01, 0.5, 500, dtype=np.float32)
    assert len(onset.accept(silence[:320])) == 0
    actual = onset.accept(np.concatenate((silence[320:], speech)))
    assert onset.trimmed == 340
    np.testing.assert_array_equal(actual, np.concatenate((silence[:60], speech)))
    # Mid-sentence pauses and quiet endings must remain byte-for-byte identical.
    np.testing.assert_array_equal(onset.accept(silence), silence)


def test_trim_is_bounded_and_short_quiet_output_finishes_without_hanging():
    onset = _LeadingSilence(1000, 0.6)
    quiet = np.full(800, 0.0001, dtype=np.float32)
    np.testing.assert_array_equal(onset.accept(quiet), quiet[600:])
    assert onset.trimmed == 600
    short = _LeadingSilence(1000, 0.6)
    assert len(short.accept(quiet[:100])) == 0
    np.testing.assert_array_equal(short.accept(quiet[:0], finished=True), quiet[:60])
    disabled = _LeadingSilence(1000, 0.0)
    assert disabled.accept(quiet) is quiet


def test_trim_keeps_weak_speech_onset_with_sixty_ms_context():
    onset = _LeadingSilence(1000, 0.6)
    source = np.concatenate((np.zeros(300), np.full(40, 0.0008), np.full(60, 0.01)))
    result = onset.accept(source)
    np.testing.assert_array_equal(result, source[280:])
