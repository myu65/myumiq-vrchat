import sys
import threading
import time
from types import SimpleNamespace

from myumiq_vrchat.tts import SapiSpeechOutput


def test_later_reply_cannot_uncancel_old_worker_or_receive_its_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(stop=lambda: None))
    output = SapiSpeechOutput(0, expected_device_name="fixture")
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    cancellations = {}

    def synthesize(text, cancel):
        cancellations[text] = cancel
        if text == "old":
            entered.set()
            assert release.wait(4)
            finished.set()
            raise RuntimeError("stale worker failed")

    monkeypatch.setattr(output, "_synthesize_and_play", synthesize)
    try:
        output.speak("old")
        assert entered.wait(2)
        old_thread = output._thread
        output.speak("new")
        output._thread.join(2)
        assert cancellations["old"].is_set()
        assert not cancellations["new"].is_set()
        release.set()
        old_thread.join(2)
        assert finished.is_set() and output.error is None
    finally:
        release.set()
        output.stop()


def test_failed_synthesis_is_reported_instead_of_silent_success(monkeypatch):
    import myumiq_vrchat.tts as module

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            stop=lambda: None,
            query_devices=lambda _: {"name": "fixture", "default_samplerate": 48000},
        ),
    )
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(wait=lambda: 7)
    )
    output = SapiSpeechOutput(0, expected_device_name="fixture")
    output.speak("failed speech")
    output._thread.join(3)
    assert isinstance(output.error, RuntimeError) and "exit code 7" in str(output.error)
    assert not output.speaking and output._process is None


def test_cancel_is_nonblocking_and_only_playback_worker_closes_stream(monkeypatch):
    import numpy as np
    import pytest

    opened = threading.Event()
    calls = []
    callbacks = {}

    class Abort(Exception):
        pass

    class Stream:
        def __init__(self, **kwargs):
            callbacks.update(kwargs)

        def __enter__(self):
            calls.append(("open", threading.get_ident()))
            opened.set()
            return self

        def abort(self):
            calls.append(("abort", threading.get_ident()))

        def __exit__(self, *args):
            calls.append(("close", threading.get_ident()))

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(OutputStream=Stream, CallbackAbort=Abort, CallbackStop=Abort),
    )
    output = SapiSpeechOutput(0, expected_device_name="fixture")
    cancel = output._cancel
    worker = threading.Thread(target=output._play_pcm, args=(np.ones(10000), 48000, cancel))
    worker.start()
    try:
        assert opened.wait(2)
        began = time.perf_counter()
        output.stop()
        assert time.perf_counter() - began < 0.1
        block = np.ones((100, 1))
        with pytest.raises(Abort):
            callbacks["callback"](block, 100, None, None)
        assert not block.any()
        worker.join(2)
        assert not worker.is_alive()
        assert [name for name, _ in calls] == ["open", "abort", "close"]
        assert {owner for _, owner in calls} == {worker.ident}
    finally:
        cancel.set()
        worker.join(2)
