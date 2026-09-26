import os
import time

from myumiq_vrchat.process_sapi import ProcessSapiSpeechOutput


def audio_worker_fixture(connection, settings):
    """Simulate a native, GIL-holding audio stop inside the real child process."""
    connection.send(("ready",))
    generation, speaking = 0, False
    while True:
        if connection.poll(0.01):
            message = connection.recv()
            if message[0] == "close":
                break
            _, generation, text = message
            if text == "crash":
                os._exit(7)
            if text is None:
                if os.name == "nt":
                    import ctypes

                    ctypes.PyDLL("kernel32").Sleep(700)
                else:
                    time.sleep(0.7)
            speaking = text is not None
        connection.send(("status", generation, speaking, None))
    connection.close()


def wait_until(predicate):
    deadline = time.monotonic() + 4
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


def test_native_stop_stall_does_not_block_parent_or_revive_cancelled_speech(monkeypatch):
    monkeypatch.setattr("myumiq_vrchat.process_sapi._sapi_worker", audio_worker_fixture)
    output = ProcessSapiSpeechOutput(0, expected_device_name="fixture")
    try:
        assert output._process.pid != os.getpid()
        output.speak("first")
        wait_until(lambda: output._pending is None and output.diagnostics.get("last_health_at"))
        started = time.perf_counter()
        output.stop()
        assert time.perf_counter() - started < 0.05
        assert not output.speaking
        ticks = []
        for _ in range(50):
            ticks.append(time.perf_counter())
            time.sleep(0.01)
        assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.2
        output.speak("latest")
        wait_until(lambda: output._pending is None)
        assert output.speaking and output.error is None
    finally:
        output.close()
    assert not output._process.is_alive()


def test_child_crash_is_reported_without_crashing_the_parent(monkeypatch):
    monkeypatch.setattr("myumiq_vrchat.process_sapi._sapi_worker", audio_worker_fixture)
    output = ProcessSapiSpeechOutput(0, expected_device_name="fixture")
    try:
        output.speak("crash")
        wait_until(lambda: output.error is not None)
        assert not output.speaking
    finally:
        output.close()
    assert output._process.exitcode == 7
