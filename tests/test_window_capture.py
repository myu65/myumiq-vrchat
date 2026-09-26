from types import SimpleNamespace

import numpy as np
import pytest

from myumiq_vrchat.vision import CaptureUnavailable
from myumiq_vrchat.window_capture import (
    LatestWindowFrame,
    NativeWindowCapture,
    WindowCapture,
    crop_client,
)


def test_borrowed_pixels_are_copied_and_capture_time_is_not_renewed():
    latest = LatestWindowFrame(2)
    frame = np.full((240, 320, 4), 73, dtype=np.uint8)
    latest.accept(frame, 10.0, 10.01)
    frame.fill(0)
    image, captured_at = latest.read(10.8)
    assert captured_at == 10.0
    assert np.all(image == 73)
    latest.accept(frame, 10.0, 10.9)
    with pytest.raises(CaptureUnavailable):
        latest.read(11.01)
    latest.accept(frame, 11.1, 11.2)
    assert latest.read(11.3)[1] == 11.1


def test_stale_future_and_closed_frames_do_not_become_observations():
    latest = LatestWindowFrame(2)
    frame = np.zeros((240, 320, 4), dtype=np.uint8)
    for capture in (8.0, 12.0):
        latest.accept(frame, capture, 10.0)
        with pytest.raises(CaptureUnavailable):
            latest.read(10.0)
    latest.accept(frame, 10.0, 10.0)
    latest.close()
    latest.accept(frame, 11.0, 11.0)
    with pytest.raises(CaptureUnavailable):
        latest.read(11.0)


def test_small_presentation_lead_is_bounded_without_restamping_duplicate():
    latest = LatestWindowFrame(2)
    frame = np.zeros((240, 320, 4), dtype=np.uint8)
    latest.accept(frame, 10.012, 10.0)
    assert latest.read(10.0)[1] == 10.0
    latest.accept(frame, 10.012, 10.8)
    assert latest.read(10.8)[1] == 10.0


def test_client_crop_excludes_titlebar_and_rejects_resize_mismatch():
    frame = np.ones((280, 340, 3), dtype=np.uint8)
    frame[:30] = 70
    client = crop_client(frame, (7, 0, 347, 280), (8, 30), (338, 249))
    assert client.shape == (249, 338, 3) and np.all(client == 1)
    with pytest.raises(CaptureUnavailable, match="geometry changed"):
        crop_client(frame, (7, 0, 387, 280), (8, 30), (378, 249))


def test_replaced_window_restarts_capture_but_minimization_only_pauses(monkeypatch):
    import myumiq_vrchat.window_capture as module

    source = NativeWindowCapture.__new__(NativeWindowCapture)
    source.handle, source.latest = 7, LatestWindowFrame(2)
    api = SimpleNamespace(IsWindow=lambda _: False, IsIconic=lambda _: False)
    monkeypatch.setattr(module.ctypes, "windll", SimpleNamespace(user32=api), raising=False)
    with pytest.raises(RuntimeError, match="must be reopened"):
        source.read()
    api.IsWindow, api.IsIconic = lambda _: True, lambda _: True
    with pytest.raises(CaptureUnavailable, match="minimized"):
        source.read()


def test_native_stall_has_one_request_deadline_and_terminates_only_capture_child(monkeypatch):
    import myumiq_vrchat.window_capture as module

    source = WindowCapture.__new__(WindowCapture)
    calls, clock = [], [10.0]
    source.pending_since, source.latest, source.closed, source.max_age = None, None, False, 1.0
    source.connection = SimpleNamespace(
        poll=lambda timeout: calls.append(("poll", timeout)) or False,
        send=lambda msg: calls.append(("send", msg)),
        close=lambda: calls.append(("close_pipe",)),
    )
    alive = [True]

    def terminate():
        calls.append(("terminate_owned_child",))
        alive[0] = False

    source.process = SimpleNamespace(
        is_alive=lambda: alive[0],
        join=lambda t: calls.append(("join", t)),
        terminate=terminate,
        close=lambda: calls.append(("close_handle",)),
    )
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    for _ in range(5):
        with pytest.raises(CaptureUnavailable):
            source.read()
    assert calls.count(("send", "frame")) == 1
    assert all(row[1] <= 0.03 for row in calls if row[0] == "poll")
    clock[0] = 20.1
    with pytest.raises(RuntimeError, match="deadline expired"):
        source.read()
    source.close()
    source.close()
    assert calls.count(("terminate_owned_child",)) == 1
