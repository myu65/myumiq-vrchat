import sys
from types import SimpleNamespace

import numpy as np
import pytest

from myumiq_vrchat.body import WorldState
from myumiq_vrchat.vision import CaptureUnavailable, LiveVisionLoop


def test_occlusion_invalidates_image_and_recovers_with_same_detector(monkeypatch):
    import myumiq_vrchat.vision as module

    calls = []

    class Capture:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def grab(self, rect):
            calls.append("capture")
            return np.zeros((20, 20, 4), dtype=np.uint8)

    class Detector:
        def detect(self, image, timestamp):
            calls.append("detect")
            loop._running = False
            return []

    detector = Detector()
    loop = LiveVisionLoop(detector)
    loop._world = WorldState(timestamp=1.0)
    loop._decision_image = "stale"
    attempts = []

    def rect(title):
        attempts.append(title)
        if len(attempts) == 1:
            raise CaptureUnavailable("not foreground")
        return {}

    def sleep(_):
        if len(attempts) == 1:
            assert loop.world() == WorldState()
            assert loop.capture_pending == "not foreground"
            with pytest.raises(RuntimeError, match="no decision image"):
                loop.decision_snapshot()

    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=Capture))
    monkeypatch.setattr(module, "vrchat_client_rect", rect)
    monkeypatch.setattr(module.time, "sleep", sleep)
    loop._running = True
    loop._work()
    assert loop.error is None
    assert loop.capture_pending is None
    assert loop.detector is detector
    assert loop.frames == 1
    assert calls == ["capture", "detect"]
    assert loop.decision_snapshot()[2] is not None


def test_decision_image_is_bounded_and_snapshot_does_no_encoding(monkeypatch):
    import base64

    import cv2

    from myumiq_vrchat.vision import decision_image

    frame = np.random.default_rng(10).integers(0, 256, (1200, 1920, 3), dtype=np.uint8)
    encoded = decision_image(frame)
    assert len(encoded) <= 80000
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), 1)
    assert max(decoded.shape[:2]) <= 640
    loop = LiveVisionLoop(None)
    loop._world = WorldState(timestamp=4.0)
    loop._decision_image = encoded
    monkeypatch.setattr(cv2, "imencode", lambda *args: pytest.fail("encoding on caller thread"))
    assert loop.decision_snapshot() == (loop._world, encoded, 4.0)


def test_services_keep_pending_capture_without_restart():
    from myumiq_vrchat.autonomous_services import Services

    vision = SimpleNamespace(world=lambda: WorldState(), frames=2, capture_pending="not foreground")
    health = {}
    services = Services(SimpleNamespace(vision=True, voice=None), health, lambda: WorldState())
    services.vision = vision
    world, _ = services.poll(10.0)
    assert world == WorldState()
    assert services.vision is vision
    assert health["vision"]["state"] == "pending"
    assert services.retry["vision"] == 0.0
