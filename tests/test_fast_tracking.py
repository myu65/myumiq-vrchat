from concurrent.futures import Future

import cv2
import numpy as np
import pytest

from myumiq_vrchat.fast_tracking import FastVisualTracker
from myumiq_vrchat.perception import SpatialDetection
from myumiq_vrchat.vision import LiveVisionLoop
from myumiq_vrchat.vision_multirate import PerceptionUpdates


def scene(dx=0):
    image = np.zeros((160, 240, 3), np.uint8)
    patch = np.random.default_rng(31).integers(0, 256, (60, 60, 3), np.uint8)
    image[50:110, 90 + dx : 150 + dx] = patch
    return image


def target():
    return SpatialDetection(
        "person-1", (2, 0, 1.6), 0.95, "player", (0, 0), (0.375, 0.3125, 0.625, 0.6875)
    )


def test_late_detection_replays_real_frames_and_lost_target_is_removed():
    tracker = FastVisualTracker()
    tracker.update(scene(), 1)
    tracker.update(scene(3), 1.05)
    tracker.update(scene(6), 1.1)
    assert tracker.detected([target()], 1)
    detection = tracker.detections(1.1)[0]
    assert detection.track_id == "person-1"
    assert detection.image_position[0] == pytest.approx(0.05, abs=0.007)
    assert not tracker.update(scene(15), 1.1)  # repeated frame is not fresh
    tracker.update(np.zeros_like(scene()), 1.15)
    assert not tracker.detections(1.15)


def test_late_or_ambiguous_history_cannot_resurrect_old_identity():
    tracker = FastVisualTracker()
    tracker.update(scene(), 1)
    tracker.reset()
    tracker.update(scene(6), 2)
    assert not tracker.detected([target()], 1)
    assert tracker.detections(2) == []
    assert tracker.detected([target()], 2)
    tracker.update(scene(6), 2.5)  # camera discontinuity
    assert tracker.detections(2.5) == []


def test_slow_detector_does_not_block_fresh_camera_or_grow_requests():
    jobs = []

    def submit(fn):
        future = Future()
        jobs.append((future, fn))
        return future

    loop = LiveVisionLoop(None, fast_hz=20)
    updates = PerceptionUpdates(loop, submit)
    for i in range(20):
        stamp = 10 + i * 0.05
        updates.update(scene(), stamp, stamp)
    assert loop.frames == 20 and len(jobs) == 1
    assert loop.world().timestamp == pytest.approx(10.95)
    assert loop.decision_snapshot()[2] >= 10.8
    jobs[0][0].set_result([target()])
    updates.update(scene(), 11, 11)
    assert loop.world().objects[0].name == "person-1"
    assert len(jobs) == 2
    updates.invalidate("window lost")
    jobs[1][0].set_result([target()])
    updates.update(scene(), 11.05, 11.05)
    assert not loop.world().objects
    updates.update(scene(), 11.05, 12)
    assert loop.world().timestamp is None


def test_blank_patch_does_not_invent_a_track():
    tracker = FastVisualTracker()
    frame = cv2.blur(np.zeros_like(scene()), (3, 3))
    tracker.update(frame, 1)
    tracker.detected([target()], 1)
    assert tracker.detections(1) == []
