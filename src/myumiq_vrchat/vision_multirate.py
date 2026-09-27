"""One camera owner, bounded detector work and fast local tracking."""

import time
from contextlib import ExitStack

from .body import WorldState
from .fast_tracking import FastVisualTracker
from .perception import TemporalWorldTracker
from .vision import CaptureUnavailable, decision_image, vrchat_client_rect


class PerceptionUpdates:
    """Purely polled by the existing vision worker; one detector job at a time."""

    def __init__(self, loop, submit=None):
        from .purpose_runtime import background

        self.loop, self.submit = loop, submit or background
        self.tracker = FastVisualTracker()
        self.pending = None
        self.generation = 0
        self.next_detection = self.next_image = 0.0
        self.detections = 0
        self.error = None

    def invalidate(self, reason):
        self.generation += 1
        self.tracker.reset()
        with self.loop._lock:
            self.loop._world = WorldState()
            self.loop._image_world = self.loop._decision_image = None
            self.loop.capture_pending = str(reason)

    def update(self, image, stamp, now):
        if not 0 <= now - stamp <= 0.25:
            self.invalidate("camera frame expired")
            return
        new_frame = self.tracker.update(image, stamp)
        if self.pending and self.pending[0].done():
            future, captured, generation = self.pending
            self.pending = None
            try:
                detections = future.result()
                if generation == self.generation:
                    self.tracker.detected(detections, captured)
                    self.detections += 1
                self.error = None
            except Exception as exc:
                self.error = str(exc)[:300]
        if new_frame and self.pending is None and now >= self.next_detection:
            # Detectors may mutate their input and are never allowed concurrent calls.
            frozen = image.copy()
            self.pending = (
                self.submit(lambda: self.loop.detector.detect(frozen, stamp)),
                stamp,
                self.generation,
            )
            self.next_detection = now + 1 / self.loop.hz
        if not new_frame:
            return
        # Lost tracks disappear immediately; prediction is not a fresh observation.
        world = TemporalWorldTracker(smoothing=1).update(self.tracker.detections(stamp), stamp)
        encoded = decision_image(image) if now >= self.next_image else None
        if encoded is not None:
            self.next_image = now + 0.2
        with self.loop._lock:
            self.loop._world = world
            if encoded is not None:
                self.loop._image_world, self.loop._decision_image = world, encoded
            self.loop.capture_pending = None
            self.loop.tracking_health = dict(
                fast_frames=self.loop.frames + 1,
                detector_frames=self.detections,
                detector_pending=self.pending is not None,
                detector_error=self.error,
                detector_age_s=now - self.pending[1] if self.pending else None,
                tracks=len(world.objects),
                captured_at=stamp,
            )
        self.loop.frames += 1


def run(loop):
    updates = PerceptionUpdates(loop)
    try:
        with ExitStack() as stack:
            if loop.capture_backend == "windows_graphics":
                from .window_capture import WindowCapture

                capture = WindowCapture(loop.title, loop.fast_hz)
                stack.callback(capture.close)
                read = capture.read
            else:
                import mss
                import numpy as np

                capture = stack.enter_context(mss.mss())

                def read():
                    rect = vrchat_client_rect(loop.title)
                    started = time.perf_counter()
                    return np.asarray(capture.grab(rect))[:, :, :3], started

            while loop._running:
                began = time.perf_counter()
                try:
                    image, stamp = read()
                    updates.update(image, stamp, time.perf_counter())
                except CaptureUnavailable as exc:
                    updates.invalidate(exc)
                time.sleep(max(0, 1 / loop.fast_hz - (time.perf_counter() - began)))
    except Exception as exc:
        loop.error = exc
        loop._running = False
