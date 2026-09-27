"""CPU image continuity, independent of semantic detection and model requests."""

from collections import deque
from dataclasses import replace

import cv2
import numpy as np


class FastVisualTracker:
    """Sparse forward/backward optical flow; no identity or metric-depth claims.

    Detector results are seeded on their original frame, then replayed through a
    bounded image history. A late result can never be stamped as a current view.
    """

    def __init__(self, max_age=2.0):
        self.max_age = max_age
        self.reset()

    def reset(self):
        self.history = deque(maxlen=45)
        self.tracks = {}
        self.last_detection = None

    @staticmethod
    def gray(image):
        ratio = min(1.0, 640 / max(image.shape[:2]))
        frame = cv2.resize(image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio)))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def seed(gray, detection):
        if detection.image_position is None:
            return None
        h, w = gray.shape
        x, y = ((v + 1) * n / 2 for v, n in zip(detection.image_position, (w, h)))
        mask = np.zeros_like(gray)
        box = detection.image_box
        if box is None:
            x0, y0, x1, y1 = x - 28, y - 28, x + 28, y + 28
        else:
            x0, y0, x1, y1 = [v * n for v, n in zip(box, (w, h, w, h))]
        mask[max(0, int(y0)) : min(h, int(y1)), max(0, int(x0)) : min(w, int(x1))] = 255
        points = cv2.goodFeaturesToTrack(gray, 32, 0.02, 5, mask=mask)
        return points if points is not None and len(points) >= 4 else None

    @staticmethod
    def flow(before, after, item):
        detection, points, detected_at = item
        if before.shape != after.shape:
            return None
        following, good, _ = cv2.calcOpticalFlowPyrLK(
            before, after, points, None, winSize=(21, 21), maxLevel=3
        )
        if following is None:
            return None
        backward, reverse, _ = cv2.calcOpticalFlowPyrLK(
            after, before, following, None, winSize=(21, 21), maxLevel=3
        )
        if backward is None:
            return None
        valid = (good.ravel() == 1) & (reverse.ravel() == 1)
        valid &= np.linalg.norm((backward - points).reshape(-1, 2), axis=1) < 1.5
        valid &= np.isfinite(following).all(axis=(1, 2))
        if valid.sum() < max(4, len(points) * 0.5):
            return None
        shift = (following - points).reshape(-1, 2)[valid]
        delta = np.median(shift, axis=0)
        if np.median(np.linalg.norm(shift - delta, axis=1)) > 4:
            return None
        h, w = after.shape
        dx, dy = 2 * delta / (w, h)
        x, y = np.asarray(detection.image_position) + (dx, dy)
        if not (-1 <= x <= 1 and -1 <= y <= 1):
            return None
        forward, left, up = detection.position
        # Preserve the detector's uncertain depth; only update the bearing proxy.
        moved = replace(
            detection,
            image_position=(float(x), float(y)),
            position=(forward, left - dx * forward / 1.6, up - dy * forward / 1.6),
        )
        return moved, following[valid], detected_at

    def update(self, image, stamp):
        if self.history and stamp <= self.history[-1][0]:
            return False
        gray = self.gray(image)
        if self.history:
            previous_at, previous = self.history[-1]
            tracks = {}
            if stamp - previous_at <= 0.25:
                for key, item in self.tracks.items():
                    result = self.flow(previous, gray, item)
                    if result and stamp - result[2] <= self.max_age:
                        tracks[key] = result
            self.tracks = tracks
        self.history.append((stamp, gray))
        return True

    def detected(self, detections, stamp):
        frames = list(self.history)
        index = next((i for i, (t, _) in enumerate(frames) if t == stamp), None)
        if index is None or (self.last_detection is not None and stamp <= self.last_detection):
            return False
        tracks = {}
        for detection in detections[:32]:
            points = self.seed(frames[index][1], detection)
            if points is None:
                continue
            item = (detection, points, stamp)
            for (_, before), (at, after) in zip(frames[index:], frames[index + 1 :]):
                item = self.flow(before, after, item)
                if item is None or at - stamp > self.max_age:
                    break
            else:
                tracks[detection.track_id] = item
        self.tracks = tracks
        self.last_detection = stamp
        return True

    def detections(self, now):
        return [
            replace(d, confidence=d.confidence * max(0, 1 - (now - at) / (2 * self.max_age)))
            for d, _, at in self.tracks.values()
            if 0 <= now - at <= self.max_age
        ]
