"""Whole-body facing references; the learned actor generates all joint rates."""

import math

import numpy as np

from .body import qmul, rotate
from .gaze import image_target
from .tracker_policy import pose_error
from .whole_body import target_from_vector, vector


class BodyFacing:
    def __init__(self, current, target):
        self.pose, self.name = current, target
        self.last_seen = None
        self.centred_frames = 0
        self.error = None
        self.pending_since = None
        self.pending_reason = None

    def observe(self, current, world, now):
        try:
            obj = image_target(world, self.name, now)
        except ValueError as exc:
            # A low-rate detector can miss one frame. Hold the observed whole
            # posture while waiting; stale geometry must never drive rotation.
            self.pose = current
            self.pending_reason = str(exc)
            if self.pending_since is None:
                self.pending_since = now
            if now - self.pending_since >= 1.5:
                self.error = self.pending_reason
            return False
        self.pending_since = self.pending_reason = None
        if obj.last_seen == self.last_seen:
            return True
        if self.last_seen is not None and obj.last_seen - self.last_seen > 0.75:
            self.centred_frames = 0
        self.last_seen = obj.last_seen
        error_x = obj.image_position[0]
        self.centred_frames = self.centred_frames + 1 if abs(error_x) < 0.12 else 0
        error = pose_error(current, self.pose)
        if (
            np.linalg.norm(error[:, :3], axis=1).max() > 0.12
            or np.linalg.norm(error[:, 3:], axis=1).max() > 0.35
        ):
            return True
        # Image right is canonical negative yaw. This is a bounded task reference,
        # not a per-frame head command. One common transform preserves the posture.
        angle = max(-0.12, min(0.12, -0.35 * error_x)) if abs(error_x) >= 0.08 else 0.0
        turn = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
        pivot = np.asarray(current.pelvis.position)
        points = vector(current)
        for row in points:
            row[:3] = pivot + rotate(turn, tuple(row[:3] - pivot))
            row[3:] = qmul(turn, tuple(row[3:]))
        self.pose = target_from_vector(points)
        return True

    def evidence(self, world, now):
        try:
            obj = image_target(world, self.name, now)
            success = self.centred_frames >= 2 and abs(obj.image_position[0]) < 0.12
        except ValueError:
            success = None
        return {
            "success": success,
            "scope": "fresh_visual_body_heading",
            "centred_frames": self.centred_frames,
            "vertical_alignment_verified": False,
            "avatar_verified": False,
        }
