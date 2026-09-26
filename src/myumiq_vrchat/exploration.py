"""Bounded controller locomotion, independent of model inference."""

import base64
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from .body import Controls, Frozen, Number

DIRECTED_MOVEMENT = {"MOVE_FORWARD": "forward", "TURN_LEFT": "left", "TURN_RIGHT": "right"}
NAVIGATION_SKILLS = frozenset(("EXPLORE_HOME", *DIRECTED_MOVEMENT))


class ExplorationSettings(Frozen):
    enabled: bool = False
    allow_controlled_home: bool = False
    mode: Literal["continuous", "pulse_test"] = "continuous"
    controller_hz: Number = Field(default=30, ge=10, le=60)
    acceleration: Number = Field(default=0.9, gt=0, le=5)
    deceleration: Number = Field(default=1.8, gt=0, le=10)
    continuous_strength: Number = Field(default=0.45, gt=0, le=0.6)
    turn_strength: Number = Field(default=0.85, gt=0, le=1)


class PrivateHomeGate:
    """Fail closed on missing startup evidence or a later world transition."""

    def __init__(self, root, allow_controlled_home=False):
        self.root = root
        self.allow_controlled_home = allow_controlled_home

    def valid(self):
        try:
            startup = json.loads((self.root / "startup.json").read_text("utf-8"))
            private = startup.get("private_home") is True
            controlled = (
                self.allow_controlled_home
                and startup.get("controlled_home") is True
                and bool(startup.get("instance"))
            )
            if not (private or controlled):
                return False
            text = Path(startup["vrchat_log"]).read_text("utf-8", errors="replace")
            joins = [s for s in text.splitlines() if "[Behaviour] Joining wrld_" in s]
            rooms = [s for s in text.splitlines() if "[Behaviour] Entering Room:" in s]
            if (
                not joins
                or not rooms
                or not rooms[-1].strip().endswith("Entering Room: VRChat Home")
            ):
                return False
            instance = joins[-1].split("[Behaviour] Joining ", 1)[1].strip()
            if startup.get("instance") and instance != startup["instance"]:
                return False
            friends_plus = (
                controlled
                and startup.get("allow_friends_plus_home") is True
                and "~hidden(" in instance
            )
            return "~private(" in instance or controlled and "~friends(" in instance or friends_plus
        except (OSError, KeyError, ValueError):
            return False


def describe_image(encoded):
    import cv2

    image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid exploration image")
    # Central scene region, excluding desktop borders. A descriptor is not a map.
    h, w = image.shape[:2]
    crop = image[h // 8 : 7 * h // 8, w // 8 : 7 * w // 8]
    return cv2.resize(crop, (24, 16)).astype(np.float32).reshape(-1) / 255.0


class Explorer:
    """One writer on slow executive. Motors consume expiring immutable leases."""

    def __init__(self):
        self.key = None
        self.views = []
        self.phase = "idle"
        self.deadline = 0.0
        self.before = None
        self.before_time = 0.0
        self.direction = "right"
        self.cycles = self.unchanged = 0
        self.last_frame = -1.0
        self.lease = None
        self.outcomes = []

    def reset(self):
        self.key = None
        self.lease = None
        self.phase = "idle"

    def tick(self, key, now, snapshot, permitted, motion_ready=True, *, direction=None):
        if direction is not None and direction not in DIRECTED_MOVEMENT.values():
            raise ValueError("unknown directed movement")
        self.lease = None
        if not permitted:
            self.reset()
            return None
        if key != self.key:
            self.key = key
            self.phase = "observe"
            self.deadline = now + 1.0
            self.before = None
        fresh = snapshot is not None and 0 <= now - snapshot[2] < 0.75
        if not fresh or not motion_ready:
            # A missing observation revokes actuation, not the attempt. Never
            # replay the remainder of a pulse after an unobserved interval.
            if self.phase == "pulse":
                self.phase = "settle"
                self.deadline = now + 1.0
            return None
        if self.phase == "pulse":
            if now < self.deadline:
                self.lease = (key, min(self.deadline, now + 0.15), self.direction)
                return self.lease
            self.phase = "settle"
            self.deadline = now + 1.0
        if now < self.deadline or snapshot[2] == self.last_frame:
            return None
        descriptor = describe_image(snapshot[1])
        self.last_frame = snapshot[2]
        novelty = min((float(np.mean(np.abs(descriptor - v))) for v in self.views), default=1.0)
        if self.before is not None and snapshot[2] > self.before_time:
            change = float(np.mean(np.abs(descriptor - self.before)))
            self.unchanged = self.unchanged + 1 if change < 0.015 else 0
            self.outcomes.append(
                dict(
                    direction=self.direction,
                    image_change=change,
                    novelty=novelty,
                    outcome="changed_view" if change >= 0.015 else "unchanged_view",
                    scope="image_change_not_metric_displacement",
                    timestamp=snapshot[2],
                )
            )
            self.cycles += 1
        self.views.append(descriptor)
        self.views = self.views[-64:]
        # Repeated no-response stops the skill instead of pushing a wall forever.
        if self.unchanged >= 3:
            self.phase = "blocked"
            self.deadline = now + 60.0
            return None
        # Scan first; alternate short advances and scans. Novelty changes turn choice.
        self.direction = direction or (
            "forward"
            if self.cycles % 2 and self.unchanged == 0
            else "left"
            if novelty < 0.02 and self.cycles % 4 == 2
            else "right"
        )
        self.before, self.before_time = descriptor, snapshot[2]
        self.phase = "pulse"
        self.deadline = now + 0.35
        self.lease = (key, now + 0.15, self.direction)
        return self.lease


class ContinuousExplorer(Explorer):
    """Keep issuing short leases while an unchanged goal has valid visual input.

    Image response is sampled independently of the controller clock. It remains
    a mapless navigation limit, never a distance or collision observation.
    """

    def tick(self, key, now, snapshot, permitted, motion_ready=True, *, direction=None):
        self.lease = None
        if not permitted:
            self.reset()
            return None
        if key != self.key:
            self.key, self.phase = key, "continuous"
            self.before = None
            self.deadline = now
        if snapshot is None or not 0 <= now - snapshot[2] < 0.75:
            self.phase = "waiting_observation"
            return None
        if self.unchanged >= 3:
            self.phase = "blocked"
            return None
        if now >= self.deadline and snapshot[2] != self.last_frame:
            descriptor = describe_image(snapshot[1])
            self.last_frame = snapshot[2]
            if self.before is not None:
                change = float(np.mean(np.abs(descriptor - self.before)))
                self.unchanged = self.unchanged + 1 if change < 0.015 else 0
                self.outcomes.append(
                    dict(
                        direction=self.direction,
                        image_change=change,
                        outcome="changed_view" if change >= 0.015 else "unchanged_view",
                        scope="image_change_not_metric_displacement",
                        timestamp=snapshot[2],
                    )
                )
                self.cycles += 1
            self.before, self.before_time = descriptor, snapshot[2]
            self.deadline = now + 1.0
        if self.unchanged >= 3:
            self.phase = "blocked"
            return None
        self.direction = direction or ("forward" if self.cycles % 4 in (1, 2) else "right")
        if self.direction not in ("forward", "backward", "left", "right"):
            raise ValueError("unknown navigation direction")
        self.phase = "continuous"
        self.lease = (key, now + 0.15, self.direction)
        return self.lease


def drive_target(target, direction, strength=0.45):
    """Official VMT compatible profile: joystick 1 is the Index thumbstick."""
    if direction not in ("forward", "backward", "left", "right"):
        raise ValueError("unknown locomotion direction")
    if not 0 < strength <= 0.6:
        raise ValueError("bounded locomotion strength required")
    hand = "left" if direction in ("forward", "backward") else "right"
    # Snap-turn bindings require crossing a threshold; the short lease bounds it.
    if hand == "right":
        strength = 0.85
    stick = (
        (0.0, strength if direction == "forward" else -strength)
        if hand == "left"
        else (-strength if direction == "left" else strength, 0.0)
    )
    controls = Controls(
        sticks=((0.0, 0.0), stick, (0.0, 0.0), (0.0, 0.0)),
        stick_touches=(False, True, False, False),
    )
    return target.model_copy(
        update={hand: getattr(target, hand).model_copy(update={"controls": controls})}
    )
