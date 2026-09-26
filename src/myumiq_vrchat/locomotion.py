"""Local navigation clock and shared gait demand; no inference or device writes."""

import math
from dataclasses import dataclass, replace

from .actuation import LocomotionCommand


@dataclass(frozen=True)
class LocomotionState:
    requested_forward_speed: float = 0.0
    requested_strafe_speed: float = 0.0
    requested_yaw_rate: float = 0.0
    smoothed_forward_speed: float = 0.0
    smoothed_strafe_speed: float = 0.0
    smoothed_yaw_rate: float = 0.0
    gait_phase: float = 0.0
    heading_error: float | None = None
    target_track_id: str | None = None
    confidence: float = 0.0

    @property
    def gait_speed_scale(self):
        return min(
            1.0,
            max(
                math.hypot(self.smoothed_forward_speed, self.smoothed_strafe_speed),
                abs(self.smoothed_yaw_rate),
            )
            / 0.45,
        )

    def command(self):
        return LocomotionCommand(
            forward=self.smoothed_forward_speed,
            strafe=self.smoothed_strafe_speed,
            turn=self.smoothed_yaw_rate,
        )


class LocomotionController:
    def __init__(self, hz=30.0, acceleration=0.9, deceleration=1.8):
        if not all(math.isfinite(v) and v > 0 for v in (hz, acceleration, deceleration)):
            raise ValueError("positive finite locomotion rates required")
        self.hz, self.acceleration, self.deceleration = hz, acceleration, deceleration
        self.state = LocomotionState()
        self.last_at = None
        self.next_at = None
        self.key = None

    def step(self, command, now, *, permitted, key):
        if not math.isfinite(now):
            raise ValueError("finite locomotion clock required")
        if (
            not permitted
            or key != self.key
            or self.last_at is not None
            and (now < self.last_at or now - self.last_at > 0.25)
        ):
            # Expired authority is an immediate release, never a deceleration
            # tail that can continue past stop, Home loss or command preemption.
            self.state = LocomotionState()
            self.last_at, self.key = now, key
            self.next_at = now + 1 / self.hz
            if not permitted:
                return self.state
        if self.last_at is None:
            self.last_at, self.key = now, key
            self.next_at = now + 1 / self.hz
        dt = now - self.last_at
        if now + 1e-9 < self.next_at:
            return self.state
        self.last_at = now
        # Keep the deadline anchored. Resetting it to now+period would turn a
        # 30 Hz controller sampled by a 100 Hz producer into a 25 Hz controller.
        period = 1 / self.hz
        self.next_at += max(1, math.floor((now - self.next_at) / period) + 1) * period

        def smooth(current, requested):
            slowing = abs(requested) < abs(current) or current * requested < 0
            # Reverse by decelerating through zero first.
            target = 0.0 if current * requested < 0 else requested
            limit = (self.deceleration if slowing else self.acceleration) * min(dt, 0.1)
            return current + max(-limit, min(limit, target - current))

        self.state = replace(
            self.state,
            requested_forward_speed=command.forward,
            requested_strafe_speed=command.strafe,
            requested_yaw_rate=command.turn,
            smoothed_forward_speed=smooth(self.state.smoothed_forward_speed, command.forward),
            smoothed_strafe_speed=smooth(self.state.smoothed_strafe_speed, command.strafe),
            smoothed_yaw_rate=smooth(self.state.smoothed_yaw_rate, command.turn),
            confidence=1.0,
        )
        return self.state


def direction_command(direction, strength=0.45):
    if direction == "forward":
        return LocomotionCommand(forward=strength)
    if direction == "backward":
        return LocomotionCommand(forward=-strength)
    if direction in ("left", "right"):
        return LocomotionCommand(turn=-strength if direction == "left" else strength)
    raise ValueError("unknown navigation direction")


def apply_locomotion(target, command):
    def controls(hand, stick):
        values = getattr(target, hand).controls
        return getattr(target, hand).model_copy(
            update={
                "controls": values.model_copy(
                    update={
                        "sticks": tuple(
                            stick if i == 1 else v for i, v in enumerate(values.sticks)
                        ),
                        "stick_touches": tuple(
                            stick != (0.0, 0.0) if i == 1 else v
                            for i, v in enumerate(values.stick_touches)
                        ),
                    }
                )
            }
        )

    return target.model_copy(
        update={
            "left": controls("left", (command.strafe, command.forward)),
            "right": controls("right", (command.turn, 0.0)),
        }
    )
