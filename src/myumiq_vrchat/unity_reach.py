"""Local Unity REACH environment client; no VRChat or device configuration."""

import json
import math
import socket
from pathlib import Path
from typing import Literal

from pydantic import Field

from .body import Frozen, Number, Vec3


class ReachStep(Frozen):
    protocol: Literal["myumiq-reach-v1"]
    error: str | None = None
    observation: tuple[Number, ...] = Field(min_length=9, max_length=9)
    hand: Vec3
    goal: Vec3
    reward: Number
    terminated: bool
    truncated: bool
    success: bool
    step: int = Field(ge=0, le=150)


def integrate_reach(
    position: Vec3,
    velocity: Vec3,
    action: Vec3,
    dt: float = 1 / 30,
) -> tuple[Vec3, Vec3]:
    """Same bounded velocity/acceleration contract as Unity, in canonical metres."""
    if not 0 < dt <= 0.1 or not all(math.isfinite(v) for v in (*position, *velocity, *action)):
        raise ValueError("invalid reach integration input")
    if any(abs(v) > 1 for v in action):
        raise ValueError("reach action must be within -1..1")
    norm = max(1.0, math.hypot(*action))
    desired = tuple(v * 0.6 / norm for v in action)
    delta = tuple(a - b for a, b in zip(desired, velocity))
    fraction = min(1.0, 1.8 * dt / max(1e-12, math.hypot(*delta)))
    speed = tuple(a + fraction * b for a, b in zip(velocity, delta))
    next_position = tuple(a + dt * b for a, b in zip(position, speed))
    if math.dist(next_position, (0.05, -0.2, 1.35)) > 0.65 or next_position[2] < 0.6:
        return position, (0.0, 0.0, 0.0)
    return next_position, speed


class UnityReachClient:
    def __init__(self, ready_file: Path):
        ready = json.loads(ready_file.read_text(encoding="utf-8-sig"))
        if (
            ready.get("protocol") != "myumiq-reach-v1"
            or not isinstance(ready.get("port"), int)
            or not 1024 <= ready["port"] <= 65535
            or not isinstance(ready.get("token"), str)
            or len(ready["token"]) != 32
        ):
            raise ValueError("invalid Unity environment readiness file")
        self._token = ready["token"]
        self.socket = socket.create_connection(("127.0.0.1", ready["port"]), timeout=10)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.stream = self.socket.makefile("rwb")

    def _request(self, command: str, **data) -> ReachStep:
        message = json.dumps({"token": self._token, "command": command, **data}, allow_nan=False)
        self.stream.write(message.encode("utf-8") + b"\n")
        self.stream.flush()
        response = self.stream.readline(8193)
        if not response or len(response) > 8192:
            raise RuntimeError("Unity response missing or oversized")
        content = json.loads(response)
        if content.get("error"):
            raise RuntimeError(f"Unity REACH: {content['error']}")
        return ReachStep.model_validate_json(response)

    def reset(self, seed: int = 0, goal: Vec3 | None = None) -> ReachStep:
        return self._request("reset", seed=seed, goal=goal)

    def step(self, action: Vec3) -> ReachStep:
        if len(action) != 3 or not all(math.isfinite(v) and -1 <= v <= 1 for v in action):
            raise ValueError("expected three normalized finite action values")
        return self._request("step", action=action)

    def close(self) -> None:
        # Closing the sole client also ends this dedicated Unity environment.
        try:
            self.stream.close()
        finally:
            self.socket.close()
