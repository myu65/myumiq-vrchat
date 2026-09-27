"""Temporal spatial adapter; encoders/detectors remain replaceable."""

from dataclasses import dataclass

from .body import Vec3, WorldObject, WorldState


@dataclass(frozen=True)
class SpatialDetection:
    track_id: str
    position: Vec3
    confidence: float
    kind: str = "unknown"
    image_position: tuple[float, float] | None = None
    image_box: tuple[float, float, float, float] | None = None


class TemporalWorldTracker:
    def __init__(self, *, ttl_s: float = 1.5, smoothing: float = 0.35):
        if ttl_s <= 0 or not 0 < smoothing <= 1:
            raise ValueError("invalid tracker parameters")
        self.ttl_s, self.smoothing = ttl_s, smoothing
        self._tracks: dict[str, WorldObject] = {}

    def update(self, detections: list[SpatialDetection], timestamp: float) -> WorldState:
        for detection in detections:
            if not detection.track_id or not 0 <= detection.confidence <= 1:
                raise ValueError("invalid spatial detection")
            if detection.kind not in ("player", "object", "unknown"):
                raise ValueError("invalid detection kind")
            old = self._tracks.get(detection.track_id)
            position = detection.position
            velocity = (0.0, 0.0, 0.0)
            if old is not None and old.last_seen is not None and timestamp > old.last_seen:
                alpha = self.smoothing
                position = tuple(
                    (1 - alpha) * a + alpha * b for a, b in zip(old.position, detection.position)
                )
                dt = timestamp - old.last_seen
                velocity = tuple((a - b) / dt for a, b in zip(position, old.position))
            self._tracks[detection.track_id] = WorldObject(
                name=detection.track_id,
                position=position,
                source="vision",
                kind=detection.kind,
                confidence=detection.confidence,
                last_seen=timestamp,
                velocity=velocity,
                image_position=detection.image_position,
            )
        self._tracks = {
            key: value
            for key, value in self._tracks.items()
            if value.last_seen is not None and timestamp - value.last_seen <= self.ttl_s
        }
        return WorldState(
            objects=tuple(sorted(self._tracks.values(), key=lambda x: x.name)), timestamp=timestamp
        )
