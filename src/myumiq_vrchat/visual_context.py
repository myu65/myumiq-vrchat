"""Coherent, bounded visual evidence for slow model workers, never motor I/O."""

import hashlib
import time
from dataclasses import dataclass

from .body import WorldState


def visible_world(world, now, max_age_s=3.0):
    """Retain fixture facts, but never promote stale detector tracks to observations."""
    objects = tuple(
        o
        for o in world.objects
        if o.source != "vision"
        or (
            o.last_seen is not None and 0 <= now - o.last_seen <= max_age_s and o.confidence >= 0.35
        )
    )
    # Recent confident detections get the bounded context budget, not lexical IDs.
    objects = tuple(sorted(objects, key=lambda o: (-o.confidence, o.name)))[:8]
    return world.model_copy(update={"objects": objects})


@dataclass(frozen=True)
class VisualContext:
    world: WorldState
    image_base64: str | None = None
    captured_at: float | None = None
    reason: str | None = None
    max_age_s: float = 3.0
    image_mime: str = "image/jpeg"

    def fresh_image(self, now=None):
        now = time.perf_counter() if now is None else now
        return (
            self.image_base64
            if (self.captured_at is not None and 0 <= now - self.captured_at <= self.max_age_s)
            else None
        )

    def metadata(self, now=None):
        image = self.fresh_image(now)
        return {
            "image_used": image is not None,
            "captured_at": self.captured_at,
            "image_sha256": hashlib.sha256(image.encode()).hexdigest() if image else None,
            "unavailable_reason": None if image else self.reason or "stale_or_missing_image",
        }


def visual_context(vision, world, now, *, use_image=False, max_age_s=3.0):
    reason = "image_disabled" if not use_image else "capture_unavailable"
    if use_image and vision is not None:
        try:
            observed, image, captured = vision.decision_snapshot()
            if image and 0 <= now - captured <= max_age_s:
                return VisualContext(
                    visible_world(observed, now, max_age_s), image, captured, max_age_s=max_age_s
                )
            reason = "stale_or_missing_image"
        except (RuntimeError, ValueError, OSError) as exc:
            reason = str(exc)[:160]
    return VisualContext(visible_world(world, now, max_age_s), reason=reason, max_age_s=max_age_s)
