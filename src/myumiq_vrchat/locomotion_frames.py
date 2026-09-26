"""Separate tracking-space body motion from world locomotion measurements."""

from pydantic import Field

from .body import EXTRA_PARTS, BodyTarget, Frozen, Pose, Stick, compose, inverse


def transform_body(frame: Pose, body: BodyTarget) -> BodyTarget:
    updates = {"head": compose(frame, body.head)}
    for hand in ("left", "right"):
        value = getattr(body, hand)
        updates[hand] = value.model_copy(update={"pose": compose(frame, value.pose)})
    for part in EXTRA_PARTS:
        pose = getattr(body, part)
        updates[part] = compose(frame, pose) if pose is not None else None
    return body.model_copy(update=updates)


class LocomotionRequest(Frozen):
    """Requested axes, not metres or observed displacement."""

    move: Stick = (0.0, 0.0)
    turn: Stick = (0.0, 0.0)
    binding_profile: str = Field(min_length=1)


class LocomotionFrames(Frozen):
    """Transforms use canonical forward/left/up metres and active wxyz."""

    tracking_from_body: Pose
    world_from_tracking: Pose | None = None
    tracking_epoch: int = Field(default=0, ge=0)

    def tracker_targets(self, articulation: BodyTarget) -> BodyTarget:
        # Sending world coordinates to a tracker would double-count locomotion.
        return transform_body(self.tracking_from_body, articulation)

    def world_targets(self, articulation: BodyTarget) -> BodyTarget:
        if self.world_from_tracking is None:
            raise ValueError("world placement is unobserved")
        return transform_body(
            compose(self.world_from_tracking, self.tracking_from_body), articulation
        )

    def recentered(self, new_tracking_from_old: Pose) -> "LocomotionFrames":
        """A coordinate change, not physical movement or a rewardable action."""
        return LocomotionFrames(
            tracking_from_body=compose(new_tracking_from_old, self.tracking_from_body),
            world_from_tracking=(
                compose(self.world_from_tracking, inverse(new_tracking_from_old))
                if self.world_from_tracking is not None
                else None
            ),
            tracking_epoch=self.tracking_epoch + 1,
        )
