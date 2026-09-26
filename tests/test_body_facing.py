import numpy as np
import pytest

from myumiq_vrchat.body import WorldObject, WorldState
from myumiq_vrchat.body_facing import BodyFacing
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.whole_body import vector


def world(now, x=0.4):
    return WorldState(
        timestamp=now,
        objects=(
            WorldObject(
                name="person",
                source="vision",
                kind="player",
                confidence=0.9,
                position=(1.0, 0.0, 1.6),
                last_seen=now,
                image_position=(x, 0.1),
            ),
        ),
    )


def test_facing_rotates_whole_pose_without_default_reset_or_reusing_a_frame():
    current = posture_target("crouching")
    reference = BodyFacing(current, "person")
    reference.observe(current, world(10.0), 10.0)
    a, b = vector(current), vector(reference.pose)
    assert reference.pose.pelvis.position == current.pelvis.position
    assert reference.pose.head.position[2] == pytest.approx(current.head.position[2])
    assert not np.allclose(a[:, 3:], b[:, 3:])
    np.testing.assert_allclose(
        np.linalg.norm(a[:, None, :3] - a[None, :, :3], axis=2),
        np.linalg.norm(b[:, None, :3] - b[None, :, :3], axis=2),
    )
    held = reference.pose
    reference.observe(held, world(10.0), 10.1)
    assert reference.pose is held
    reference.observe(held, world(10.0), 11.0)
    assert reference.pose is held and reference.error is None
    assert reference.pending_reason
    assert reference.observe(held, world(10.0), 12.5) is False
    assert reference.error


def test_brief_detection_loss_holds_measured_pose_and_recovers_same_target():
    current = posture_target("crouching")
    reference = BodyFacing(current, "person")
    assert reference.observe(current, world(10.0), 10.0)
    old_goal = reference.pose
    assert reference.observe(current, WorldState(timestamp=10.2), 10.2) is False
    assert reference.pose is current  # no continuing toward the stale goal
    assert reference.pose != old_goal
    assert reference.centred_frames == 0 and reference.error is None
    assert reference.observe(current, world(10.5), 10.5)
    assert reference.pending_since is None and reference.pending_reason is None
    assert reference.pose != current and reference.error is None


def test_facing_success_requires_multiple_fresh_centered_images():
    reference = BodyFacing(posture_target("standing"), "person")
    reference.observe(reference.pose, world(10.0, 0.01), 10.0)
    assert reference.evidence(world(10.0, 0.01), 10.0)["success"] is False
    reference.observe(reference.pose, world(10.1, 0.01), 10.1)
    result = reference.evidence(world(10.1, 0.01), 10.1)
    assert result["success"] is True and result["scope"] == "fresh_visual_body_heading"
    assert reference.evidence(world(10.1, 0.01), 12.0)["success"] is None


def test_waiting_for_next_result_preserves_history_but_never_uses_stale_geometry():
    reference = BodyFacing(posture_target("standing"), "person")
    current = reference.pose
    reference.observe(current, world(10.0, 0.01), 10.3)
    assert not reference.observe(current, world(10.0, 0.01), 10.76)
    assert reference.pose is current
    assert reference.evidence(world(10.0, 0.01), 10.76)["success"] is None
    # Two adjacent captures can be published after the old snapshot expires.
    reference.observe(current, world(10.5, 0.01), 10.8)
    assert reference.evidence(world(10.5, 0.01), 10.8)["success"] is True
    # A genuinely missing capture interval breaks consecutive-image evidence.
    reference.observe(current, world(11.5, 0.01), 11.8)
    assert reference.centred_frames == 1
