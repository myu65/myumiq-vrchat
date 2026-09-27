import pytest

from myumiq_vrchat.actuation import (
    ActuatorCompositor,
    HandInputCommand,
    LocomotionCommand,
    PoseTarget,
)
from myumiq_vrchat.body import Controls
from myumiq_vrchat.postures import posture_target


def test_pose_refresh_cannot_renew_navigation_or_buttons():
    compositor = ActuatorCompositor()
    posture = posture_target("sitting_floor")
    pose = PoseTarget.from_target(posture)
    compositor.publish_pose(pose, 1.0, 1.0)
    compositor.publish_locomotion(LocomotionCommand(forward=0.4, turn=0.2), 1.0, 1.0)
    buttons = Controls(buttons=(True,) + (False,) * 17, curls=(0.8,) * 5)
    compositor.publish_hands(HandInputCommand(left=buttons), 1.0, 1.0)
    moving = compositor.compose(1.1)
    assert moving.left.controls.sticks[1] == (0.0, 0.4)
    assert moving.right.controls.sticks[1] == (0.2, 0.0)
    assert moving.left.controls.buttons[0]
    assert moving.pelvis == posture.pelvis
    compositor.publish_pose(pose, 1.2, 1.2)
    held = compositor.compose(1.2)
    assert held.left.controls == held.right.controls == Controls()
    assert held.pelvis == posture.pelvis  # Expiry never resets the body to neutral.


def test_hands_and_movement_have_independent_leases_and_pose_expiry_wins():
    compositor = ActuatorCompositor()
    compositor.publish_pose(PoseTarget.from_target(posture_target("standing")), 1.0, 1.0)
    compositor.publish_hands(HandInputCommand(left=Controls(curls=(1.0,) * 5)), 1.0, 1.0)
    compositor.publish_locomotion(LocomotionCommand(forward=0.4), 1.1, 1.1)
    target = compositor.compose(1.2)
    assert target.left.controls.sticks[1] == (0.0, 0.4)
    assert target.left.controls.curls == (0.0,) * 5
    compositor.publish_hands(HandInputCommand(left=Controls(curls=(1.0,) * 5)), 1.3, 1.3)
    target = compositor.compose(1.3)
    assert target.left.controls.sticks[1] == (0.0, 0.0) and target.left.controls.curls == (1.0,) * 5
    compositor.publish_locomotion(LocomotionCommand(forward=0.4), 1.6, 1.6)
    assert compositor.compose(1.6) is None  # Input heartbeat cannot make stale pose fresh.


def test_legacy_frame_keeps_entire_inventory_but_pose_refresh_does_not_renew_it():
    compositor = ActuatorCompositor()
    target = posture_target("standing")
    pressed = Controls(
        sticks=((0.1, 0.1), (0.2, 0.3), (0.4, 0.5), (0.6, 0.7)),
        stick_clicks=(True,) * 4,
        stick_touches=(True,) * 4,
        triggers=(0.5,) * 9,
        curls=(1.0,) * 5,
    )
    target = target.model_copy(
        update={"right": target.right.model_copy(update={"controls": pressed})}
    )
    compositor.publish_frame(target, 1.0, 1.0)
    assert compositor.compose(1.1) == target
    compositor.publish_pose(PoseTarget.from_target(target), 1.2, 1.2)
    assert compositor.compose(1.2).right.controls == Controls()


def test_conflicting_hand_stick_and_stale_or_reordered_commands_are_rejected():
    with pytest.raises(ValueError, match="locomotion"):
        HandInputCommand(left=Controls(sticks=((0.0, 0.0), (0.0, 0.2), (0.0, 0.0), (0.0, 0.0))))
    compositor = ActuatorCompositor()
    command = LocomotionCommand(forward=0.2)
    compositor.publish_locomotion(command, 1.0, 1.0)
    for stamp, now in ((0.9, 1.0), (1.0, 1.2), (2.0, 1.0), (float("nan"), 1.0)):
        with pytest.raises(ValueError, match="stale|order"):
            compositor.publish_locomotion(command, stamp, now)


def test_delayed_frame_releases_expired_inputs_but_preserves_fresh_pose():
    compositor = ActuatorCompositor()
    target = posture_target("standing")
    pressed = Controls(sticks=((0.0, 0.0), (0.4, 0.3), (0.0, 0.0), (0.0, 0.0)), curls=(1.0,) * 5)
    target = target.model_copy(
        update={"left": target.left.model_copy(update={"controls": pressed})}
    )
    compositor.publish_frame(target, 1.0, 1.2)
    held = compositor.compose(1.2)
    assert held.head == target.head and held.pelvis == target.pelvis
    assert held.left.controls == held.right.controls == Controls()
    assert compositor.stamps == dict(pose=1.0, locomotion=1.0, hands=1.0)
    assert compositor.compose(1.51) is None
    with pytest.raises(ValueError, match="stale"):
        compositor.publish_frame(target, 1.0, 1.51)
    compositor.publish_locomotion(LocomotionCommand(forward=0.2), 2.0, 2.0)
    with pytest.raises(ValueError, match="order"):
        compositor.publish_frame(target, 1.9, 2.0)
    assert compositor.stamps["pose"] == 1.0  # Failed ordering does not partly publish a pose.
