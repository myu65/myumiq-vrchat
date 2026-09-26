import math

import pytest

from myumiq_vrchat.body import Pose, rest_target
from myumiq_vrchat.locomotion_frames import LocomotionFrames
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.whole_body import PARTS


def test_tracking_and_controller_world_placement_are_composed_once():
    frames = LocomotionFrames(
        tracking_from_body=Pose(position=(1.0, 0.0, 0.0)),
        world_from_tracking=Pose(
            position=(10.0, 0.0, 0.0), orientation=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
        ),
    )
    body = posture_target("standing")
    assert frames.tracker_targets(body).head.position == pytest.approx((1, 0, 1.6))
    assert frames.world_targets(body).head.position == pytest.approx((10, 1, 1.6))
    assert frames.world_targets(body).left.controls == body.left.controls


def test_recenter_changes_tracking_coordinates_without_world_motion():
    frames = LocomotionFrames(
        tracking_from_body=Pose(position=(1.0, 2.0, 0.0)),
        world_from_tracking=Pose(position=(5.0, 3.0, 0.0)),
    )
    body = posture_target("standing")
    changed = frames.recentered(Pose(position=(-2.0, 1.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)))
    assert changed.tracking_epoch == 1
    for part in PARTS:
        assert changed.world_targets(body).pose_for(part).position == pytest.approx(
            frames.world_targets(body).pose_for(part).position
        )
    assert changed.tracker_targets(body).head.position != frames.tracker_targets(body).head.position


def test_missing_world_measurement_stays_unknown_after_recenter():
    frames = LocomotionFrames(tracking_from_body=Pose(position=(1.0, 0.0, 0.0)))
    changed = frames.recentered(Pose(position=(2.0, 0.0, 0.0)))
    assert changed.world_from_tracking is None
    with pytest.raises(ValueError, match="unobserved"):
        changed.world_targets(rest_target())
