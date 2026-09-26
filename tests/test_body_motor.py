import math

import pytest
from pydantic import ValidationError

from myumiq_vrchat.backends.osc import from_openvr, to_openvr
from myumiq_vrchat.body import Controls, Pose, WorldObject, WorldState, rest_target, simulated_body
from myumiq_vrchat.cognition import Goal
from myumiq_vrchat.motor import MotionCommand, ProceduralMotor


@pytest.mark.parametrize(
    "payload",
    [
        '{"skill":"MOVE","duration_s":1}',
        '{"skill":"WAIT","duration_s":NaN}',
        '{"skill":"WAIT","duration_s":true}',
        '{"skill":"WAIT","duration_s":"2"}',
        '{"skill":"WAVE","duration_s":1}',
        '{"skill":"LOOK_AT","duration_s":1}',
        '{"skill":"WAIT","duration_s":1,"button":1}',
        '{"skill":"WAIT","duration_s":1,"hand":"left"}',
        '{"skill":"WAIT","duration_s":11}',
    ],
)
def test_reject_invalid_intents(payload):
    with pytest.raises(ValidationError):
        Goal.model_validate_json(payload)


def test_unknown_world_reference_rejected():
    goal = Goal(skill="LOOK_AT", duration_s=1.0, target="missing")
    with pytest.raises(ValueError):
        goal.validate_world(WorldState())


def test_body_is_frozen_finite_and_distinct_from_targets():
    pose = Pose(position=(1.0, 2.0, 3.0))
    with pytest.raises(ValidationError):
        pose.position = (0.0, 0.0, 0.0)
    with pytest.raises(ValidationError):
        Pose(position=(0.0, float("nan"), 0.0))
    with pytest.raises(ValidationError):
        Pose(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 0.0))
    target = rest_target()
    observed = simulated_body(target, 123.0)
    assert observed.head.source == "simulated"
    assert observed.head.timestamp == 123.0
    assert not hasattr(target, "source")


def test_coordinate_basis_and_quaternion_round_trip():
    for canonical, expected in [
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ]:
        assert to_openvr(Pose(position=canonical)).position == expected
    pose = Pose(position=(1.0, 2.0, 3.0), orientation=(0.5, 0.5, -0.5, 0.5))
    assert from_openvr(to_openvr(pose)) == pose


def test_complete_control_inventory_rejects_sparse_invalid_inputs():
    with pytest.raises(ValidationError):
        Controls(buttons=(False,))
    with pytest.raises(ValidationError):
        Controls(triggers=(2.0,) * 9)
    with pytest.raises(ValidationError):
        Controls(sticks=((float("inf"), 0.0),) * 4)


def test_wave_moves_selected_hand_with_bounded_speed_and_holds_after_expiry():
    policy = ProceduralMotor()
    rest = rest_target()
    goal = Goal(skill="WAVE", hand="right", duration_s=2.0)
    previous, highest = rest, rest.right.pose.position[2]
    for n in range(360):
        body = simulated_body(previous, n / 60)
        target = policy.step(body, MotionCommand(goal=goal, elapsed_s=n / 60), WorldState(), 1 / 60)
        assert (
            math.dist(target.right.pose.position, previous.right.pose.position) <= 0.6 / 60 + 1e-9
        )
        assert target.left == rest.left
        assert target.head == rest.head
        assert not any(target.right.controls.buttons)
        assert not any(target.right.controls.triggers)
        if n >= 120:
            assert target.right.pose == previous.right.pose
            assert target.right.controls == Controls()
        highest = max(highest, target.right.pose.position[2])
        previous = target
    assert highest > rest.right.pose.position[2] + 0.2
    assert target.right.pose != rest.right.pose


@pytest.mark.parametrize(
    "goal,elapsed",
    [
        (Goal(skill="WAIT", duration_s=1), 0.1),
        (Goal(skill="LOOK_AT", target="gone", duration_s=1), 2),
    ],
)
def test_hold_uses_observed_full_body_and_does_not_require_expired_target(goal, elapsed):
    from myumiq_vrchat.postures import posture_target

    observed = posture_target("lying")
    motor = ProceduralMotor(posture_target("standing"))
    assert (
        motor.step(
            simulated_body(observed, 10),
            MotionCommand(goal=goal, elapsed_s=elapsed),
            WorldState(),
            0.02,
        )
        == observed
    )


def test_look_and_reach_use_explicit_geometry_without_root_motion():
    rest = rest_target()
    body = simulated_body(rest, 0.0)
    world = WorldState(
        objects=(WorldObject(name="point", position=(0.3, 0.3, 1.4), source="fixture"),)
    )
    for skill, hand in (("LOOK_AT", None), ("REACH", "left")):
        motor = ProceduralMotor()
        goal = Goal(skill=skill, hand=hand, target="point", duration_s=3.0)
        for n in range(120):
            result = motor.step(body, MotionCommand(goal=goal, elapsed_s=n / 60), world, 1 / 60)
        assert result.head.position == rest.head.position
        if skill == "LOOK_AT":
            assert result.head.orientation != rest.head.orientation
        else:
            assert result.left.pose.position == world.locate("point")
    too_far = WorldState(
        objects=(WorldObject(name="point", position=(9.0, 9.0, 1.4), source="fixture"),)
    )
    with pytest.raises(ValueError, match="workspace"):
        motor.step(body, MotionCommand(goal=goal, elapsed_s=0.0), too_far, 1 / 60)
