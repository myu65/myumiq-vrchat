import math

import pytest

from myumiq_vrchat.body import WorldState, rest_target, simulated_body
from myumiq_vrchat.cognition import Goal
from myumiq_vrchat.motor import MotionCommand, ProceduralMotor


def test_explicit_rest_returns_wave_to_neutral_without_teleporting():
    motor = ProceduralMotor()
    target = rest_target()
    world = WorldState()
    wave = Goal(skill="WAVE", hand="right", duration_s=2)
    for n in range(60):
        target = motor.step(
            simulated_body(target, n / 60),
            MotionCommand(goal=wave, elapsed_s=n / 60),
            world,
            1 / 60,
        )
    assert math.dist(target.right.pose.position, rest_target().right.pose.position) > 0.2
    rest = Goal(skill="RETURN_TO_REST", duration_s=2)
    for n in range(120):
        previous = target
        target = motor.step(
            simulated_body(target, 1 + n / 60),
            MotionCommand(goal=rest, elapsed_s=n / 60),
            world,
            1 / 60,
        )
        assert (
            math.dist(target.right.pose.position, previous.right.pose.position) <= 0.6 / 60 + 1e-9
        )
    assert target.right.pose == rest_target().right.pose
    assert target.left.pose == rest_target().left.pose
    assert target.head == rest_target().head


@pytest.mark.parametrize("extra", [{"hand": "right"}, {"target": "person"}])
def test_rest_has_no_hand_or_external_target(extra):
    with pytest.raises(ValueError):
        Goal(skill="RETURN_TO_REST", duration_s=1, **extra)
