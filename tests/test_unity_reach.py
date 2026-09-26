import math

import pytest

from myumiq_vrchat.unity_reach import integrate_reach


def test_reach_bounds_acceleration_speed_and_workspace():
    position, velocity = (0.2, -0.25, 1.15), (0.0, 0.0, 0.0)
    for _ in range(150):
        before = position
        position, new_velocity = integrate_reach(position, velocity, (1.0, 1.0, 1.0))
        assert math.hypot(*new_velocity) <= 0.6 + 1e-9
        assert math.dist(position, (0.05, -0.2, 1.35)) <= 0.65
        assert math.dist(before, position) <= 0.6 / 30 + 1e-9
        if position != before:
            assert math.dist(velocity, new_velocity) <= 1.8 / 30 + 1e-9
        velocity = new_velocity


@pytest.mark.parametrize("action", [(float("nan"), 0, 0), (1.01, 0, 0)])
def test_reach_rejects_invalid_policy_output(action):
    with pytest.raises(ValueError):
        integrate_reach((0.2, -0.25, 1.15), (0.0, 0.0, 0.0), action)


def test_reach_first_step_matches_contract():
    position, velocity = integrate_reach((0.2, -0.25, 1.15), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert velocity == pytest.approx((0.06, 0.0, 0.0))
    assert position == pytest.approx((0.202, -0.25, 1.15))


def test_learned_reach_records_action_and_observation():
    from myumiq_vrchat.body import WorldObject, WorldState, rest_target, simulated_body
    from myumiq_vrchat.cognition import Decision, Goal
    from myumiq_vrchat.replay import Observation, Transition
    from myumiq_vrchat.runtime import BodyAgent

    class Actor:
        def predict(self, observation):
            assert len(observation) == 9
            return (1.0, 0.0, 0.0)

    goal = Goal(skill="REACH", hand="right", target="target", duration_s=1.0)
    agent = BodyAgent(Decision(goal=goal, source="fixed"), 30.0, reach_policy=Actor())
    records = []
    agent.collector = type(
        "Collector",
        (),
        {
            "collect": lambda self, value: records.append(Transition.model_validate_json(value)),
        },
    )()
    world = WorldState(
        objects=(
            WorldObject(
                name="target",
                position=(0.4, -0.25, 1.2),
                source="fixture",
            ),
        )
    )
    target = rest_target()
    for n in range(3):
        target = agent.step(
            Observation(
                timestamp=n / 30,
                world=world,
                body=simulated_body(target, n / 30),
            )
        )
    record = records[-1]
    assert record.environment == "mock"
    assert record.motor_action == (1.0, 0.0, 0.0)
    assert len(record.motor_observation) == len(record.next_motor_observation) == 9
    assert target.right.pose.position[0] > 0.2
