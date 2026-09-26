import pytest

from myumiq_vrchat.body import WorldState, rest_target, simulated_body
from myumiq_vrchat.cognition import Decision, Goal
from myumiq_vrchat.motor import MotionCommand
from myumiq_vrchat.reach_replay import reach_sample
from myumiq_vrchat.replay import Observation, Transition


def record(**changes):
    observation = Observation(
        timestamp=0.0, world=WorldState(), body=simulated_body(rest_target(), 0)
    )
    goal = Goal(skill="REACH", target="target", hand="right", duration_s=5)
    data = dict(
        observation=observation,
        next_observation=observation,
        decision=Decision(goal=goal, source="fixed"),
        command=MotionCommand(goal=goal, elapsed_s=0),
        action=rest_target(),
        reward=0.1,
        outcome="simulated",
        environment="unity",
        motor_contract="myumiq-reach-v1",
        reward_kind="reach_task_v1",
        motor_observation=(0.0,) * 9,
        motor_action=(0.0,) * 3,
        next_motor_observation=(0.0,) * 9,
        terminated=False,
        truncated=False,
    )
    data.update(changes)
    return Transition(**data)


def test_time_limit_preserves_bootstrapping_but_success_terminates():
    success = reach_sample(record(terminated=True))
    timeout = reach_sample(record(truncated=True))
    assert success[4] and not success[5]["TimeLimit.truncated"]
    assert timeout[4] and timeout[5]["TimeLimit.truncated"]


@pytest.mark.parametrize(
    "changes",
    [
        {"motor_contract": None},
        {"terminated": None},
        {"truncated": None},
        {"reward_kind": "device_progress"},
        {"environment": "mock"},
        {"environment": "vrchat", "outcome": "device_feedback"},
        {"motor_action": (2.0, 0.0, 0.0)},
        {"terminated": True, "truncated": True},
    ],
)
def test_incomplete_or_incompatible_replay_is_not_training_data(changes):
    with pytest.raises(ValueError):
        reach_sample(record(**changes))
