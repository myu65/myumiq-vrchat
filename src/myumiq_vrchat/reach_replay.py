"""Strict conversion of intent-bearing PAMIQ records to SAC replay entries."""

from .replay import Transition


def reach_sample(record: Transition) -> tuple:
    if (
        record.motor_contract != "myumiq-reach-v1"
        or record.reward_kind != "reach_task_v1"
        or record.reward is None
        or record.terminated is None
        or record.truncated is None
        or record.decision.goal != record.command.goal
        or record.command.goal.skill != "REACH"
        or record.command.goal.hand != "right"
        or record.environment not in ("unity", "vrchat")
    ):
        raise ValueError("record lacks explicit REACH learning semantics")
    expected = "simulated" if record.environment == "unity" else "visual_feedback"
    if record.outcome != expected:
        raise ValueError("REACH training requires task feedback, not device tracking reward")
    before, action, after = (
        record.motor_observation,
        record.motor_action,
        record.next_motor_observation,
    )
    if (
        before is None
        or len(before) != 9
        or after is None
        or len(after) != 9
        or action is None
        or len(action) != 3
        or any(abs(x) > 2 for x in (*before, *after))
        or any(abs(x) > 1 for x in action)
    ):
        raise ValueError("invalid REACH policy observation/action dimensions or bounds")
    if record.terminated and record.truncated:
        raise ValueError("REACH success and time limit must be distinct")
    return (
        before,
        action,
        record.reward,
        after,
        record.terminated or record.truncated,
        {"TimeLimit.truncated": record.truncated},
    )
