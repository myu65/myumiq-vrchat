"""Compare exported actor through Virtual Body against Unity held-out goals."""

import argparse
import json
import math
from pathlib import Path

from myumiq_vrchat.body import WorldObject, WorldState, rest_target, simulated_body
from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.cognition import Goal
from myumiq_vrchat.motor import MotionCommand, ProceduralMotor
from myumiq_vrchat.reach_policy import LearnedReachPolicy
from myumiq_vrchat.unity_reach import UnityReachClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = outside_repo(args.output)
    if output.exists():
        raise ValueError("output must be new")
    actor = LearnedReachPolicy(outside_repo(args.policy))
    client = UnityReachClient(outside_repo(args.ready))
    rows = []
    try:
        for seed in range(10000, 10020):
            state = client.reset(seed)
            world = WorldState(objects=(WorldObject(
                name="target", position=state.goal, source="fixture",
            ),))
            goal = Goal(skill="REACH", target="target", hand="right", duration_s=5)
            motor = ProceduralMotor(reach_policy=actor)
            target = rest_target()
            maximum_difference = 0.0
            for step in range(150):
                body = simulated_body(target, step / 30)
                target = motor.step(body, MotionCommand(goal=goal, elapsed_s=step / 30), world, 1 / 30)
                state = client.step(motor.last_motor_action)
                maximum_difference = max(maximum_difference, math.dist(state.hand, target.right.pose.position))
                if state.terminated or state.truncated:
                    break
            row = {"seed": seed, "goal": state.goal, "unity_success": state.success,
                   "motor_unity_max_difference_m": maximum_difference}
            # The live loop usually runs at 60 Hz; evaluate that time-step change
            # in mock without attributing it to Unity or VRChat feedback.
            motor = ProceduralMotor(reach_policy=actor)
            target = rest_target()
            success = False
            for step in range(300):
                previous = target.right.pose.position
                target = motor.step(simulated_body(target, step / 60),
                                    MotionCommand(goal=goal, elapsed_s=step / 60), world, 1 / 60)
                distance = math.dist(state.goal, target.right.pose.position)
                speed = math.dist(previous, target.right.pose.position) * 60
                if distance < 0.025 and speed < 0.12:
                    success = True
                    break
            row.update(mock_60hz_success=success, mock_60hz_distance_m=distance)
            rows.append(row)
    finally:
        client.close()
    result = {"vrchat_transfer_verified": False, "episodes": rows,
              "unity_success_rate": sum(r["unity_success"] for r in rows) / len(rows),
              "mock_60hz_success_rate": sum(r["mock_60hz_success"] for r in rows) / len(rows),
              "maximum_motor_unity_difference_m": max(r["motor_unity_max_difference_m"] for r in rows)}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}))


if __name__ == "__main__":
    main()
