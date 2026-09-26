"""Bounded private-VR calibration using existing watchdog and PAMIQ replay.

The approved launcher must validate the live configuration and release its neutral
pose owner before starting this program. This does not launch or reconfigure VR.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

from myumiq_vrchat.backends.osc import LiveConfig
from myumiq_vrchat.backends.readback import OpenVRReadback, require_console
from myumiq_vrchat.backends.supervisor import OutputSupervisor
from myumiq_vrchat.body import Pose, qmul
from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.cognition import Decision, Goal
from myumiq_vrchat.gaze import image_target
from myumiq_vrchat.motor import MotionCommand, approach_pose
from myumiq_vrchat.replay import Observation, ReplayBuffer, Transition
from myumiq_vrchat.vision import LiveVisionLoop, TemplateTargetDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-config", type=Path, required=True)
    parser.add_argument("--hmd-serial", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--target", default="calibration-target")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_console()
    config = LiveConfig.model_validate_json(outside_repo(args.live_config).read_text(encoding="utf-8-sig"))
    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=False)
    detector = TemplateTargetDetector(outside_repo(args.template), name=args.target)
    vision = LiveVisionLoop(detector, hz=4)
    buffer = ReplayBuffer(100)
    owner = OutputSupervisor(output / "output-events.jsonl", config)
    sensor = OpenVRReadback(config, args.hmd_serial)
    previous = config.safe_target
    started = time.perf_counter()
    last_tick = started
    error = None

    def settled(yaw, pitch):
        nonlocal previous, last_tick
        desired = Pose(position=config.safe_target.head.position, orientation=qmul(
            (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
            (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0),
        ))
        deadline = time.perf_counter() + 4
        steady = None
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now - started > 180:
                raise TimeoutError("visual calibration total deadline")
            dt = min(0.1, max(0.0001, now - last_tick))
            last_tick = now
            previous = config.safe_target.model_copy(update={
                "head": approach_pose(previous.head, desired, dt, angular_speed=0.4),
            })
            owner.publish(previous)
            body = sensor.observe()
            if all(getattr(body, key).valid for key in ("head", "left", "right")):
                dot = abs(sum(a * b for a, b in zip(body.head.pose.orientation, desired.orientation)))
                angle = 2 * math.acos(min(1.0, dot))
                if angle < 0.005:
                    steady = now if steady is None else steady
                    world = vision.world()
                    try:
                        target = image_target(world, args.target, now)
                    except ValueError:
                        target = None
                    if target is not None and target.last_seen > steady + 0.3:
                        return Observation(timestamp=now, body=body, world=world), previous
                else:
                    steady = None
            elif now - started > 2:
                raise RuntimeError("owned head/controller readback lost")
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))
        raise TimeoutError("could not observe settled head and confident target")

    try:
        vision.start()
        owner.start()
        sensor.start()
        before, _ = settled(0.0, 0.0)
        goal = Goal(skill="LOOK_AT", target=args.target, duration_s=4)
        rng = random.Random(1701)
        for _ in range(40):
            yaw, pitch = rng.uniform(-0.09, 0.09), rng.uniform(-0.07, 0.07)
            after, action = settled(yaw, pitch)
            old = image_target(before.world, args.target, before.timestamp)
            new = image_target(after.world, args.target, after.timestamp)
            reward = math.hypot(*old.image_position) - math.hypot(*new.image_position)
            buffer.add(Transition(
                observation=before, next_observation=after,
                decision=Decision(goal=goal, source="fixed"),
                command=MotionCommand(goal=goal, elapsed_s=after.timestamp-before.timestamp),
                action=action, reward=reward, outcome="visual_feedback",
            ).model_dump_json())
            before = after
        settled(0.0, 0.0)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup_errors = []
        for component in (owner, sensor, vision):
            try:
                (component.stop if component is vision else component.close)()
            except Exception as exc:
                cleanup_errors.append(f"{type(component).__name__}: {exc}")
        buffer.save_state(output / "experience.jsonl")
        (output / "result.json").write_text(json.dumps({
            "transitions": len(buffer), "elapsed_s": time.perf_counter()-started,
            "error": error, "mode": "live_calibration", "learned_policy_updated": False,
            "cleanup_errors": cleanup_errors,
        }, indent=2), encoding="utf-8")
        if cleanup_errors:
            raise RuntimeError(f"calibration cleanup failed: {cleanup_errors}")


if __name__ == "__main__":
    main()
