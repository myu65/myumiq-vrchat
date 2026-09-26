"""Bounded whole-body evaluation through the existing watchdog and PAMIQ replay.

Mock by default. Live requires the existing console, device, private-world and
calibration gates; this script does not launch VRChat or assign tracker roles.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from myumiq_vrchat.backends.osc import LiveConfig
from myumiq_vrchat.backends.readback import OpenVRReadback, require_console
from myumiq_vrchat.backends.supervisor import OutputSupervisor
from myumiq_vrchat.body import BodyGoal, BodyTask, WorldState, simulated_body
from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.replay import Observation, ReplayBuffer, WholeBodyTransition
from myumiq_vrchat.whole_body import (
    PARTS,
    PeriodicImitation,
    WholeBodyPolicy,
    bounded_step,
    state_target,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--policy", type=Path)
    choice.add_argument("--postures", action="store_true")
    parser.add_argument("--live-config", type=Path)
    parser.add_argument("--hmd-serial")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    live = None
    if args.live_config:
        require_console()
        live = LiveConfig.model_validate_json(outside_repo(args.live_config).read_text(encoding="utf-8-sig"))
        if not live.trackers or not args.hmd_serial:
            raise ValueError("full-body ownership and explicit HMD identity are required")
    elif args.hmd_serial:
        raise ValueError("HMD identity without live configuration")
    model = PeriodicImitation.model_validate_json(args.policy.read_text()) if args.policy else None
    policy = WholeBodyPolicy(model) if model else None
    policy_id = hashlib.sha256(args.policy.read_bytes()).hexdigest() if model else "diagnostic-fk-v1"
    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=False)
    owner = OutputSupervisor(output / "output-events.jsonl", live)
    sensor = OpenVRReadback(live, args.hmd_serial) if live else None
    rest = live.safe_target if live else posture_target("standing")
    previous = rest
    replay = ReplayBuffer(2400)
    error = None
    names = ("standing", "crouching", "sitting_floor", "lying", "standing_up")
    goals = [BodyGoal(tasks=(BodyTask(id=name, kind="posture", effectors=PARTS,
                                    posture=name),), duration_s=4.0) for name in names]
    walk_goal = BodyGoal(tasks=(BodyTask(id="imitation", kind="locomotion", effectors=PARTS,
                                       target=model.clip if model else "unused"),), duration_s=20.0)
    started = time.perf_counter()

    def observe():
        now = time.perf_counter()
        body = sensor.observe() if sensor else simulated_body(previous, now)
        return Observation(timestamp=now, body=body, world=WorldState())

    try:
        if sensor:
            sensor.start()
        owner.start()
        # Publish only calibrated safe pose until all owned devices return valid.
        deadline = time.perf_counter() + 2
        while True:
            owner.publish(rest)
            observation = observe()
            if all(observation.body.signal_for(p).valid for p in PARTS):
                break
            if time.perf_counter() > deadline:
                raise RuntimeError("not all eleven owned poses became valid")
            time.sleep(1 / 60)
        # Settle to first learned frame without advancing the clip phase.
        last_tick = time.perf_counter()
        run_start = last_tick
        for frame in range(24 * 60):
            now = time.perf_counter()
            elapsed = now - run_start
            if elapsed >= 24:
                break
            dt = max(1e-4, now - last_tick) if frame else 1 / 60
            if dt > 0.1:
                raise TimeoutError("whole-body producer missed its timestep")
            last_tick = now
            before = observe()
            current = state_target(before.body)
            if model:
                goal = walk_goal
                if elapsed < 2:
                    desired = model.sample(0.0)
                    action = bounded_step(current, desired, dt)
                elif elapsed < 20:
                    action = policy.step(before.body, goal, dt)
                else:
                    goal = goals[-1]
                    action = bounded_step(current, rest, dt)
            else:
                index = min(int(elapsed // 4), 4)
                goal = goals[index]
                name = names[index]
                desired = posture_target("standing" if name == "standing_up" else name)
                # FK fixture has fixed 1.6m morphology, not automatic avatar fitting.
                action = bounded_step(current, desired, dt)
            owner.publish(action)
            previous = action
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))
            after = observe()
            state_target(after.body)  # reject partial feedback before recording success
            replay.add(WholeBodyTransition(
                observation=before, next_observation=after, body_goal=goal,
                action=action, policy_id=policy_id,
                environment="vrchat" if live else "mock",
                outcome="device_feedback" if live else "simulated",
            ).model_dump_json())
            if time.perf_counter() - run_start > 28:
                raise TimeoutError("whole-body experiment deadline")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup = []
        for component in (owner, sensor):
            if component:
                try:
                    component.close()
                except Exception as exc:
                    cleanup.append(str(exc))
        replay.save_state(output / "experience.jsonl")
        (output / "result.json").write_text(json.dumps({
            "transitions": len(replay), "elapsed_s": time.perf_counter()-started,
            "mode": "live" if live else "mock", "error": error,
            "avatar_visual_confirmation": False, "cleanup_errors": cleanup,
        }, indent=2), encoding="utf-8")
        if cleanup:
            raise RuntimeError(f"whole-body cleanup failed: {cleanup}")


if __name__ == "__main__":
    main()
