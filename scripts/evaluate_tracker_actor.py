"""Audit an exported candidate's reach, posture distortion and hold drift, offline."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from myumiq_vrchat.body import BodyTarget
from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.tracker_env import TrackerGoalEnv
from myumiq_vrchat.tracker_policy import SEGMENTS, TrackerActor, distance, pose_error
from myumiq_vrchat.whole_body import vector


def segment_lengths(pose):
    points = vector(pose)[:, :3]
    return np.array([np.linalg.norm(points[a] - points[b]) for a, b in SEGMENTS])


def rollout(actor, poses, low, high, *, dt, seed, hold=False):
    # Goal-switch trials last five seconds; already-at-goal trials last twenty.
    steps = round((20 if hold else 5) / dt)
    env = TrackerGoalEnv(poses, dt=dt, horizon=min(1000, steps))
    obs, _ = env.reset(seed=8000 + seed)
    start = env.current
    if hold:
        obs = env.set_goal(start)
    initial = distance(start, env.goal)
    initial_goal = env.goal
    endpoints, latencies, outside, foot_heights = [], [], [], []
    max_drift = 0.0
    error = None
    try:
        for step in range(steps):
            if not hold and step == steps // 2:
                obs = env.set_goal(poses[(seed + 3) % len(poses)])
            began = time.perf_counter()
            rates = actor.predict(obs)
            latencies.append(time.perf_counter() - began)
            obs, _, _, _, info = env.step(rates)
            lengths = segment_lengths(env.current)
            outside.append(float(np.maximum(0, np.maximum(low - lengths, lengths - high)).max()))
            foot_heights.append(float(vector(env.current)[9:, 2].min()))
            max_drift = max(max_drift, distance(start, env.current))
            if step == steps - 1 or (not hold and step == steps // 2 - 1):
                errors = pose_error(env.current, env.goal)
                endpoints.append(
                    {
                        "weighted_error": info["pose_error"],
                        "hold_baseline_error": distance(start, env.goal),
                        "mean_position_m": float(np.linalg.norm(errors[:, :3], axis=1).mean()),
                        "max_position_m": float(np.linalg.norm(errors[:, :3], axis=1).max()),
                        "mean_rotation_deg": float(
                            np.degrees(np.linalg.norm(errors[:, 3:], axis=1)).mean()
                        ),
                        "pose": env.current.model_dump(mode="json"),
                        "goal": env.goal.model_dump(mode="json"),
                    }
                )
    except (ValueError, RuntimeError) as exc:
        # A failed rollout is evidence, never silently omitted from its summary.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    return {
        "dt": dt,
        "seed": seed,
        "kind": "already_at_goal" if hold else "goal_switch",
        "initial_error": initial,
        "initial_pose": start.model_dump(mode="json"),
        "initial_goal": initial_goal.model_dump(mode="json"),
        "elapsed_simulated_s": env.steps * dt,
        "error": error,
        "endpoints": endpoints,
        "max_segment_outside_reference_m": max(outside, default=None),
        "minimum_foot_height_m": min(foot_heights, default=None),
        "maximum_drift_from_start": max_drift,
        "inference_p95_ms": float(np.percentile(latencies, 95) * 1000) if latencies else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument(
        "--poses", type=Path, required=True, help="training run's split pose manifest"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = outside_repo(args.output)
    if output.exists():
        raise ValueError("audit output must be new")
    torch.set_num_threads(1)
    data = json.loads(args.poses.read_text("utf-8"))
    reference = [BodyTarget.model_validate_json(json.dumps(row["pose"])) for row in data]
    heldout = [
        BodyTarget.model_validate_json(json.dumps(row["pose"]))
        for row in data
        if row["split"] == "heldout"
    ]
    if len(heldout) < 2:
        raise ValueError("at least two withheld poses are required")
    actor = TrackerActor(args.actor)
    lengths = np.stack([segment_lengths(pose) for pose in reference])
    # Heldout references inform the diagnostic envelope only, never policy fitting.
    low, high = lengths.min(axis=0), lengths.max(axis=0)
    trials, summary = [], []
    for dt in (0.01, 0.02, 0.05, 0.1):
        rows = [rollout(actor, heldout, low, high, dt=dt, seed=seed) for seed in range(6)]
        endpoints = [e for row in rows for e in row["endpoints"]]
        trials.extend(rows)
        summary.append(
            {
                "dt": dt,
                "failed_trials": sum(row["error"] is not None for row in rows),
                "completed_endpoints": len(endpoints),
                "mean_endpoint_error": float(np.mean([e["weighted_error"] for e in endpoints]))
                if endpoints
                else None,
                "mean_hold_baseline_error": float(
                    np.mean([e["hold_baseline_error"] for e in endpoints])
                )
                if endpoints
                else None,
                "endpoints_worse_than_hold": sum(
                    e["weighted_error"] > e["hold_baseline_error"] + 1e-6 for e in endpoints
                ),
                "maximum_segment_outside_reference_m": max(
                    (
                        r["max_segment_outside_reference_m"]
                        for r in rows
                        if r["max_segment_outside_reference_m"] is not None
                    ),
                    default=None,
                ),
                "maximum_position_error_m": max(
                    (e["max_position_m"] for e in endpoints), default=None
                ),
            }
        )
    # A zero-rate actuator holds indefinitely. The learned actor must be checked separately.
    holds = [rollout(actor, heldout, low, high, dt=0.02, seed=seed, hold=True) for seed in range(6)]
    result = {
        "scope": "ideal_tracker_geometry_not_avatar",
        "actor_sha256": hashlib.sha256(args.actor.read_bytes()).hexdigest(),
        "pose_manifest_sha256": hashlib.sha256(args.poses.read_bytes()).hexdigest(),
        "decoder_id": actor.manifest.get("decoder_id"),
        "error_units": "mean_position_m + 0.15 * mean_rotation_rad",
        "segment_envelope": "min/max tracker-pair distances in supplied reference poses; not bones",
        "summary": summary,
        "hold_summary": {
            "duration_s": 20,
            "dt": 0.02,
            "failed_trials": sum(r["error"] is not None for r in holds),
            "maximum_pose_drift": max(r["maximum_drift_from_start"] for r in holds),
        },
        "goal_switch_trials": trials,
        "hold_trials": holds,
        "promoted": False,
        "avatar_verified": False,
        "world_motion_verified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), "utf-8")
    print(json.dumps({"summary": summary, "hold_summary": result["hold_summary"]}, indent=2))


if __name__ == "__main__":
    main()
