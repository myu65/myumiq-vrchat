"""Device evidence for bounded practice; never an avatar-naturalness oracle."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .body import BodyTarget
from .body_conditions import resolve_conditions
from .condition_validation import ConditionCase
from .motion_quality import trajectory_quality
from .replay import WholeBodyTransition
from .whole_body import PARTS, state_target, vector


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sampled_quality(times, poses, dt=0.05, max_gap=0.25):
    """Resample irregular observed positions; missing evidence cannot score as smooth."""
    times = np.asarray(times, dtype=float)
    points = np.asarray(poses, dtype=float)
    if (
        len(times) < 4
        or points.shape != (len(times), 11, 3)
        or not np.isfinite(times).all()
        or not np.isfinite(points).all()
        or np.any(np.diff(times) <= 0)
        or np.max(np.diff(times)) > max_gap
        or times[-1] - times[0] < 3 * dt
    ):
        raise ValueError("insufficient continuous fresh device samples for trajectory scoring")
    grid = np.arange(0, times[-1] - times[0] + 1e-9, dt)
    sampled = np.stack(
        [np.interp(grid, times - times[0], col) for col in points.reshape(len(times), -1).T],
        axis=1,
    ).reshape(-1, 11, 3)
    result = trajectory_quality(sampled, dt)
    # Preserve extrema rather than hiding a brief floor crossing in interpolation.
    result["minimum_tracker_height_m"] = float(points[..., 2].min())
    result["maximum_foot_displacement_m"] = float(
        np.linalg.norm(points[:, 9:] - points[0:1, 9:], axis=-1).max()
    )
    return result | {"samples": len(times), "resample_dt_s": dt}


def observed_pose(observation):
    for part in PARTS:
        signal = observation.body.signal_for(part)
        if (
            not signal.valid
            or not signal.connected
            or signal.pose is None
            or signal.source != "openvr_raw"
            or not 0 <= observation.timestamp - signal.timestamp <= 0.25
        ):
            raise ValueError("fresh raw eleven-device observations required")
    return state_target(observation.body)


def collect(report, cases, actor_sha256, floor):
    """Read the adapter's fixed-session replay, checking exact submitted requests."""
    if report.get("controlled_home") is not True or report.get("error"):
        raise ValueError("live adapter did not finish an owned-Home trial")
    if report.get("actor_sha256") != actor_sha256:
        raise ValueError("live actor differs from requested actor")
    probes = report["probes"]
    if [p["case_id"] for p in probes] != [c.id for c in cases]:
        raise ValueError("live probes do not match requested order")
    generations = [p["generation"] for p in probes]
    if len(set(generations)) != len(generations):
        raise ValueError("each live probe requires its own generation")
    replay = Path(report["session"]) / "experience.jsonl"
    groups = {g: [] for g in generations}
    with replay.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            generation = raw.get("intent_metadata", {}).get("id")
            if generation in groups and raw.get("schema_version") == 2:
                groups[generation].append(WholeBodyTransition.model_validate_json(line))
    replay_hash = digest(replay)
    result, starts = [], []
    for case, probe in zip(cases, probes):
        rows = groups[probe["generation"]]
        if not rows:
            raise ValueError("probe has no recorded device feedback")
        observations = {}
        completed = False
        for row in rows:
            meta = row.intent_metadata
            if (
                row.environment != "vrchat"
                or row.outcome != "device_feedback"
                or row.body_goal != case.goal
                or meta.get("source") != "operator_body_goal"
                or meta.get("motor_policy") != "articulated:" + actor_sha256[:16]
            ):
                raise ValueError("probe evidence has a different goal, actor or source")
            completed |= meta.get("execution", {}).get("phase") == "completed"
            for obs in (row.observation, row.next_observation):
                observations[obs.timestamp] = observed_pose(obs)
        times = sorted(observations)
        first, last = observations[times[0]], observations[times[-1]]
        resolved = resolve_conditions(case.goal, first, case.world, times[0])
        measured = resolved.measure(last)
        positions = [vector(observations[t])[:, :3] for t in times]
        try:
            quality, quality_error = sampled_quality(times, positions), None
        except ValueError as exc:
            quality, quality_error = None, str(exc)
        result.append(
            dict(
                id=case.id,
                generation=probe["generation"],
                completed=completed,
                conditions=measured,
                duration_s=times[-1] - times[0],
                start_pose=first.model_dump(mode="json"),
                motion_quality=quality,
                quality_error=quality_error,
                floor_failure=bool(min(p[:, 2].min() for p in positions) < floor),
            )
        )
        if case.split == "train":
            starts.append(
                case.model_copy(
                    update={
                        "id": f"live-{replay_hash[:10]}-{probe['generation']}",
                        "start": None,
                        "start_pose": first,
                        "observed_at": times[0],
                    }
                )
            )
    return {
        "scope": "observed_device_kinematics_not_avatar_or_contact",
        "actor_sha256": actor_sha256,
        "replay": str(replay),
        "replay_sha256": replay_hash,
        "probes": result,
    }, starts


def append_training(cases, starts):
    """Retain the frozen test set and original starts; refuse test-to-train leakage."""
    result = list(cases)

    def signature(case):
        return (case.start, case.start_pose, case.goal, case.world)

    for start in starts:
        if start.split != "train":
            raise ValueError("held-out observations cannot become training examples")
        if any(signature(start) == signature(c) for c in result):
            continue
        if len(result) >= 64:
            break
        result.append(ConditionCase.model_validate_json(start.model_dump_json()))
    return result


def live_admissible(before, after):
    """Match ordered probes; require completion plus no measured trajectory regression."""
    old, new = before["probes"], after["probes"]
    if not old or [p["id"] for p in old] != [p["id"] for p in new]:
        return False
    for a, b in zip(old, new):
        if not a.get("start_pose") or not b.get("start_pose"):
            return False
        starts = [
            vector(BodyTarget.model_validate_json(json.dumps(p["start_pose"]))) for p in (a, b)
        ]
        if (
            np.linalg.norm(starts[0][:, :3] - starts[1][:, :3], axis=-1).max() > 0.03
            or np.max(
                2
                * np.arccos(
                    np.clip(np.abs(np.sum(starts[0][:, 3:] * starts[1][:, 3:], axis=-1)), 0, 1)
                )
            )
            > 0.15
        ):
            return False  # An easier initial pose cannot masquerade as actor improvement.
        if (
            not b["completed"]
            or not b["conditions"]["success"]
            or b["floor_failure"]
            or not a["motion_quality"]
            or not b["motion_quality"]
            or b["duration_s"] > a["duration_s"] * 1.1 + 0.5
        ):
            return False
        for key, allowance in (
            ("maximum_foot_displacement_m", 0.003),
            ("maximum_acceleration_m_s2", 0.2),
            ("maximum_jerk_m_s3", 4.0),
        ):
            if b["motion_quality"][key] > a["motion_quality"][key] * 1.1 + allowance:
                return False
    return True
