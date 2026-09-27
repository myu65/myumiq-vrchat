"""Evaluate posture transitions with the actual actor, without live device output."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .articulated_body import JointState, articulated_observation
from .articulated_fit import fit_body
from .tracker_policy import pose_error
from .whole_body import vector


def transition(actor, start, goal, *, duration_s=12.0, dt=0.05, floor=0.0, criterion=None):
    """Ideal-actuator admission diagnostic, not avatar/contact/task observation.

    Check every tracker against the plane, not only the feet: head and hands can
    be the lowest points in a lying posture. Preserve the initial fitting residual
    as the live actor does. End with a hold after the normal settling criterion.
    """
    from .tracker_action import integrate_tracker_action

    if not 1 <= duration_s <= 30 or not 0.01 <= dt <= 0.1 or not np.isfinite(floor):
        raise ValueError("invalid posture evaluation limits")
    if vector(goal)[:, 2].min() < floor:
        raise ValueError("posture goal crosses the declared tracker plane")
    rig = actor.rig
    prior = JointState(
        np.asarray(start.pelvis.position), np.tile([1.0, 0, 0, 0], (len(rig.names), 1))
    )
    state, fit = fit_body(rig, start, prior)
    current, previous = start, np.zeros(66)
    minimum = float(vector(start)[:, 2].min())
    settled_at, stable = None, 0
    floor_failure = minimum < floor
    maximum_step = 0.0
    for index in range(round(duration_s / dt)):
        error = pose_error(current, goal)
        within = bool(
            np.linalg.norm(error[:, :3], axis=1).max() <= 0.12
            and np.linalg.norm(error[:, 3:], axis=1).max() <= 0.35
        )
        if criterion is not None:
            within = within and criterion(current)["success"]
        stable = stable + 1 if within else 0
        if stable >= max(3, int(np.ceil(0.15 / dt)) + 1):
            settled_at = index * dt
            break
        if floor_failure:
            break
        obs = articulated_observation(current, goal, previous, dt, state.rotations)
        latent = actor.predict(obs)[0]
        following, _, rates, _ = rig.advance(state, actor.decoder.decode(latent), dt)
        target = integrate_tracker_action(current, rates.reshape(11, 6), dt)
        minimum = min(minimum, float(vector(target)[:, 2].min()))
        maximum_step = max(
            maximum_step,
            float(np.linalg.norm(vector(target)[:, :3] - vector(current)[:, :3], axis=1).max()),
        )
        floor_failure = minimum < floor
        if floor_failure:
            break  # An invalid proposal is not applied, just as in the live controller.
        state, current, previous = following, target, rates
    error = pose_error(current, goal)
    return {
        "scope": "ideal_actuator_fitted_start_not_avatar_or_contact",
        "accepted": settled_at is not None and not floor_failure,
        "settled_at_s": settled_at,
        "minimum_proposed_tracker_height_m": minimum,
        "floor_failure": floor_failure,
        "maximum_step_m": maximum_step,
        "maximum_position_error_m": float(np.linalg.norm(error[:, :3], axis=1).max()),
        "maximum_rotation_error_rad": float(np.linalg.norm(error[:, 3:], axis=1).max()),
        "fit": fit,
        "endpoint": current.model_dump(mode="json"),
    }


def main():
    from .articulated_actor import ArticulatedActor
    from .articulated_tasks import ArticulatedTasks
    from .cli import outside_repo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--goals", nargs="+", required=True)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    task_bytes = outside_repo(args.tasks).read_bytes()
    settings = ArticulatedTasks.model_validate_json(task_bytes)
    if not 2 <= len(set(args.goals)) <= 8 or set(args.goals) - settings.goals.keys():
        raise ValueError("supply two to eight different configured posture goals")
    actor = ArticulatedActor(outside_repo(settings.actor))
    if (
        actor.manifest["sha256"] != settings.actor_sha256
        or hashlib.sha256(actor.rig.model_dump_json().encode()).hexdigest() != settings.rig_sha256
        or actor.manifest.get("reference_floor") not in (None, settings.reference_floor)
    ):
        raise ValueError("task actor, rig or floor differs from evaluated configuration")
    output = outside_repo(args.out)
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for start in dict.fromkeys(args.goals):
        for goal in dict.fromkeys(args.goals):
            try:
                report = transition(
                    actor,
                    settings.goals[start],
                    settings.goals[goal],
                    duration_s=args.duration_s,
                    floor=settings.reference_floor,
                )
            except (ValueError, RuntimeError) as exc:
                report = {"accepted": False, "error": str(exc)}
            reports.append({"start": start, "goal": goal, **report})
            print(
                json.dumps({k: v for k, v in reports[-1].items() if k not in ("endpoint", "fit")}),
                flush=True,
            )
    (output / "result.json").write_text(
        json.dumps(
            {
                "actor_sha256": settings.actor_sha256,
                "tasks_sha256": hashlib.sha256(task_bytes).hexdigest(),
                "rig_sha256": settings.rig_sha256,
                "reference_floor": settings.reference_floor,
                "joint_constraint_contract": actor.manifest.get("joint_constraint_contract"),
                "control_contract": "ideal_actuator_static_goal_settling_all_tracker_floor_v1",
                "duration_s": args.duration_s,
                "all_accepted": all(row["accepted"] for row in reports),
                "avatar_verified": False,
                "cases": reports,
            },
            indent=2,
        ),
        "utf-8",
    )
    if not all(row["accepted"] for row in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
