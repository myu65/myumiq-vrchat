"""Shared bounded reference evaluation; no live device output."""

import numpy as np

from .articulated_env import ArticulatedGoalEnv
from .joint_limits import violation
from .tracker_policy import distance, pose_error
from .whole_body import vector


def evaluate(
    model,
    rig,
    decoder,
    states,
    *,
    dt=0.05,
    record=False,
    oracle=False,
    hold=False,
    projected_oracle=False,
    reference_floor=None,
    floor_weight=20.0,
    floor_power=2,
    worst_tracker_weight=0.0,
    settle_on_goal=False,
):
    steps = round((20 if hold else 5) / dt)
    valid_starts = [
        s for s in states if float(violation(s.rotations, rig.joint_limits).max()) <= 1e-6
    ]
    if not valid_starts:
        raise ValueError("no feasible held-out start for this joint envelope")
    env = ArticulatedGoalEnv(
        rig,
        states,
        decoder=None if oracle and not projected_oracle else decoder,
        dt=dt,
        horizon=min(steps, 1000),
        recording=record,
        reference_floor=reference_floor,
        floor_weight=floor_weight,
        floor_power=floor_power,
        worst_tracker_weight=worst_tracker_weight,
    )
    trials = []
    for seed in range(6):
        options = (
            {"start": valid_starts[seed % len(valid_starts)], "goal": states[seed % len(states)]}
            if rig.joint_limits
            else None
        )
        obs, _ = env.reset(seed=8000 + seed, options=options)
        held = env.current
        if hold:
            obs = env.set_goal(env.state)
        initial = distance(held, env.goal)
        endpoints, max_drift, max_rate = [], 0.0, 0.0
        min_foot_height = float("inf")
        max_length_change = 0.0
        stable, settled = 0, False
        pairs = ((3, 5), (4, 6), (7, 9), (8, 10))
        initial_points = vector(held)[:, :3]
        lengths = np.array(
            [np.linalg.norm(initial_points[a] - initial_points[b]) for a, b in pairs]
        )
        for step in range(steps):
            if not hold and step == steps // 2:
                previous = env.state
                obs = env.set_goal(states[(seed + 3) % len(states)])
                assert env.state is previous
                stable, settled = 0, False
            if settle_on_goal:
                error = pose_error(env.current, env.goal)
                within = (
                    np.linalg.norm(error[:, :3], axis=1).max() <= 0.12
                    and np.linalg.norm(error[:, 3:], axis=1).max() <= 0.35
                )
                stable = stable + 1 if within else 0
                settled |= stable >= max(3, int(np.ceil(0.15 / dt)) + 1)
            if settled:
                action = np.zeros(env.action_space.shape)
            elif oracle:
                action = 2 * rig.rates_between(env.state, env.goal_state, 1.0)
                action /= max(1.0, float(np.linalg.norm(action.reshape(-1, 3), axis=1).max()))
                if projected_oracle:
                    action = action @ np.asarray(decoder.rows).T
                    action /= max(1.0, float(np.abs(action).max()))
            elif model is not None:
                action = model.predict(obs, deterministic=True)[0]
            else:
                action = np.zeros(env.action_space.shape)
            obs, _, _, _, info = env.step(action)
            max_drift = max(max_drift, distance(held, env.current))
            points = vector(env.current)[:, :3]
            min_foot_height = min(min_foot_height, float(points[9:, 2].min()))
            actual_lengths = np.array([np.linalg.norm(points[a] - points[b]) for a, b in pairs])
            max_length_change = max(
                max_length_change, float(np.abs(actual_lengths - lengths).max())
            )
            max_rate = max(
                max_rate, float(np.linalg.norm(env.previous.reshape(-1, 3), axis=1).max())
            )
            if step == steps - 1 or (not hold and step == steps // 2 - 1):
                endpoints.append(
                    {
                        "pose_error": info["pose_error"],
                        "settled": settled,
                        "hold_baseline_error": distance(held, env.goal),
                        "target": env.current.model_dump(mode="json"),
                        "goal": env.goal.model_dump(mode="json"),
                    }
                )
        trials.append(
            {
                "seed": seed,
                "initial_error": initial,
                "endpoints": endpoints,
                "maximum_drift": max_drift,
                "maximum_physical_rate_norm": max_rate,
                "minimum_foot_height_m": min_foot_height,
                "maximum_forearm_shin_length_change_m": max_length_change,
            }
        )
    endpoints = [e for t in trials for e in t["endpoints"]]
    report = {
        "control_contract": "static_goal_settling_v1" if settle_on_goal else "continuous_actor_v1",
        "static_goals_reached": sum(e["settled"] for e in endpoints) if settle_on_goal else None,
        "dt": dt,
        "duration_s": steps * dt,
        "mean_endpoint_error": float(np.mean([e["pose_error"] for e in endpoints])),
        "endpoints_worse_than_hold": sum(
            e["pose_error"] > e["hold_baseline_error"] + 1e-6 for e in endpoints
        ),
        "maximum_drift": max(t["maximum_drift"] for t in trials),
        "minimum_foot_height_m": min(t["minimum_foot_height_m"] for t in trials),
        "maximum_forearm_shin_length_change_m": max(
            t["maximum_forearm_shin_length_change_m"] for t in trials
        ),
        "trials": trials,
        "joint_envelope_unsupported_start_count": len(states) - len(valid_starts),
        "joint_envelope_total_reference_count": len(states),
    }
    env.close()
    return report, env.replay
