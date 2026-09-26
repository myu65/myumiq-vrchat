"""Optimize a whole-body actor through verified ideal kinematics; no live promotion."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from myumiq_vrchat.articulated_body import ArticulatedRig, JointState
from myumiq_vrchat.articulated_checkpoint import save_training_state as save_training_state
from myumiq_vrchat.articulated_dynamics import (
    DifferentiableRig,
    blended_goal,
    increment,
    observation,
    policy_rollout_start,
    reward_components,
    turned_goal,
)
from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
from myumiq_vrchat.articulated_evaluation import evaluate
from myumiq_vrchat.articulated_policy import (
    ArticulatedActor,
    export_articulated,
    policy_for_frame,
)
from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.tracker_basis import RateBasis


def save_checkpoint(model, optimizer, rig, decoder, output, source_hash, settings):
    directory = output/f'checkpoint-{model.model_based_updates:06d}'
    directory.mkdir(exist_ok=False)
    export_articulated(model, directory/'candidate-actor.pt', rig, decoder)
    save_training_state(model, optimizer, directory/'training-state.pt')
    metadata = {'kind': 'articulated_model_based_checkpoint_v1', 'source_sha256': source_hash,
                'model_based_updates': model.model_based_updates,
                'bc_updates': int(getattr(model, 'prior_updates', 0)),
                'settings': settings, 'evaluated': False, 'promoted': False,
                'sampling_restart_on_resume': True}
    (directory/'checkpoint.json').write_text(json.dumps(metadata, indent=2), 'utf-8')
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="licensed articulated pose corpus"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument(
        "--resume-source", type=Path, help="continue an explicit compatible candidate"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--floor-weight", type=float, default=20.0)
    parser.add_argument("--floor-power", type=int, choices=(1, 2), default=2)
    parser.add_argument("--worst-tracker-weight", type=float, default=0.0,
                        help="add the largest normalized tracker error to the mean objective")
    parser.add_argument("--rollout-start-steps", type=int, default=0,
                        help="maximum policy rollout steps before the differentiable horizon")
    parser.add_argument("--rollout-switch-probability", type=float, default=0.0,
                        help="probability of following another training goal before goal replacement")
    parser.add_argument(
        "--goal-augmentation",
        action="store_true",
        help="mix source goals with joint-space blends from training poses only",
    )
    parser.add_argument('--heading-goal-range', type=float, default=0.,
                        help='optional maximum whole-body goal yaw change in radians')
    args = parser.parse_args()
    if not 100 <= args.updates <= 10000 or not 1 <= args.horizon <= 16:
        raise ValueError("invalid model-based training budget")
    if not 0 < args.learning_rate <= 0.01:
        raise ValueError("invalid learning rate")
    if not 0 <= args.heading_goal_range <= np.pi:
        raise ValueError('invalid heading goal augmentation')
    if not 0 <= args.rollout_start_steps <= 100:
        raise ValueError("invalid rollout-start step bound")
    if not 0 <= args.rollout_switch_probability <= 1 or (
        args.rollout_switch_probability and not args.rollout_start_steps
    ):
        raise ValueError("goal switching requires rollout initialization and probability in [0,1]")
    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source_report = json.loads((args.source / "result.json").read_text("utf-8"))
    if source_report.get("license") != "CC0":
        raise ValueError("verified CC0 source corpus required")
    torch.set_num_threads(1)
    torch.manual_seed(43)
    rng = np.random.default_rng(43)
    rig = ArticulatedRig.model_validate_json((args.source / "rig.json").read_text("utf-8"))
    source_hash = hashlib.sha256((args.source/'poses.json').read_bytes()).hexdigest()
    rows = json.loads((args.source / "poses.json").read_text("utf-8"))
    groups = {}
    for split in ("train", "heldout"):
        groups[split] = [
            JointState(
                np.asarray(r["joint_state"]["root"]),
                np.asarray(r["joint_state"]["local_orientations"]),
            )
            for r in rows
            if r["split"] == split
        ]
    train, withheld = groups["train"], groups["heldout"]
    # Full-rank coordinates; no lost directions from the earlier truncated basis.
    decoder = RateBasis(
        action_size=rig.action_size,
        rows=tuple(tuple(float(x) for x in row) for row in np.eye(rig.action_size)),
        variance_fraction=1.0,
    )
    objective_settings = dict(
        reference_floor=0.0, floor_weight=args.floor_weight, floor_power=args.floor_power,
        worst_tracker_weight=args.worst_tracker_weight,
    )
    env = ArticulatedGoalEnv(rig, train, decoder=decoder, **objective_settings)
    prior = ArticulatedActor(args.resume_source / "candidate-actor.pt") if args.resume_source else None
    frame = prior.manifest.get('policy_frame', 'tracking') if prior else 'tracking'
    policy, frame_kwargs = policy_for_frame(frame, rig)
    architecture = prior.manifest.get('net_arch', [128, 128]) if prior else [128, 128]
    model = SAC(
        policy,
        env,
        device="cpu",
        seed=43,
        buffer_size=10,
        policy_kwargs={"net_arch": architecture, **frame_kwargs},
        verbose=0,
    )
    model.algorithm_label = "short-horizon differentiable ideal-actuator policy optimization"
    model.model_based_updates = 0
    model.reference_floor = 0.0
    model.floor_weight, model.floor_power = args.floor_weight, args.floor_power
    model.worst_tracker_weight = args.worst_tracker_weight
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.learning_rate)
    if args.resume_source:
        if prior.rig != rig or prior.decoder != decoder:
            raise ValueError("resume candidate has different rig or decoder")
        checkpoint = args.resume_source/'checkpoint.json'
        if checkpoint.exists():
            metadata = json.loads(checkpoint.read_text('utf-8'))
            if (metadata.get('kind') != 'articulated_model_based_checkpoint_v1'
                    or metadata.get('source_sha256') != source_hash):
                raise ValueError('checkpoint source corpus differs')
        state = torch.load(
            args.resume_source / "training-state.pt", map_location="cpu", weights_only=True
        )
        model.actor.load_state_dict(state["actor"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        model.model_based_updates = int(state["model_based_updates"])
        model.prior_updates = int(state.get('prior_updates', 0))
        if model.model_based_updates != prior.manifest["model_based_updates"]:
            raise ValueError("resume update counters differ")
        samples = np.random.default_rng(57).uniform(-1, 1, (8, 290)).astype(np.float32)
        for sample in samples:
            np.testing.assert_allclose(
                model.predict(sample, deterministic=True)[0], prior.predict(sample)[0], atol=1e-6
            )
    initial_updates = model.model_based_updates
    dynamics = DifferentiableRig(rig).float()
    roots = torch.tensor(np.stack([s.root for s in train]), dtype=torch.float32)
    joints = torch.tensor(np.stack([s.rotations for s in train]), dtype=torch.float32)
    with torch.no_grad():
        goals = dynamics(roots, joints)
    initial, _ = evaluate(model, rig, decoder, withheld, **objective_settings)
    hold, _ = evaluate(None, rig, decoder, withheld, **objective_settings)
    began = time.monotonic()
    log = []
    checkpoint_settings = {key: getattr(args, key) for key in (
        'horizon', 'learning_rate', 'floor_weight', 'floor_power', 'worst_tracker_weight',
        'rollout_start_steps', 'rollout_switch_probability', 'goal_augmentation', 'heading_goal_range')}
    switched_samples = 0
    for update in range(args.updates):
        size = 64
        indices = torch.randint(len(train), (size,))
        desired = torch.randint(len(train), (size,))
        root, q = roots[indices].clone(), joints[indices].clone()
        goal = goals[desired]
        with torch.no_grad():
            if args.goal_augmentation:
                other = torch.randint(len(train), (size,))
                blend = torch.rand(size, 1)
                augmented = blended_goal(
                    dynamics, roots[desired], joints[desired], roots[other], joints[other], blend
                )
                goal = torch.where((torch.rand(size, 1, 1) < 0.5), augmented, goal)
            # Sample near-goal starts as well as entire posture transitions.
            fraction = torch.rand(size, 1) * (torch.rand(size, 1) < 0.3)
            root = root * (1 - fraction) + roots[desired] * fraction
            goal_q = joints[desired] * torch.where(
                (q * joints[desired]).sum(-1, keepdim=True) < 0, -1.0, 1.0
            )
            q = torch.nn.functional.normalize(
                q * (1 - fraction[:, :, None]) + goal_q * fraction[:, :, None], dim=-1
            )
            q = dynamics.envelope(q)
            angles = (torch.rand(size, len(rig.names)) * 2 - 1) * torch.pi
            angles *= torch.rand(size, 1) < 0.5
            q = dynamics.equivalent_joints(q, angles)
            perturb = torch.rand(size, 1, 1) < 0.25
            q = increment(q, torch.randn_like(q[..., 1:]) * 0.12 * perturb, 1.0)
            q = dynamics.envelope(q)
            root += torch.randn_like(root) * 0.025 * perturb.squeeze(-1)
            low = dynamics(root, q)[:, 9:, 2].amin(1)
            root[:, 2] += torch.relu(0.05 - low)  # sampling valid starts, never a live correction
            if args.heading_goal_range:
                pure_turn = torch.rand(size, 1, 1) < .25
                goal = torch.where(pure_turn, dynamics(root, q), goal)
                angles = (torch.rand(size)*2-1)*args.heading_goal_range
                angles *= (pure_turn[:, 0, 0] | (torch.rand(size) < .5))
                goal = turned_goal(goal, angles)
        previous = torch.zeros(size, 66)
        dt = float(rng.choice([0.02, 0.05, 0.1]))
        if args.rollout_start_steps:
            rollout_goal, replaced = goal, None
            if args.rollout_switch_probability:
                replaced = torch.rand(size) < args.rollout_switch_probability
                prior_goal = goals[torch.randint(len(train), (size,))]
                rollout_goal = torch.where(replaced[:, None, None], prior_goal, goal)
                switched_samples += int(replaced.sum())
            root, q, previous = policy_rollout_start(
                model.actor, dynamics, root, q, rollout_goal, previous, dt,
                int(rng.integers(args.rollout_start_steps + 1)), model.reference_floor,
                reset_previous=replaced)
        objective = root.new_zeros(size)
        optimizer.zero_grad()
        for step in range(args.horizon):
            before = dynamics(root, q)
            obs = observation(before, goal, previous, dt, q)
            action = model.actor(obs, deterministic=True)
            root, q, after, rates, _ = dynamics.step(root, q, action, dt)
            components = reward_components(
                before, after, goal, rates, previous, **objective_settings
            )
            objective = objective + (0.99**step) * sum(components.values())
            previous = rates
        loss = -objective.mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite model-based objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.actor.parameters(), 10.0, error_if_nonfinite=True
        )
        optimizer.step()
        model.model_based_updates += 1
        if (update + 1) % 100 == 0:
            checkpoint = save_checkpoint(model, optimizer, rig, decoder, output, source_hash,
                                         checkpoint_settings)
            progress = {
                "updates": update + 1,
                "total_model_based_updates": model.model_based_updates,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "elapsed_s": time.monotonic() - began,
                "rollout_switched_samples": switched_samples,
                "checkpoint": str(checkpoint.resolve()),
            }
            log.append(progress)
            (output / "progress.json").write_text(json.dumps(progress), "utf-8")
    export_articulated(model, output / "candidate-actor.pt", rig, decoder)
    save_training_state(model, optimizer, output/'training-state.pt')
    actor = ArticulatedActor(output / "candidate-actor.pt")
    candidate, replay = evaluate(actor, rig, decoder, withheld, **objective_settings, record=True)
    replay.save_state(output / "evaluation-experience.jsonl")
    faster, _ = evaluate(actor, rig, decoder, withheld, dt=0.02, **objective_settings)
    stationary, _ = evaluate(
        actor, rig, decoder, withheld, dt=0.02, hold=True, **objective_settings
    )
    result = {
        "scope": "ideal_articulated_tracker_actuator_not_avatar",
        "algorithm": model.algorithm_label,
        "model_based_updates": model.model_based_updates,
        "updates_this_run": model.model_based_updates - initial_updates,
        "resume_source": str(args.resume_source.resolve()) if args.resume_source else None,
        "sampling_seed": 43,
        "sampling_restart_on_resume": bool(args.resume_source),
        "goal_augmentation": args.goal_augmentation,
        "heading_goal_range": args.heading_goal_range,
        "rollout_start_steps": args.rollout_start_steps,
        "rollout_switch_probability": args.rollout_switch_probability,
        "rollout_switched_samples": switched_samples,
        "learning_rate": args.learning_rate,
        "sac_updates": 0,
        "bc_updates": int(getattr(model, 'prior_updates', 0)),
        "policy_frame": frame,
        "net_arch": architecture,
        "batch_size": 64,
        "horizon": args.horizon,
        "elapsed_s": time.monotonic() - began,
        "train_poses": len(train),
        "heldout_poses": len(withheld),
        "source": str(args.source.resolve()),
        "source_manifest_sha256": source_hash,
        "augmentation": "training poses, near-goal interpolation, equivalent hidden twists and joint perturbations",
        "reference_floor": 0.0,
        "reference_floor_weight": args.floor_weight,
        "reference_floor_power": args.floor_power,
        "worst_tracker_weight": args.worst_tracker_weight,
        "hold_baseline": hold,
        "initial_candidate" if args.resume_source else "untrained": initial,
        "candidate": candidate,
        "faster_timestep": faster,
        "already_at_goal": stationary,
        "training_log": log,
        "promoted": False,
        "avatar_verified": False,
        "world_motion_verified": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), "utf-8")
    print(
        json.dumps(
            {
                key: result[key]["mean_endpoint_error"]
                for key in (
                    "hold_baseline",
                    "initial_candidate" if args.resume_source else "untrained",
                    "candidate",
                    "faster_timestep",
                    "already_at_goal",
                )
            }
        ),
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
