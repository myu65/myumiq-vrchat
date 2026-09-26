"""Learn an offline whole-body prior from training-split kinematic teacher rollouts."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from train_articulated_actor import evaluate

from myumiq_vrchat.articulated_actor import ArticulatedActor
from myumiq_vrchat.articulated_body import ArticulatedRig, JointState
from myumiq_vrchat.articulated_dynamics import (
    DifferentiableRig,
    joint_teacher,
    multiply,
    observation,
    rotate,
)
from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
from myumiq_vrchat.articulated_frame import FRAME, JOINT_WORLD_FRAME
from myumiq_vrchat.articulated_policy import export_articulated, policy_for_frame
from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.tracker_basis import RateBasis


def heading(q):
    forward = rotate(q, q.new_tensor([1., 0., 0.]))
    left = rotate(q, q.new_tensor([0., 1., 0.]))
    use_forward = forward[:, :2].square().sum(-1) >= .01
    return torch.atan2(torch.where(use_forward, forward[:, 1], -left[:, 0]),
                       torch.where(use_forward, forward[:, 0], left[:, 1]))


def anchor_state(root, joints, current_root, current_joints):
    angle = heading(current_joints[:, 0])-heading(joints[:, 0])
    zero = torch.zeros_like(angle)
    turn = torch.stack((torch.cos(angle/2), zero, zero, torch.sin(angle/2)), dim=1)
    joints = torch.cat((multiply(turn, joints[:, 0])[:, None], joints[:, 1:]), dim=1)
    return torch.cat((current_root[:, :2], root[:, 2:]), dim=1), joints


@torch.no_grad()
def collect(dynamics, roots, joints, count, rng, progress, *,
            rollin_actor=None, rollin_probability=0., goal_steps=16, heading_goal_range=0.):
    observations, actions = [], []
    accepted = rejected = switches = student_actions = student_rejected = attempted = 0
    batch, steps = 128, max(64, goal_steps*2)
    timesteps = []
    while accepted < count:
        start, finish = torch.randint(len(roots), (2, batch))
        root, q = roots[start].clone(), joints[start].clone()
        goal_root, goal_q = roots[finish].clone(), joints[finish].clone()
        amount = torch.rand(batch, 1)
        root = root*(1-amount)+goal_root*amount
        goal_q = goal_q*torch.where((q*goal_q).sum(-1, keepdim=True)<0, -1., 1.)
        q = torch.nn.functional.normalize(q*(1-amount[:, :, None])+goal_q*amount[:, :, None], dim=-1)
        q = dynamics.envelope(q)
        twists = (torch.rand(batch, q.shape[1])*2-1)*torch.pi
        q = dynamics.equivalent_joints(q, twists)
        # Common tracking XY/yaw varies while the complete posture is preserved.
        yaw = (torch.rand(batch)*2-1)*torch.pi
        zero = torch.zeros_like(yaw)
        turn = torch.stack((torch.cos(yaw/2), zero, zero, torch.sin(yaw/2)), dim=1)
        root = torch.cat(((torch.rand(batch, 2)*2-1), root[:, 2:]), dim=1)
        q = torch.cat((multiply(turn, q[:, 0])[:, None], q[:, 1:]), dim=1)
        floor = dynamics(root, q)[:, 9:, 2].amin(1)
        root[:, 2] += torch.relu(.05-floor)  # Offline valid-start sampling only.
        previous = torch.zeros(batch, 66)
        student = (torch.rand(batch) < rollin_probability if rollin_probability
                   else torch.zeros(batch, dtype=torch.bool))
        for step in range(steps):
            if step % 8 == 0:
                if not timesteps:
                    timesteps = list(rng.permutation([.02, .05, .1]))
                dt = float(timesteps.pop())
            attempted += batch
            if attempted > count*20:
                raise RuntimeError('insufficient valid teacher transitions within collection budget')
            if step % goal_steps == 0:
                desired = torch.randint(len(roots), (batch,))
                goal_q = dynamics.equivalent_joints(joints[desired], twists)
                goal_root, goal_q = anchor_state(roots[desired], goal_q, root, q)
                if heading_goal_range:
                    # A task may request a new whole-body heading, not only a
                    # posture anchored to the heading the body already has.
                    pure_turn = torch.rand(batch) < .25
                    goal_root = torch.where(pure_turn[:, None], root, goal_root)
                    goal_q = torch.where(pure_turn[:, None, None], q, goal_q)
                    angle = (torch.rand(batch) * 2 - 1) * heading_goal_range
                    angle *= pure_turn | (torch.rand(batch) < .5)
                    zeros = torch.zeros_like(angle)
                    yaw = torch.stack((torch.cos(angle/2), zeros, zeros, torch.sin(angle/2)), dim=1)
                    goal_q = torch.cat((multiply(yaw, goal_q[:, 0])[:, None], goal_q[:, 1:]), dim=1)
                goal = dynamics(goal_root, goal_q)
                previous = torch.zeros_like(previous)
                switches += batch
            current = dynamics(root, q)
            obs = observation(current, goal, previous, dt, q)
            action = joint_teacher(root, q, goal_root, goal_q)
            next_root, next_q, after, rates, _ = dynamics.step(root, q, action, dt)
            valid = after[:, 9:, 2].amin(1) >= 0.
            rejected += int((~valid).sum())
            if bool(valid.any()):
                observations.append(obs[valid].clone())
                actions.append(action[valid].clone())
                accepted += int(valid.sum())
            if rollin_actor is not None and bool(student.any()):
                student_action = rollin_actor.model(obs)
                rollin_action = torch.where(student[:, None], student_action, action)
                next_root, next_q, after, rates, _ = dynamics.step(root, q, rollin_action, dt)
                valid = after[:, 9:, 2].amin(1) >= 0.
                student_actions += int(student.sum())
                student_rejected += int((student & ~valid).sum())
            # A rejected whole-body action never becomes an observed teacher transition.
            root = torch.where(valid[:, None], next_root, root)
            q = torch.where(valid[:, None, None], next_q, q)
            previous = torch.where(valid[:, None], rates, torch.zeros_like(rates))
            if accepted >= count:
                break
        progress({'phase': 'collecting', 'accepted_samples': accepted,
                  'rejected_floor_actions': rejected, 'teacher_goal_switches': switches,
                  'student_rollin_actions': student_actions,
                  'rejected_student_floor_actions': student_rejected})
    return torch.cat(observations)[:count], torch.cat(actions)[:count], {
        'accepted_samples': count, 'rejected_floor_actions': rejected, 'teacher_goal_switches': switches,
        'student_rollin_actions': student_actions, 'rejected_student_floor_actions': student_rejected,
        'teacher_goal_steps': goal_steps, 'attempted_samples': attempted, 'collector_version': 4,
        'heading_goal_range': heading_goal_range}


def tracker_loss(dynamics, obs, prediction, labels):
    total = prediction.new_zeros(())
    count = 0
    for dt in (.02, .05, .1):
        selected = torch.isclose(obs[:, 209], obs.new_tensor(dt/.05))
        size = int(selected.sum())
        if not size:
            continue
        # The imported rig's pelvis is its root; collection stores ideal FK observations.
        root = obs[selected, 80:83]
        q = obs[selected, 210:].reshape(size, -1, 4)
        with torch.no_grad():
            _, _, _, expected, _ = dynamics.step(root, q, labels[selected], dt)
        _, _, _, actual, _ = dynamics.step(root, q, prediction[selected], dt)
        total = total + (actual-expected).square().mean()*size
        count += size
    if count != len(obs):
        raise ValueError('unknown teacher timestep')
    return total/count + .01*(prediction-labels).square().mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=131072)
    parser.add_argument('--updates', type=int, default=4000)
    parser.add_argument('--hidden-size', type=int, choices=(128, 256), default=256)
    parser.add_argument('--policy-frame', choices=(FRAME, JOINT_WORLD_FRAME))
    parser.add_argument('--resume-source', type=Path)
    parser.add_argument('--tracker-rate-loss', action='store_true')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--rollin-probability', type=float, default=0.)
    parser.add_argument('--goal-steps', type=int)
    parser.add_argument('--heading-goal-range', type=float, default=0.)
    args = parser.parse_args()
    if not 4096 <= args.samples <= 524288 or not 100 <= args.updates <= 20000:
        raise ValueError('invalid offline prior budget')
    if not 0 < args.learning_rate <= .01:
        raise ValueError('invalid cloning learning rate')
    if not 0 <= args.heading_goal_range <= np.pi:
        raise ValueError('heading goal range must be between zero and pi radians')
    if (not 0 <= args.rollin_probability <= 1 or (args.rollin_probability and not args.resume_source)
            or (args.goal_steps is not None and not 16 <= args.goal_steps <= 128)):
        raise ValueError('invalid dataset aggregation settings')
    output = outside_repo(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source_report = json.loads((args.source/'result.json').read_text('utf-8'))
    if source_report.get('license') != 'CC0':
        raise ValueError('verified CC0 corpus required')
    source_path = args.source/'poses.json'
    rows = json.loads(source_path.read_text('utf-8'))
    rig = ArticulatedRig.model_validate_json((args.source/'rig.json').read_text('utf-8'))
    groups = {split: [JointState(np.asarray(r['joint_state']['root']),
                                np.asarray(r['joint_state']['local_orientations']))
                      for r in rows if r['split']==split] for split in ('train', 'heldout')}
    torch.set_num_threads(1)
    torch.manual_seed(61)
    rng = np.random.default_rng(61)
    started = time.monotonic()
    def progress(row):
        (output/'progress.json').write_text(json.dumps(row | {'elapsed_s': time.monotonic()-started}), 'utf-8')
    dynamics = DifferentiableRig(rig).float()
    if rig.tracker_nodes[2] != 0:
        raise ValueError('teacher observation must identify the root pelvis')
    roots = torch.tensor(np.stack([s.root for s in groups['train']]), dtype=torch.float32)
    joints = torch.tensor(np.stack([s.rotations for s in groups['train']]), dtype=torch.float32)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    prior = prior_report = None
    frame = args.policy_frame or FRAME
    if args.resume_source:
        prior_report = json.loads((args.resume_source/'result.json').read_text('utf-8'))
        prior = ArticulatedActor(args.resume_source/'candidate-actor.pt')
        frame = args.policy_frame or prior.manifest.get('policy_frame')
        if (prior_report['source_sha256'] != source_hash or prior.rig != rig
                or prior.manifest.get('policy_frame') != frame
                or frame not in (FRAME, JOINT_WORLD_FRAME)
                or prior.manifest.get('net_arch') != [args.hidden_size]*2
                or prior.manifest['model_based_updates'] != 0):
            raise ValueError('incompatible prior resumption')
    goal_steps = args.goal_steps or (prior_report.get('goal_steps', 16) if prior_report else 16)
    if args.resume_source and not args.rollin_probability:
        if goal_steps != prior_report.get('goal_steps', 16):
            raise ValueError('reused teacher data has a different goal dwell')
        if args.heading_goal_range != prior_report.get('heading_goal_range', 0.):
            raise ValueError('reused teacher data has a different heading goal range')
        data_path = Path(prior_report.get('teacher_data_path', args.resume_source/'teacher-data.pt'))
        data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if prior_report.get('teacher_data_sha256', data_hash) != data_hash:
            raise ValueError('teacher data changed')
        data = torch.load(data_path, map_location='cpu', weights_only=True)
        obs, labels, sampling = data['observations'], data['teacher_actions'], prior_report['sampling']
        if (obs.shape != (args.samples, 210+4*len(rig.names))
                or labels.shape != (args.samples, rig.action_size)
                or not torch.isfinite(obs).all() or not torch.isfinite(labels).all()):
            raise ValueError('invalid teacher dataset')
    else:
        obs, labels, sampling = collect(dynamics, roots, joints, args.samples, rng, progress,
                                       rollin_actor=prior, rollin_probability=args.rollin_probability,
                                       goal_steps=goal_steps, heading_goal_range=args.heading_goal_range)
        data_path = output/'teacher-data.pt'
        torch.save({'observations': obs, 'teacher_actions': labels}, data_path)
        data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    timestep_counts = {str(dt): int(torch.isclose(obs[:, 209], obs.new_tensor(dt/.05)).sum())
                       for dt in (.02, .05, .1)}
    decoder = RateBasis(action_size=rig.action_size, rows=tuple(map(tuple, np.eye(rig.action_size))),
                        variance_fraction=1.)
    env = ArticulatedGoalEnv(rig, groups['train'], decoder=decoder, reference_floor=0.,
                             floor_weight=50., floor_power=1, worst_tracker_weight=1.)
    architecture = [args.hidden_size, args.hidden_size]
    policy, frame_kwargs = policy_for_frame(frame, rig)
    model = SAC(policy, env, device='cpu', seed=61, buffer_size=10,
                policy_kwargs={'net_arch': architecture, **frame_kwargs})
    model.algorithm_label = 'offline whole-body kinematic teacher behavior cloning'
    model.model_based_updates, model.prior_updates = 0, 0
    model.reference_floor, model.floor_weight, model.floor_power, model.worst_tracker_weight = 0., 50., 1, 1.
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.learning_rate)
    if args.resume_source:
        state = torch.load(args.resume_source/'training-state.pt', map_location='cpu', weights_only=True)
        model.actor.load_state_dict(state['actor'])
        optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate
        model.prior_updates = int(state['prior_updates'])
        if model.prior_updates != prior.manifest['motion_bc_updates']:
            raise ValueError('cloning counters differ')
        with torch.no_grad():
            expected = prior.model(obs[:8])
            actual = model.actor(obs[:8], deterministic=True)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    initial_updates = model.prior_updates
    losses = []
    for step in range(args.updates):
        ids = torch.randint(len(obs), (256,))
        prediction = model.actor(obs[ids], deterministic=True)
        loss = (tracker_loss(dynamics, obs[ids], prediction, labels[ids]) if args.tracker_rate_loss
                else (prediction-labels[ids]).square().mean())
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        model.prior_updates += 1
        if (step+1) % 250 == 0:
            row = {'phase': 'cloning', 'updates': step+1, 'loss': float(loss.detach()),
                   'gradient_norm': float(norm), **sampling}
            losses.append(row)
            progress(row)
    export_articulated(model, output/'candidate-actor.pt', rig, decoder)
    torch.save({'actor': model.actor.state_dict(), 'optimizer': optimizer.state_dict(),
                'model_based_updates': 0, 'prior_updates': model.prior_updates}, output/'training-state.pt')
    candidate, _ = evaluate(model, rig, decoder, groups['heldout'], reference_floor=0., floor_weight=50.,
                            floor_power=1, worst_tracker_weight=1.)
    report = {'scope': 'offline_ideal_articulated_teacher_not_avatar', 'license': 'CC0',
              'source_sha256': source_hash,
              'source': str(args.source.resolve()), 'train_poses': len(groups['train']),
              'heldout_poses': len(groups['heldout']), 'seed': 61, 'policy_frame': frame,
              'net_arch': architecture, 'samples': args.samples, 'bc_updates': model.prior_updates,
              'updates_this_run': model.prior_updates-initial_updates,
              'cloning_objective': 'tracker_rates_plus_0.01_joint_rates' if args.tracker_rate_loss else 'joint_rates',
              'learning_rate': args.learning_rate, 'teacher_data_path': str(data_path.resolve()),
              'teacher_data_sha256': data_hash,
              'resume_source': str(args.resume_source.resolve()) if args.resume_source else None,
              'rollin_probability': args.rollin_probability, 'goal_steps': goal_steps,
              'heading_goal_range': args.heading_goal_range,
              'timestep_counts': timestep_counts,
              'model_based_updates': 0, 'sac_updates': 0, 'sampling': sampling,
              'losses': losses, 'candidate': candidate, 'elapsed_s': time.monotonic()-started,
              'promoted': False, 'avatar_verified': False}
    (output/'result.json').write_text(json.dumps(report, indent=2), 'utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('losses', 'candidate')}), flush=True)
    env.close()


if __name__ == '__main__':
    main()
