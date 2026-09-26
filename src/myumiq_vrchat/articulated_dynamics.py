"""Batched differentiable counterpart of the ideal joint-rate actuator, for training."""

import torch
from torch import nn
from torch.nn import functional as F

from .joint_limits_torch import JointEnvelope
from .tracker_policy import SEGMENTS


def multiply(a, b):
    return torch.cat(
        (
            a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True),
            a[..., :1] * b[..., 1:]
            + b[..., :1] * a[..., 1:]
            + torch.linalg.cross(a[..., 1:], b[..., 1:]),
        ),
        dim=-1,
    )


def rotate(q, v):
    v = torch.broadcast_to(v, q[..., 1:].shape)
    return v + 2 * torch.linalg.cross(
        q[..., 1:], torch.linalg.cross(q[..., 1:], v) + q[..., :1] * v
    )


def increment(q, angular, dt):
    angle = torch.linalg.vector_norm(angular, dim=-1, keepdim=True) * dt
    xyz = angular * (dt / 2) * torch.sinc(angle / (2 * torch.pi))
    return F.normalize(multiply(torch.cat((torch.cos(angle / 2), xyz), dim=-1), q), dim=-1)


def blended_goal(dynamics, root, joints, other_root, other_joints, fraction, clearance=0.05):
    """Training-data augmentation only; never a correction to observed body state."""
    root = root * (1 - fraction) + other_root * fraction
    aligned = other_joints * torch.where(
        (joints * other_joints).sum(-1, keepdim=True) < 0, -1.0, 1.0
    )
    q = F.normalize(joints * (1 - fraction[:, :, None]) + aligned * fraction[:, :, None], dim=-1)
    q = dynamics.envelope(q)
    pose = dynamics(root, q)
    lift = torch.relu(clearance - pose[:, 9:, 2].amin(1))
    adjusted_root = root + torch.stack(
        (torch.zeros_like(lift), torch.zeros_like(lift), lift), dim=1
    )
    return dynamics(adjusted_root, q)


def turned_goal(goal, angle):
    """Rotate a complete training goal about its pelvis without resetting posture."""
    zeros = torch.zeros_like(angle)
    turn = torch.stack((torch.cos(angle / 2), zeros, zeros, torch.sin(angle / 2)), dim=-1)
    rotations = turn[:, None].expand(-1, 11, -1)
    pivot = goal[:, 2:3, :3]
    return torch.cat(
        (pivot + rotate(rotations, goal[:, :, :3] - pivot), multiply(rotations, goal[:, :, 3:])),
        dim=-1,
    )


def pose_error(current, goal):
    qa, qb = current[..., 3:], goal[..., 3:]
    w = (qa * qb).sum(-1, keepdim=True)
    v = (
        -qb[..., :1] * qa[..., 1:]
        + qa[..., :1] * qb[..., 1:]
        - torch.linalg.cross(qb[..., 1:], qa[..., 1:])
    )
    sign = torch.where(w < 0, -1.0, 1.0)
    w, v = w * sign, v * sign
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    exact = 2 * torch.atan2(norm, w.clamp(0, 1)) / norm.clamp(min=1e-7)
    factor = torch.where(norm > 1e-5, exact, 2 / w.clamp(min=0.5))
    return torch.cat((goal[..., :3] - current[..., :3], v * factor), dim=-1)


def observation(current, goal, previous, dt, joints):
    error = pose_error(current, goal) / current.new_tensor([0.6, 0.6, 0.6, 2, 2, 2])
    poses = torch.cat(
        (current[..., :3], current[..., 3:] * torch.where(current[..., 3:4] < 0, -1.0, 1.0)), dim=-1
    )
    q = joints * torch.where(joints[..., :1] < 0, -1.0, 1.0)
    return torch.cat(
        (
            error.flatten(1),
            poses.flatten(1),
            previous,
            current.new_full((len(current), 1), dt / 0.05),
            q.flatten(1),
        ),
        dim=1,
    )


def distance(current, goal):
    error = pose_error(current, goal)
    return torch.linalg.vector_norm(error[..., :3], dim=-1).mean(
        -1
    ) + 0.15 * torch.linalg.vector_norm(error[..., 3:], dim=-1).mean(-1)


def worst_tracker_distance(current, goal):
    error = pose_error(current, goal)
    position = error[..., :3].norm(dim=-1).amax(-1) / 0.12
    rotation = error[..., 3:].norm(dim=-1).amax(-1) / 0.35
    return 0.12 * torch.maximum(position, rotation)


def joint_teacher(root, joints, goal_root, goal_joints, gain=2.0):
    """Offline complete-body teacher; never installed as a runtime fallback."""
    padding = torch.zeros_like(joints[..., :3])
    rotations = pose_error(
        torch.cat((padding, joints), dim=-1), torch.cat((padding, goal_joints), dim=-1)
    )[..., 3:]
    action = gain * torch.cat(((goal_root - root) / 0.6, (rotations / 2).flatten(1)), dim=1)
    maximum = action.reshape(root.shape[0], -1, 3).norm(dim=-1).amax(1).clamp(min=1)
    return action / maximum[:, None]


def reward_components(
    before,
    after,
    goal,
    rates,
    previous,
    reference_floor=None,
    floor_weight=20.0,
    floor_power=2,
    worst_tracker_weight=0.0,
):
    old, new = distance(before, goal), distance(after, goal)
    errors = []
    for a, b in SEGMENTS:
        current_length = torch.linalg.vector_norm(after[:, a, :3] - after[:, b, :3], dim=-1)
        goal_length = torch.linalg.vector_norm(goal[:, a, :3] - goal[:, b, :3], dim=-1)
        errors.append((current_length - goal_length).abs())
    result = {
        "progress": 10 * (old - new),
        "pose_error": -0.2 * new,
        "segment_distortion": -0.1 * torch.stack(errors, dim=1).mean(1),
        "effort": -0.002 * rates.square().mean(1),
        "action_change": -0.01 * (rates - previous).square().mean(1),
    }
    if worst_tracker_weight:
        old_worst, new_worst = (
            worst_tracker_distance(before, goal),
            worst_tracker_distance(after, goal),
        )
        result["worst_tracker_progress"] = 10 * worst_tracker_weight * (old_worst - new_worst)
        result["worst_tracker_error"] = -0.2 * worst_tracker_weight * new_worst
    if reference_floor is not None:
        result["reference_floor_intrusion"] = -floor_weight * torch.relu(
            reference_floor + 0.02 - after[:, 9:, 2]
        ).pow(floor_power).mean(1)
    return result


@torch.no_grad()
def policy_rollout_start(
    actor,
    dynamics,
    root,
    joints,
    goal,
    previous,
    dt,
    steps,
    reference_floor,
    *,
    reset_previous=None,
):
    """Sample valid states visited by the policy, without a differentiable burn-in."""
    if not isinstance(steps, int) or not 0 <= steps <= 100:
        raise ValueError("invalid bounded rollout-start length")
    if reset_previous is not None and (
        reset_previous.dtype != torch.bool or reset_previous.shape != (len(root),)
    ):
        raise ValueError("goal replacement mask must contain one boolean per body")
    active = torch.ones(len(root), dtype=torch.bool, device=root.device)
    for _ in range(steps):
        current = dynamics(root, joints)
        action = actor(observation(current, goal, previous, dt, joints), deterministic=True)
        next_root, next_joints, target, rates, _ = dynamics.step(root, joints, action, dt)
        if reference_floor is not None:
            active = active & (target[:, 9:, 2].amin(1) >= reference_floor)
        root = torch.where(active[:, None], next_root, root)
        joints = torch.where(active[:, None, None], next_joints, joints)
        previous = torch.where(active[:, None], rates, previous)
        if not bool(active.any()):
            break
    if reset_previous is not None:
        previous = torch.where(reset_previous[:, None], 0.0, previous)
    return root.detach(), joints.detach(), previous.detach()


class DifferentiableRig(nn.Module):
    def __init__(self, rig):
        super().__init__()
        self.parents = rig.parents
        self.tracker_nodes = rig.tracker_nodes
        self.orientation_nodes = rig.orientation_nodes
        self.envelope = JointEnvelope(rig.joint_limits)
        self.register_buffer("offsets", torch.tensor(rig.offsets, dtype=torch.float64))

    def forward(self, root, rotations):
        q = F.normalize(rotations, dim=-1)
        positions, orientations = [], []
        for i, parent in enumerate(self.parents):
            if parent < 0:
                positions.append(root)
                orientations.append(q[:, i])
            else:
                positions.append(positions[parent] + rotate(orientations[parent], self.offsets[i]))
                orientations.append(multiply(orientations[parent], q[:, i]))
        return torch.stack(
            [
                torch.cat((positions[p], orientations[q]), dim=-1)
                for p, q in zip(self.tracker_nodes, self.orientation_nodes)
            ],
            dim=1,
        )

    def step(self, root, joints, action, dt):
        if not 0 < dt <= 0.1:
            raise ValueError("invalid articulated training timestep")
        if bool((self.envelope.violation(joints).detach() > 1e-6).any()):
            raise ValueError("joint state is outside the configured joint envelope")
        before = self(root, joints)
        scale = (
            torch.linalg.vector_norm(action.reshape(len(action), -1, 3), dim=-1)
            .amax(1)
            .clamp(min=1)
            .reciprocal()
        )
        for _ in range(12):
            new_root = root + action[:, :3] * (0.6 * dt) * scale[:, None]
            new_joints = increment(
                joints, action[:, 3:].reshape(len(action), -1, 3) * 2 * scale[:, None, None], dt
            )
            new_joints = self.envelope(new_joints)
            after = self(new_root, new_joints)
            rates = (
                pose_error(before, after) / after.new_tensor([0.6, 0.6, 0.6, 2, 2, 2]) / dt
            ).flatten(1)
            maximum = torch.linalg.vector_norm(rates.reshape(len(action), -1, 3), dim=-1).amax(1)
            padding = torch.zeros_like(joints[..., :3])
            angular = pose_error(
                torch.cat((padding, joints), dim=-1), torch.cat((padding, new_joints), dim=-1)
            )[..., 3:] / (2 * dt)
            maximum = torch.maximum(maximum, torch.linalg.vector_norm(angular, dim=-1).amax(1))
            if bool((maximum.detach() <= 1).all()):
                return new_root, new_joints, after, rates, scale
            scale = scale * torch.where(maximum > 1, 0.999 / maximum.clamp(min=1), 1.0)
        raise ValueError("differentiable actuator could not meet speed limits")

    def equivalent_joints(self, joints, angles):
        """Vary unobserved single-child twists without changing any tracker output."""
        q = list(joints.unbind(dim=1))
        for i in range(len(self.parents)):
            children = [j for j, p in enumerate(self.parents) if p == i]
            if i in self.orientation_nodes or len(children) != 1:
                continue
            child = children[0]
            axis = F.normalize(self.offsets[child], dim=0)
            angle = angles[:, i : i + 1]
            twist = torch.cat((torch.cos(angle / 2), torch.sin(angle / 2) * axis), dim=-1)
            q[i] = multiply(q[i], twist)
            inverse = torch.cat((twist[:, :1], -twist[:, 1:]), dim=-1)
            q[child] = multiply(inverse, q[child])
        proposed = torch.stack(q, dim=1)
        valid = self.envelope.violation(proposed).amax(-1) <= 1e-6
        # Reject the entire gauge change; projecting it would change tracker outputs.
        return torch.where(valid[:, None, None], proposed, joints)
