"""Common body-frame observation transform and inverse root-rate transform."""

import torch
from torch.nn import functional as F

from .articulated_dynamics import multiply, rotate
from .whole_body import PARTS

FRAME = "root_xy_heading_v1"
JOINT_WORLD_FRAME = "root_xy_heading_joint_world_v2"


def canonical_observation(values):
    size = values.shape[0]
    poses = values[:, 66:143].reshape(size, 11, 7)
    pelvis = poses[:, PARTS.index("pelvis")]
    orientation = F.normalize(pelvis[:, 3:], dim=-1)
    forward = rotate(orientation, values.new_tensor([1.0, 0.0, 0.0]))
    left = rotate(orientation, values.new_tensor([0.0, 1.0, 0.0]))
    # atan2(0,0) has undefined derivatives even in an unselected torch.where branch.
    use_forward = forward[:, :2].square().sum(-1) >= 0.01
    x = torch.where(use_forward, forward[:, 0], left[:, 1])
    y = torch.where(use_forward, forward[:, 1], -left[:, 0])
    yaw = torch.atan2(y, x)
    zeros = torch.zeros_like(yaw)
    turn = torch.stack((torch.cos(yaw / 2), zeros, zeros, -torch.sin(yaw / 2)), dim=-1)
    pivot = torch.cat((pelvis[:, :2], zeros[:, None]), dim=-1)
    pose_turn = turn[:, None].expand(-1, 11, -1)
    transformed_q = multiply(pose_turn, F.normalize(poses[:, :, 3:], dim=-1))
    transformed_q = transformed_q * torch.where(transformed_q[:, :, :1] < 0, -1.0, 1.0)
    transformed_poses = torch.cat(
        (rotate(pose_turn, poses[:, :, :3] - pivot[:, None]), transformed_q), dim=-1
    )
    rate_turn = turn[:, None].expand(-1, 22, -1)
    errors = rotate(rate_turn, values[:, :66].reshape(size, 22, 3)).flatten(1)
    previous = rotate(rate_turn, values[:, 143:209].reshape(size, 22, 3)).flatten(1)
    joints = F.normalize(values[:, 210:].reshape(size, -1, 4), dim=-1)
    joints = torch.cat((multiply(turn, joints[:, 0])[:, None], joints[:, 1:]), dim=1)
    joints = joints * torch.where(joints[:, :, :1] < 0, -1.0, 1.0)
    result = torch.cat(
        (errors, transformed_poses.flatten(1), previous, values[:, 209:210], joints.flatten(1)),
        dim=1,
    )
    return result, turn


def tracking_action(local_action, turn):
    # A single coordinated speed scale, matching the actuator's initial bound.
    scale = local_action.reshape(local_action.shape[0], -1, 3).norm(dim=-1).amax(1).clamp(min=1)
    bounded = local_action / scale[:, None]
    inverse = turn * turn.new_tensor([1.0, -1.0, -1.0, -1.0])
    root = rotate(inverse[:, None].expand(-1, 2, -1), bounded[:, :6].reshape(-1, 2, 3))
    return torch.cat((root.flatten(1), bounded[:, 6:]), dim=1)


class JointWorldFrame(torch.nn.Module):
    """Learn from gauge-invariant joint positions and emit parent-local joint rates."""

    def __init__(self, rig):
        super().__init__()
        pelvis = PARTS.index("pelvis")
        if rig.tracker_nodes[pelvis] != 0 or rig.orientation_nodes[pelvis] != 0:
            raise ValueError("joint-world frame requires the observed pelvis to be the root")
        self.parents = rig.parents
        self.register_buffer("offsets", torch.tensor(rig.offsets, dtype=torch.float32))

    def encode(self, values):
        canonical, turn = canonical_observation(values)
        joints = canonical[:, 210:].reshape(values.shape[0], -1, 4)
        positions, orientations = [], []
        for i, parent in enumerate(self.parents):
            if parent < 0:
                positions.append(canonical[:, 80:83])
                orientations.append(joints[:, i])
            else:
                positions.append(positions[parent] + rotate(orientations[parent], self.offsets[i]))
                orientations.append(multiply(orientations[parent], joints[:, i]))
        xyz = torch.stack(positions, dim=1)
        # Keep the external feature width. The fourth coordinate is constant padding.
        features = torch.cat(
            (
                canonical[:, :210],
                torch.cat((xyz, torch.zeros_like(xyz[:, :, :1])), dim=-1).flatten(1),
            ),
            dim=1,
        )
        parent_q = torch.stack([orientations[p] for p in self.parents[1:]], dim=1)
        return features, turn, parent_q

    def decode(self, action, turn, parent_q):
        bounded = tracking_action(action, turn)
        inverse = parent_q * parent_q.new_tensor([1.0, -1.0, -1.0, -1.0])
        child_rates = rotate(inverse, bounded[:, 6:].reshape(action.shape[0], -1, 3))
        return torch.cat((bounded[:, :6], child_rates.flatten(1)), dim=1)
