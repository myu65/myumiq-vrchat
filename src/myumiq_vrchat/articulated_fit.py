"""Bounded, non-real-time fitting of latent joints to measured tracker poses.

The fitted joints are estimates, not observations. This function never emits an
actuator command or changes the caller's current pose.
"""

import math
import time

import numpy as np

from .articulated_body import JointState
from .tracker_policy import pose_error
from .whole_body import vector


def tensor_forward(rig, root, rotations):
    import torch

    rotations = torch.nn.functional.normalize(rotations, dim=1)
    positions, orientations = [], []
    for i, parent in enumerate(rig.parents):
        local = rotations[i]
        if parent < 0:
            positions.append(root)
            orientations.append(local)
        else:
            q = orientations[parent]
            offset = root.new_tensor(rig.offsets[i])
            moved = offset + 2 * torch.linalg.cross(
                q[1:], torch.linalg.cross(q[1:], offset) + q[0] * offset
            )
            positions.append(positions[parent] + moved)
            orientations.append(
                torch.cat(
                    (
                        (q[:1] * local[:1] - (q[1:] * local[1:]).sum()).reshape(1),
                        q[0] * local[1:] + local[0] * q[1:] + torch.linalg.cross(q[1:], local[1:]),
                    )
                )
            )
    return torch.stack(
        [
            torch.cat((positions[p], orientations[q]))
            for p, q in zip(rig.tracker_nodes, rig.orientation_nodes)
        ]
    )


def fit_body(
    rig,
    target,
    prior,
    *,
    max_iter=160,
    position_tolerance=0.005,
    angular_tolerance=0.02,
    cancelled=None,
):
    import torch

    started = time.perf_counter()
    from .joint_limits import CONTRACT as LIMIT_CONTRACT
    from .joint_limits import project
    from .joint_limits_torch import JointEnvelope

    envelope = JointEnvelope(rig.joint_limits)
    if (
        not 10 <= max_iter <= 500
        or not 0 < position_tolerance <= 0.02
        or not 0 < angular_tolerance <= 0.1
    ):
        raise ValueError("invalid bounded fitting configuration")
    rig.forward(prior)  # Validate the explicit initial estimate; never reset a body to it.
    desired = torch.tensor(vector(target), dtype=torch.float64)
    root = torch.tensor(prior.root, dtype=torch.float64, requires_grad=True)
    rotations = torch.tensor(
        project(prior.rotations, rig.joint_limits), dtype=torch.float64, requires_grad=True
    )
    initial_rotations = rotations.detach().clone()
    optimizer = torch.optim.LBFGS(
        [root, rotations],
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    class Converged(Exception):
        pass

    def closure():
        nonlocal evaluations
        if cancelled is not None and cancelled():
            raise RuntimeError("articulated fitting cancelled")
        optimizer.zero_grad()
        normalized = torch.nn.functional.normalize(rotations, dim=1)
        predicted = tensor_forward(rig, root, normalized)
        evaluations += 1
        # These are the same per-tracker acceptance limits used below. Optimizing
        # far beyond them wastes the finite preparation budget and audio CPU time.
        with torch.no_grad():
            position_error = torch.linalg.vector_norm(predicted[:, :3] - desired[:, :3], dim=1)
            dot = (predicted[:, 3:] * desired[:, 3:]).sum(1).abs().clamp(0, 1)
            angular_error = 2 * torch.acos(dot)
            if (
                position_error.max() <= position_tolerance
                and angular_error.max() <= angular_tolerance
                and envelope.violation(normalized).max() <= 1e-6
            ):
                raise Converged
        signs = torch.where((predicted[:, 3:] * desired[:, 3:]).sum(1, keepdim=True) < 0, -1.0, 1.0)
        # Optimize the per-tracker acceptance contract, rather than average pose
        # error which can converge while one tracker remains outside tolerance.
        position_ratio = (
            torch.linalg.vector_norm(predicted[:, :3] - desired[:, :3], dim=1) / position_tolerance
        )
        chord_limit = 2 * math.sin(angular_tolerance / 4)
        angular_ratio = (
            torch.linalg.vector_norm(predicted[:, 3:] * signs - desired[:, 3:], dim=1) / chord_limit
        )
        loss = torch.relu(position_ratio - 0.9).square().mean()
        loss += torch.relu(angular_ratio - 0.9).square().mean()
        if envelope.enabled:
            # Fitting is an optimizer, not an actuator: a hard projected parameter
            # can stick outside a boundary with zero corrective gradient. Penalize
            # proposals against a slightly interior envelope, then require the
            # exact shared envelope at acceptance. No optimizer proposal is output.
            from .joint_limits_torch import rotation_vector

            v = rotation_vector(normalized)
            excess = torch.maximum(envelope.lower * 0.99 - v, v - envelope.upper * 0.99)
            radial = torch.relu(torch.linalg.vector_norm(v, dim=-1) - envelope.radius[:, 0] * 0.99)
            loss += 10 * (
                (torch.relu(excess) / 0.005).square().mean() + (radial / 0.005).square().mean()
            )
        # Prefer a nearby latent solution when tracker observations cannot identify a joint.
        loss += (
            1e-7
            * (torch.nn.functional.normalize(rotations, dim=1) - initial_rotations).square().mean()
        )
        loss.backward()
        return loss

    convergence = "optimizer_finished"
    try:
        optimizer.step(closure)
    except Converged:
        convergence = "tracker_tolerances"
    if cancelled is not None and cancelled():
        raise RuntimeError("articulated fitting cancelled")
    q = torch.nn.functional.normalize(rotations.detach(), dim=1).numpy()
    result = JointState(root.detach().numpy().copy(), q)
    rig.validate_limits(result)
    # Remove numerical tolerance-sized excess only after rejecting invalid fits.
    result = JointState(result.root, project(result.rotations, rig.joint_limits))
    errors = pose_error(rig.forward(result), target)
    position = float(np.linalg.norm(errors[:, :3], axis=1).max())
    angular = float(np.linalg.norm(errors[:, 3:], axis=1).max())
    if position > position_tolerance or angular > angular_tolerance:
        raise ValueError(
            f"observed pose does not fit calibrated rig: {position:.6f} m, {angular:.6f} rad"
        )
    return result, {
        "maximum_position_error_m": position,
        "maximum_rotation_error_rad": angular,
        "evaluations": evaluations,
        "elapsed_s": time.perf_counter() - started,
        "convergence": convergence,
        "joint_state_source": "fitted_tracker_estimate",
        "joint_constraint_contract": LIMIT_CONTRACT if rig.joint_limits else None,
        "avatar_verified": False,
    }
