"""Complete simultaneous end-state conditions into one feasible whole-body goal.

This bounded slow-path optimizer never writes devices or replaces the actor.
An endpoint solution does not establish a feasible trajectory or avatar contact.
"""

import hashlib
import time
from dataclasses import dataclass
from threading import Event

import numpy as np

from .body import BodyGoal, qmul
from .whole_body import PARTS, target_from_vector, vector


def metric_target(world, name, now):
    matches = [o for o in world.objects if o.name == name]
    if len(matches) != 1:
        raise ValueError("condition target identity is missing or ambiguous")
    item = matches[0]
    if (
        item.source not in ("fixture", "manual")
        or item.last_seen is None
        or not 0 <= now - item.last_seen <= 0.5
        or item.confidence < 0.5
    ):
        raise ValueError("condition target requires fresh calibrated metric evidence")
    return item


@dataclass
class ResolvedConditions:
    goal: BodyGoal
    desired: np.ndarray
    position_mask: np.ndarray
    orientation_mask: np.ndarray
    bindings: dict

    def validate_targets(self, world, now):
        for name, binding in self.bindings.items():
            item = metric_target(world, name, now)
            if (
                item.source != binding["source"]
                or np.linalg.norm(np.asarray(item.position) - binding["position"]) > 0.01
            ):
                raise ValueError("condition target changed; a fresh goal resolution is required")

    def measure(self, pose):
        from .tracker_policy import pose_error

        error = pose_error(pose, target_from_vector(self.desired))
        rows = []
        for condition in self.goal.conditions:
            index = PARTS.index(condition.part)
            position = float(np.linalg.norm(error[index, :3] * self.position_mask[index]))
            angle = float(np.linalg.norm(error[index, 3:])) if self.orientation_mask[index] else 0.0
            rows.append(
                dict(
                    part=condition.part,
                    position_error_m=position,
                    rotation_error_rad=angle,
                    accepted=position <= condition.position_tolerance
                    and angle <= condition.angular_tolerance,
                )
            )
        return {"success": all(row["accepted"] for row in rows), "conditions": rows}

    def metadata(self):
        return {
            "goal_sha256": hashlib.sha256(self.goal.model_dump_json().encode()).hexdigest(),
            "bindings": self.bindings,
            "condition_goal": self.goal.model_dump(mode="json"),
            "scope": "kinematic_end_state_not_trajectory_or_contact",
        }


def resolve_conditions(goal, current, world, now):
    if not goal.conditions or goal.tasks or goal.constraints:
        raise ValueError("this adapter requires a condition-only BodyGoal")
    desired = vector(current)
    position_mask, orientation_mask = np.zeros((11, 3), dtype=bool), np.zeros(11, dtype=bool)
    bindings = {}
    for condition in goal.conditions:
        index = PARTS.index(condition.part)
        origin = np.zeros(3)
        if condition.frame == "current":
            origin = desired[index, :3].copy()
        elif condition.frame == "target":
            item = metric_target(world, condition.target, now)
            origin = np.asarray(item.position)
            bindings[item.name] = dict(
                position=item.position, observed_at=item.last_seen, source=item.source
            )
        if condition.position is not None:
            for axis, value in enumerate(condition.position):
                if value is not None:
                    desired[index, axis] = origin[axis] + value
                    position_mask[index, axis] = True
        if condition.orientation is not None:
            desired[index, 3:] = (
                qmul(condition.orientation, tuple(desired[index, 3:]))
                if condition.frame == "current"
                else condition.orientation
            )
            orientation_mask[index] = True
    if (
        np.max(np.linalg.norm((desired[:, :3] - vector(current)[:, :3]) * position_mask, axis=1))
        > 1.5
    ):
        raise ValueError("condition endpoint exceeds the supported local workspace")
    return ResolvedConditions(goal, desired, position_mask, orientation_mask, bindings)


def complete_goal(
    rig,
    current,
    resolved,
    *,
    reference_floor,
    initial=None,
    cancelled=None,
    timeout_s=3.0,
    max_iter=120,
):
    """Solve all parts jointly; unspecified coordinates prefer the current body.

    No intermediate optimizer state can be used as an actuation pose. A rejected
    or cancelled completion returns no goal. Root translation here is tracking
    space posture, never VRChat controller/world locomotion.
    """
    import torch

    from .articulated_body import JointState
    from .articulated_fit import fit_body, tensor_forward
    from .joint_limits import project
    from .joint_limits_torch import JointEnvelope, rotation_vector

    if not 0 < timeout_s <= 8 or not 10 <= max_iter <= 300 or not np.isfinite(reference_floor):
        raise ValueError("invalid condition completion limits")
    began = time.perf_counter()

    def check_cancelled():
        if cancelled and cancelled():
            raise RuntimeError("body condition completion cancelled")
        if time.perf_counter() - began > timeout_s:
            raise RuntimeError("body condition completion timed out")
        return False

    check_cancelled()
    if vector(current)[:, 2].min() < reference_floor:
        raise ValueError("condition start is below the declared tracker floor")
    if initial is None:
        prior = JointState(
            np.asarray(current.pelvis.position), np.tile([1.0, 0, 0, 0], (len(rig.names), 1))
        )
        initial, _ = fit_body(rig, current, prior, cancelled=check_cancelled)
    rig.validate_limits(initial)
    envelope = JointEnvelope(rig.joint_limits)
    anchor = torch.tensor(vector(current), dtype=torch.float64)
    desired = torch.tensor(resolved.desired, dtype=torch.float64)
    mask = torch.tensor(resolved.position_mask, dtype=torch.float64)
    orient = torch.tensor(resolved.orientation_mask, dtype=torch.bool)
    tolerances = np.ones(11)
    angles = np.ones(11)
    for c in resolved.goal.conditions:
        tolerances[PARTS.index(c.part)], angles[PARTS.index(c.part)] = (
            c.position_tolerance,
            c.angular_tolerance,
        )
    tolerance = torch.tensor(tolerances, dtype=torch.float64)
    angle_limit = torch.tensor(angles, dtype=torch.float64)
    root = torch.tensor(initial.root, dtype=torch.float64, requires_grad=True)
    rotations = torch.tensor(initial.rotations, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [root, rotations],
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    class Converged(Exception):
        pass

    def closure():
        nonlocal evaluations
        check_cancelled()
        evaluations += 1
        optimizer.zero_grad()
        q = torch.nn.functional.normalize(rotations, dim=1)
        pose = tensor_forward(rig, root, q)
        errors = torch.linalg.vector_norm((pose[:, :3] - desired[:, :3]) * mask, dim=1) / tolerance
        with torch.no_grad():
            dot = (pose[:, 3:] * desired[:, 3:]).sum(1).abs().clamp(0, 1)
            angle_ok = not orient.any() or bool(
                (2 * torch.acos(dot[orient]) <= angle_limit[orient] * 0.75).all()
            )
            if (
                errors.max() <= 0.75
                and angle_ok
                and pose[:, 2].min() >= reference_floor
                and envelope.violation(q).max() <= 1e-6
            ):
                raise Converged
        loss = torch.relu(errors - 0.5).square().sum()
        if orient.any():
            dot = (pose[:, 3:] * desired[:, 3:]).sum(1)
            signs = torch.where(dot[:, None] < 0, -1.0, 1.0)
            chord = torch.linalg.vector_norm(pose[:, 3:] * signs - desired[:, 3:], dim=1)
            loss += (
                torch.relu(chord[orient] / (2 * torch.sin(angle_limit[orient] / 4)) - 0.5)
                .square()
                .sum()
            )
        # Nearby whole-body completion, not freezing every unmentioned joint.
        loss += 0.05 * (pose[:, :3] - anchor[:, :3]).square().sum()
        loss += 0.01 * (1 - (pose[:, 3:] * anchor[:, 3:]).sum(1).square()).sum()
        loss += 10000 * torch.relu(reference_floor + 0.002 - pose[:, 2]).square().sum()
        if envelope.enabled:
            v = rotation_vector(q)
            box = torch.maximum(envelope.lower * 0.995 - v, v - envelope.upper * 0.995)
            radial = torch.relu(torch.linalg.vector_norm(v, dim=-1) - envelope.radius[:, 0] * 0.995)
            loss += 10000 * (torch.relu(box).square().sum() + radial.square().sum())
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Converged:
        pass
    check_cancelled()
    q = torch.nn.functional.normalize(rotations.detach(), dim=1).numpy()
    result = JointState(root.detach().numpy().copy(), q)
    rig.validate_limits(result)
    result = JointState(result.root, project(result.rotations, rig.joint_limits))
    pose = rig.forward(result)
    report = resolved.measure(pose)
    if not report["success"] or vector(pose)[:, 2].min() < reference_floor:
        raise ValueError("body conditions have no accepted kinematic completion")
    if np.linalg.norm(result.root - initial.root) > 0.75:
        raise ValueError("condition completion exceeds the local root workspace")
    return pose, {
        **resolved.metadata(),
        **report,
        "evaluations": evaluations,
        "elapsed_s": time.perf_counter() - began,
        "avatar_verified": False,
    }


class ConditionPreparation:
    """One cancellable job on the existing motor preparation executor."""

    def __init__(self):
        self.pending = self.resolved = self.report = self.pose = None

    def reset(self):
        if self.pending:
            self.pending[1].set()
        self.resolved = self.report = self.pose = None

    def retiring(self):
        if self.pending and self.pending[1].is_set():
            if not self.pending[0].done():
                return True
            # A cancelled worker's result or exception has no new-goal authority.
            try:
                self.pending[0].result()
            except Exception:
                pass
            self.pending = None
        return False

    def poll(self, controller, current, goal, world, now):
        from .articulated_controller import close_enough

        if self.retiring():
            return None
        if self.pending:
            future, cancel, start = self.pending
            if not future.done():
                return None
            self.pending = None
            pose, report = future.result()
            if not close_enough(start, current, 0.008, 0.04):
                raise ValueError("body moved during condition preparation")
            self.pose, self.report = pose, report
        if self.resolved:
            self.resolved.validate_targets(world, now)
        if self.pose is not None:
            return self.pose
        if self.resolved is None:
            self.resolved = resolve_conditions(goal, current, world, now)
            cancel = Event()
            rig, initial = controller.rig, controller.state
            resolved, floor = self.resolved, controller.reference_floor
            future = controller.submit(
                lambda: complete_goal(
                    rig,
                    current,
                    resolved,
                    reference_floor=floor,
                    initial=initial,
                    cancelled=cancel.is_set,
                )
            )
            self.pending = future, cancel, current
        return None
