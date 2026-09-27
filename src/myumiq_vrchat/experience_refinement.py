"""PAMIQ training-only actor refinement from observed replay starts and goals."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from pamiq_core.torch import TorchTrainer

from .articulated_dynamics import (
    DifferentiableRig,
    multiply,
    observation,
    policy_rollout_start,
    pose_error,
    reward_components,
)


def preserve_observed_residual(before, after, observed):
    """Apply the predicted relative tracker increment to actual observed poses."""
    inverse = torch.cat((before[..., 3:4], -before[..., 4:]), dim=-1)
    delta = multiply(after[..., 3:], inverse)
    return torch.cat(
        (
            observed[..., :3] + after[..., :3] - before[..., :3],
            torch.nn.functional.normalize(multiply(delta, observed[..., 3:]), dim=-1),
        ),
        dim=-1,
    )


def nonfoot_floor_penalty(trackers, reference_floor, margin=0.02):
    """Clearance objective for lying/reaching, where the feet may not be lowest."""
    return torch.relu(reference_floor + margin - trackers[:, :9, 2]).amax(1)


class ExperienceRefinementTrainer(TorchTrainer):
    def __init__(
        self,
        rig,
        *,
        updates=100,
        horizon=4,
        batch_size=16,
        learning_rate=1e-4,
        reference_floor=0.0,
        anchor_weight=0.05,
        floor_weight=50.0,
        reference_rehearsal=False,
        whole_body_floor=False,
        extra_objective_factory=None,
        rollout_start_steps=0,
        progress=None,
    ):
        if not 1 <= updates <= 1000 or not 1 <= horizon <= 8 or not 1 <= batch_size <= 64:
            raise ValueError("invalid bounded replay refinement budget")
        if not 0 < learning_rate <= 0.001 or not 0 < anchor_weight <= 1:
            raise ValueError("invalid refinement learning rate or prior penalty")
        if not np.isfinite(floor_weight) or not 0 < floor_weight <= 1000:
            raise ValueError("invalid refinement floor penalty")
        if not isinstance(rollout_start_steps, int) or not 0 <= rollout_start_steps <= 100:
            raise ValueError("invalid practice rollout start budget")
        super().__init__(training_condition_data_user="experience", min_buffer_size=1)
        self.dynamics = DifferentiableRig(rig).float()
        self.updates, self.horizon, self.batch_size = updates, horizon, batch_size
        self.learning_rate, self.anchor_weight = learning_rate, anchor_weight
        if reference_rehearsal and batch_size < 2:
            raise ValueError("reference rehearsal requires a mixed batch")
        self.reference_rehearsal = reference_rehearsal
        self.whole_body_floor = whole_body_floor
        self.extra_objective_factory = extra_objective_factory
        self.rollout_start_steps = rollout_start_steps
        self.progress = progress
        self.objective = dict(
            reference_floor=reference_floor,
            floor_weight=floor_weight,
            floor_power=1,
            worst_tracker_weight=1.0,
        )
        self.total_updates = 0
        self.losses = []
        self.finished = False

    def on_training_models_attached(self):
        candidate = self.get_torch_training_model("candidate")
        if candidate.has_inference_model:
            raise ValueError("unvalidated candidate must not have a linked inference model")
        self.actor = candidate.model
        self.prior = copy.deepcopy(self.actor).eval().requires_grad_(False)

    def create_optimizers(self):
        return {"actor": torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)}

    def is_trainable(self):
        return not self.finished and super().is_trainable()

    def select_step(self, evaluate, accept, backtracking_steps=0):
        selected = select_refinement_step(
            self.actor, self.prior, self.optimizers["actor"], evaluate, accept, backtracking_steps
        )
        # TorchTrainer.save_state uses the optimizer snapshot cached by teardown.
        # Refresh that snapshot after a selected smaller step resets its moments.
        self.teardown()
        return selected

    def train(self):
        cases = list(self.get_data_user("experience").get_data())
        experience_count = len(cases)
        reference_count = 0
        if self.reference_rehearsal:
            references = list(self.get_data_user("reference").get_data())
            if not references:
                raise ValueError("reference rehearsal data is empty")
            cases.extend(references)
            reference_count = max(1, self.batch_size // 4)
        rng = np.random.default_rng(43)
        tensors = {
            key: torch.tensor(np.stack([getattr(c, key) for c in cases]), dtype=torch.float32)
            for key in ("root", "joints", "current", "goal", "previous")
        }
        extra_objective = (
            self.extra_objective_factory(cases) if self.extra_objective_factory else None
        )
        optimizer = self.optimizers["actor"]
        for _ in range(self.updates):
            indices = rng.integers(experience_count, size=self.batch_size - reference_count)
            if reference_count:
                indices = np.concatenate(
                    (indices, rng.integers(experience_count, len(cases), size=reference_count))
                )
            root, joints, current, goal, previous = (
                tensors[key][indices].clone()
                for key in ("root", "joints", "current", "goal", "previous")
            )
            # Each rollout uses a recorded integration interval, never wall-clock catch-up.
            dt = cases[int(indices[0])].dt
            if reference_count:
                # Visit late/near-floor reference states, not only clean initial poses.
                # These synthetic starts never acquire device-evidence identities.
                rows = slice(-reference_count, None)
                root[rows], joints[rows], previous[rows] = policy_rollout_start(
                    self.actor,
                    self.dynamics,
                    root[rows],
                    joints[rows],
                    goal[rows],
                    previous[rows],
                    dt,
                    int(rng.integers(0, 51)),
                    self.objective["reference_floor"],
                )
                current[rows] = self.dynamics(root[rows], joints[rows])
            if self.rollout_start_steps:
                # Preserve measured residuals during model-only burn-in. Never
                # invent readback IDs for these predicted intermediate starts.
                with torch.no_grad():
                    active = torch.ones(len(root), dtype=torch.bool)
                    for _ in range(int(rng.integers(self.rollout_start_steps + 1))):
                        action = self.actor(
                            observation(current, goal, previous, dt, joints), deterministic=True
                        )
                        before = self.dynamics(root, joints)
                        nr, nq, predicted, rates, _ = self.dynamics.step(root, joints, action, dt)
                        after = preserve_observed_residual(before, predicted, current)
                        active &= after[:, :, 2].amin(1) >= self.objective["reference_floor"]
                        root = torch.where(active[:, None], nr, root)
                        joints = torch.where(active[:, None, None], nq, joints)
                        current = torch.where(active[:, None, None], after, current)
                        previous = torch.where(active[:, None], rates, 0.0)
            objective = root.new_zeros(self.batch_size)
            optimizer.zero_grad()
            for step in range(self.horizon):
                obs = observation(current, goal, previous, dt, joints)
                action = self.actor(obs, deterministic=True)
                with torch.no_grad():
                    anchored = self.prior(obs, deterministic=True)
                predicted = self.dynamics(root, joints)
                root, joints, following, rates, _ = self.dynamics.step(root, joints, action, dt)
                after = preserve_observed_residual(predicted, following, current)
                reward = sum(
                    reward_components(
                        current, after, goal, rates, previous, **self.objective
                    ).values()
                )
                reward = reward - self.anchor_weight * (action - anchored).square().mean(1)
                if extra_objective is not None:
                    reward = reward + extra_objective(indices, current, after, rates, previous, dt)
                if self.whole_body_floor and self.objective["reference_floor"] is not None:
                    # Lying/reaching tasks can put hands or head below the feet.
                    # This adds non-foot clearance to the existing foot objective.
                    reward = reward - self.objective["floor_weight"] * nonfoot_floor_penalty(
                        after, self.objective["reference_floor"]
                    )
                objective = objective + 0.99**step * reward
                current, previous = after, rates
            loss = -objective.mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite replay refinement objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            self.total_updates += 1
            self.losses.append(float(loss.detach()))
            if self.progress and self.total_updates % 50 == 0:
                self.progress(self.total_updates, self.losses[-1])
        self.finished = True


@dataclass(frozen=True)
class ReferenceStart:
    """Synthetic rehearsal input, deliberately without any device-evidence identity."""

    root: np.ndarray
    joints: np.ndarray
    current: np.ndarray
    goal: np.ndarray
    previous: np.ndarray
    dt: float


def reference_starts(rig, states, reference_floor, count=64):
    from .articulated_body import JointState
    from .joint_limits import violation
    from .whole_body import vector

    valid = [
        s
        for s in states
        if float(violation(s.rotations, rig.joint_limits).max()) <= 1e-6
        and vector(rig.forward(s))[9:, 2].min() >= reference_floor
    ]
    if len(valid) < 2 or not 1 <= count <= 128:
        raise ValueError("bounded rehearsal requires at least two feasible training references")
    rng, cases = np.random.default_rng(43), []
    for index in range(count):
        start, goal = (valid[i] for i in rng.integers(len(valid), size=2))
        root = start.root.copy()
        if index % 2:
            # Translate the whole feasible skeleton, never individual feet.
            root[2] += (
                reference_floor + rng.uniform(0.02, 0.08) - vector(rig.forward(start))[9:, 2].min()
            )
        current = rig.forward(JointState(root, start.rotations))
        cases.append(
            ReferenceStart(
                root,
                start.rotations.copy(),
                vector(current),
                vector(rig.forward(goal)),
                np.zeros(66),
                0.05,
            )
        )
    return cases


def eligible_candidate(before, after, reference_before, reference_after, reference_floor):
    """Eligibility is a conjunction of task and constraint evidence, never a mean alone."""
    return bool(
        reference_before.get("control_contract") == "static_goal_settling_v1"
        and reference_after.get("control_contract") == "static_goal_settling_v1"
        and reference_after["static_goals_reached"] >= reference_before["static_goals_reached"]
        and after["floor_blocked"] <= before["floor_blocked"]
        and after["reached"] >= before["reached"]
        and after["mean_endpoint_ratio"] < before["mean_endpoint_ratio"] * 0.99
        and reference_after["mean_endpoint_error"] <= reference_before["mean_endpoint_error"] * 1.05
        and reference_after["minimum_foot_height_m"] >= reference_floor
        and reference_after["endpoints_worse_than_hold"]
        <= reference_before["endpoints_worse_than_hold"]
        and reference_after["maximum_forearm_shin_length_change_m"] <= 1e-6
    )


def select_refinement_step(actor, prior, optimizer, evaluate, accept, backtracking_steps=0):
    """Bounded validation selection; retain a rejected proposal, never relabel it."""
    if not 0 <= backtracking_steps <= 3:
        raise ValueError("backtracking requires zero to three halvings")
    proposed = copy.deepcopy(actor.state_dict())
    initial = prior.state_dict()
    attempts = []
    try:
        for step in range(backtracking_steps + 1):
            scale = 0.5**step
            if step:
                blended = {}
                for key, value in proposed.items():
                    if torch.is_floating_point(value):
                        blended[key] = initial[key] + scale * (value - initial[key])
                    elif torch.equal(value, initial[key]):
                        blended[key] = value
                    else:
                        raise ValueError("cannot interpolate changed nonfloating actor state")
                actor.load_state_dict(blended)
            metrics = evaluate()
            accepted = bool(accept(metrics))
            attempt = dict(scale=scale, eligible_for_controlled_trial=accepted, **metrics)
            attempts.append(attempt)
            if accepted:
                if step:
                    optimizer.state.clear()
                return attempt, attempts
    except BaseException:
        actor.load_state_dict(proposed)
        raise
    actor.load_state_dict(proposed)
    return attempts[0], attempts


@torch.no_grad()
def evaluate_observed_starts(actor, rig, cases, *, seconds=10.0, dt=0.05, reference_floor=0.0):
    """Replay-start goal recovery in the actuator model, not a replay of avatar motion."""
    if not cases or not 0 < seconds <= 20 or not 0.01 <= dt <= 0.1:
        raise ValueError("invalid bounded evaluation")
    dynamics = DifferentiableRig(rig).float()
    root, joints, current, goal, previous = (
        torch.tensor(np.stack([getattr(c, key) for c in cases]), dtype=torch.float32)
        for key in ("root", "joints", "current", "goal", "previous")
    )
    reached = torch.zeros(len(cases), dtype=torch.bool)
    floor_blocked = torch.zeros_like(reached)
    first = torch.full((len(cases),), float("nan"))
    stable = torch.zeros(len(cases), dtype=torch.int64)
    for step in range(int(seconds / dt)):
        errors = pose_error(current, goal)
        within = (errors[..., :3].norm(dim=-1).amax(1) <= 0.12) & (
            errors[..., 3:].norm(dim=-1).amax(1) <= 0.35
        )
        stable = torch.where(within, stable + 1, 0)
        arrived = stable >= max(3, int(np.ceil(0.15 / dt)) + 1)
        first = torch.where(arrived & ~reached, step * dt, first)
        reached |= arrived
        obs = observation(current, goal, previous, dt, joints)
        action = actor(obs, deterministic=True)
        action = torch.where((reached | floor_blocked)[:, None], 0.0, action)
        before = dynamics(root, joints)
        new_root, new_joints, following, rates, _ = dynamics.step(root, joints, action, dt)
        after = preserve_observed_residual(before, following, current)
        blocked = after[:, 9:, 2].amin(1) < reference_floor
        floor_blocked |= blocked
        root = torch.where(blocked[:, None], root, new_root)
        joints = torch.where(blocked[:, None, None], joints, new_joints)
        current = torch.where(blocked[:, None, None], current, after)
        previous = torch.where(blocked[:, None], 0.0, rates)
    errors = pose_error(current, goal)
    position = errors[..., :3].norm(dim=-1).amax(1)
    angle = errors[..., 3:].norm(dim=-1).amax(1)
    return dict(
        scope="observed_replay_start_model_rollout_not_avatar",
        samples=len(cases),
        reached=int(reached.sum()),
        floor_blocked=int(floor_blocked.sum()),
        mean_endpoint_ratio=float(torch.maximum(position / 0.12, angle / 0.35).mean()),
        max_position_error_m=float(position.max()),
        max_rotation_error_rad=float(angle.max()),
        mean_arrival_s=float(first[reached].mean()) if reached.any() else None,
    )
