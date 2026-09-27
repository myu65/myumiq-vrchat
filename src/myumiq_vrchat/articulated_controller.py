"""Explicit actor trial adapter: fit while holding, then apply increments to feedback."""

from threading import Event

import numpy as np

from .articulated_actor import ArticulatedActor
from .articulated_body import OBSERVATION_CONTRACT, JointState, articulated_observation
from .articulated_fit import fit_body
from .tracker_action import integrate_tracker_action
from .tracker_policy import pose_error
from .whole_body import state_target, vector

# Device readback confirmation is much tighter than task/rig fitting tolerance.
# Millimetre-sized motion must not be acknowledged by the unchanged prior frame.
FEEDBACK_POSITION = 0.00001
FEEDBACK_ANGLE = 0.00001


def close_enough(a, b, position=0.008, angle=0.04):
    error = pose_error(a, b)
    return (
        np.linalg.norm(error[:, :3], axis=1).max() <= position
        and np.linalg.norm(error[:, 3:], axis=1).max() <= angle
    )


def feedback_matches(expected, current, before=None):
    if not close_enough(expected, current, FEEDBACK_POSITION, FEEDBACK_ANGLE):
        return False
    if before is None:
        return True
    # Absolute precision alone can acknowledge the old frame for a tiny command.
    # Require measurable progress toward this particular target as well.
    error = pose_error(expected, current)
    original = pose_error(expected, before)
    scale = np.array([0.6, 0.6, 0.6, 2.0, 2.0, 2.0])
    # HMD and individual VMT devices can arrive in different polling frames.
    # A large head movement cannot acknowledge a still-stale small foot movement.
    return bool(
        np.all(
            np.abs(error / scale).max(axis=1)
            <= np.maximum(1e-6, 0.25 * np.abs(original / scale).max(axis=1))
        )
    )


def state_data(state):
    return {"root": state.root.tolist(), "local_orientations": state.rotations.tolist()}


class ArticulatedController:
    observation_contract = OBSERVATION_CONTRACT

    def __init__(self, path, *, reference_floor=None, submit=None):
        self.actor = ArticulatedActor(path)
        self.manifest, self.rig = self.actor.manifest, self.actor.rig
        required_floor = self.manifest.get("reference_floor")
        if (
            required_floor is not None
            and reference_floor != required_floor
            or reference_floor is not None
            and not np.isfinite(reference_floor)
        ):
            raise ValueError("declare the actor's calibrated tracking-space reference floor")
        self.reference_floor = reference_floor
        if submit is None:
            from .purpose_runtime import background

            submit = background
        self.submit = submit
        self.state = self.prior = self.expected = self.pending = None
        self.previous = np.zeros(66, dtype=np.float32)
        self.last_latent = None
        self.last_metadata = {}
        self.generation = 0
        self.error = None
        self.fit_report = None
        self.feedback_pending = False
        self.feedback_before = None
        self.hold_target = None
        self.observed_at = self.feedback_deadline = 0.0
        self.steps_executed = 0
        self.last_issued_at = None
        self.rejected_step = None

    @property
    def ready(self):
        return self.state is not None and self.error is None and not self.feedback_pending

    def reset(self):
        self.generation += 1
        if self.pending:
            self.pending[3].set()
        if self.state is not None:
            self.prior = self.state
        self.state = self.expected = None
        self.feedback_pending = False
        self.feedback_before = None
        self.hold_target = None
        self.previous.fill(0.0)
        self.error = None
        self.last_metadata = {}
        self.last_issued_at = None
        self.rejected_step = None

    def new_goal(self):
        self.last_issued_at = None
        if self.error is not None:
            self.reset()

    def end_goal(self):
        self.last_issued_at = None
        self.previous.fill(0.0)
        if self.pending or self.feedback_pending:
            self.reset()

    def observe(self, body, now):
        current = state_target(body)
        self.observed_at = now
        if self.feedback_pending:
            if feedback_matches(self.expected, current, self.feedback_before):
                self.feedback_pending = False
            elif now < self.feedback_deadline and close_enough(
                self.feedback_before, current, 0.07, 0.25
            ):
                return False
            else:
                self.reset()
        if (
            self.pending
            and self.pending[1] == self.generation
            and not self.pending[3].is_set()
            and now - self.pending[4] > 8
            and self.error is None
        ):
            self.pending[3].set()
            self.generation += 1
            self.error = "articulated fitting timed out"
        if self.pending and self.pending[0].done():
            future, generation, snapshot, _, started = self.pending
            self.pending = None
            if generation == self.generation:
                try:
                    state, report = future.result()
                    if now - started > 8 or not close_enough(snapshot, current, 0.002, 0.015):
                        raise ValueError("body changed during articulated fitting")
                    self.state, self.fit_report, self.expected = state, report, current
                except Exception as exc:
                    self.error = str(exc)
        if self.ready:
            if (
                not close_enough(self.rig.forward(self.state), current)
                or self.expected is not None
                and not close_enough(self.expected, current, 0.005, 0.04)
            ):
                self.reset()
        if self.state is None and self.pending is None and self.error is None:
            self.hold_target = current
            root = np.asarray(current.pelvis.position)
            q = (
                self.prior.rotations.copy()
                if self.prior
                else np.tile([1.0, 0.0, 0.0, 0.0], (len(self.rig.names), 1))
            )
            prior, cancel = JointState(root, q), Event()
            future = self.submit(
                lambda: fit_body(self.rig, current, prior, cancelled=cancel.is_set)
            )
            self.pending = (future, self.generation, current, cancel, now)
        return self.ready

    def step(self, body, goal, dt, *, remaining_s=None):
        current = state_target(body)
        if self.feedback_pending:
            return self.expected, None, None
        if not self.ready:
            return self.hold_target or current, None, None
        if (
            not np.isfinite(dt)
            or dt <= 0
            or remaining_s is not None
            and (not np.isfinite(remaining_s) or remaining_s <= 0)
        ):
            raise ValueError("articulated integration requires a positive finite interval")
        elapsed = (
            (self.observed_at - self.last_issued_at) if self.last_issued_at is not None else None
        )
        # Rate commands use actual confirmed update cadence, not every polling tick.
        # Do not catch up a long pause or exceed the finite action's deadline.
        dt = min(0.1, max(dt, elapsed if elapsed is not None else dt))
        if remaining_s is not None:
            dt = min(dt, remaining_s)
        obs = articulated_observation(current, goal, self.previous, dt, self.state.rotations)
        before = state_data(self.state)
        previous_state = self.state
        try:
            latent = self.actor.predict(obs)[0]
            joints = self.actor.decoder.decode(latent)
            following, _, rates, scale = self.rig.advance(self.state, joints, dt)
            # Preserve measured residuals; zero action never publishes a fitted reference pose.
            action = integrate_tracker_action(current, rates.reshape(11, 6), dt)
            if close_enough(action, current, 0.000001, 0.00001):
                # Below output resolution, neither the device nor latent joints
                # advance. Otherwise discarded microsteps accumulate rig drift.
                following, action, rates, scale = self.state, current, np.zeros(66), 0.0
        except (ValueError, RuntimeError) as exc:
            self.error = str(exc)
            self.previous.fill(0.0)
            return current, None, None
        if self.reference_floor is not None and vector(action)[:, 2].min() < self.reference_floor:
            self.error = "candidate step below declared tracking-space reference floor"
            self.rejected_step = {
                "reason": "tracking_floor",
                "observed_at": self.observed_at,
                "integration_dt_s": dt,
                "observed_minimum_foot_height_m": float(vector(current)[9:, 2].min()),
                "proposed_minimum_foot_height_m": float(vector(action)[9:, 2].min()),
                "observed_minimum_tracker_height_m": float(vector(current)[:, 2].min()),
                "proposed_minimum_tracker_height_m": float(vector(action)[:, 2].min()),
                "reference_floor_m": self.reference_floor,
                "sent": False,
            }
            self.previous.fill(0.0)
            return current, None, None
        self.state, self.expected = following, action
        self.feedback_before = current
        self.feedback_pending = not close_enough(action, current, 0.000001, 0.00001)
        self.feedback_deadline = self.observed_at + 0.35
        self.steps_executed += 1
        self.last_issued_at = self.observed_at
        self.previous = rates.copy()
        self.last_latent = latent.copy()
        self.last_metadata = {
            "integration_dt": dt,
            "elapsed_since_command_s": elapsed,
            "observation_contract": OBSERVATION_CONTRACT,
            "joint_state": before,
            "next_joint_state": state_data(following),
            "expected_tracker_pose": action.model_dump(mode="json"),
            "issued_from_tracker_pose": current.model_dump(mode="json"),
            "feedback_deadline": self.feedback_deadline,
            "joint_state_source": "fitted_tracker_estimate",
            "joint_action": joints.tolist(),
            "joint_action_scale": scale,
            "realized_joint_action": self.rig.rates_between(previous_state, following, dt).tolist(),
            "joint_constraint_contract": self.manifest.get("joint_constraint_contract"),
            "decoder_id": self.actor.decoder.identity,
            "reference_floor": self.reference_floor,
            "reference_floor_weight": self.manifest.get("reference_floor_weight", 20.0),
            "reference_floor_power": self.manifest.get("reference_floor_power", 2),
            "floor_source": "operator_declared_tracking_plane"
            if self.reference_floor is not None
            else None,
        }
        return action, obs, rates

    def replay_observation(self, body, goal, rates, dt, metadata):
        import json

        from .body import BodyTarget

        current = state_target(body)
        expected = metadata.get("expected_tracker_pose")
        before = metadata.get("issued_from_tracker_pose")
        if expected is not None and not feedback_matches(
            BodyTarget.model_validate_json(json.dumps(expected)),
            current,
            BodyTarget.model_validate_json(json.dumps(before)) if before is not None else None,
        ):
            return None
        data = metadata["next_joint_state"]
        state = JointState(np.asarray(data["root"]), np.asarray(data["local_orientations"]))
        if not close_enough(self.rig.forward(state), current):
            return None
        return articulated_observation(current, goal, rates, dt, state.rotations)

    def status(self):
        return {
            "ready": self.ready,
            "fitting": self.pending is not None,
            "awaiting_feedback": self.feedback_pending,
            "steps_executed": self.steps_executed,
            "error": self.error,
            "fit_report": self.fit_report,
            "fit_pending_age_s": max(0.0, self.observed_at - self.pending[4])
            if self.pending
            else None,
            "fit_generation_active": self.pending[1] == self.generation if self.pending else False,
            "rejected_step": self.rejected_step,
            "avatar_verified": False,
        }

    def close(self):
        self.reset()
        if self.pending:
            try:
                self.pending[0].result(timeout=1.0)
            except Exception:
                pass  # cancelled result never reaches an output owner
