"""Predict short motion horizons without waiting for individual device packets."""

import math
from collections import deque
from threading import Event

import numpy as np

from .articulated_body import articulated_observation
from .motion_buffer import MotionBuffer, MotionKnot, blend_state
from .tracker_policy import pose_error
from .whole_body import PARTS, state_target, vector


class BufferedActor:
    """Wrap the existing fitted actor; the supervisor remains the only writer.

    Buffered predictions are deliberately excluded from confirmed-step training.
    Fresh device feedback verifies the issued trajectory, not a predicted result.
    """

    def __init__(self, controller, hz=20.0, horizon_s=0.4, *, output_filter_s=0.0):
        if not 10 <= hz <= 30 or not 0.3 <= horizon_s <= 0.5:
            raise ValueError("invalid actor horizon configuration")
        self.base, self.hz, self.horizon_s = controller, hz, horizon_s
        self.buffer = None
        self.job = None
        self.epoch = 0
        self.issued = deque(maxlen=64)
        self.last_now = None
        self.diverged_at = None
        self.started = False
        self.horizons = self.underruns = 0
        self.reference = None
        self.next_submit = 0.0
        if not math.isfinite(output_filter_s) or not 0 <= output_filter_s <= 0.1:
            raise ValueError("invalid joint output filter time constant")
        self.output_filter_s = output_filter_s
        self.filtered_states = None
        self.filtered_at = None
        self.origin = self.measured = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def ready(self):
        return self.base.ready

    def reset(self):
        self.epoch += 1
        if self.job:
            self.job[3].set()
        if self.buffer is not None and self.last_now is not None:
            # A nearby latent estimate accelerates the next measured fit. It is
            # only an initial guess, never accepted as a new device observation.
            self.base.state = (
                self.filtered_states[-1]
                if self.output_filter_s and self.filtered_states is not None
                else self.buffer.sample(self.last_now).state
            )
        self.buffer = None
        self.issued.clear()
        self.started = False
        self.diverged_at = None
        self.next_submit = 0.0
        self.filtered_states = self.filtered_at = None
        self.origin = self.measured = None
        self.base.reset()

    def new_goal(self):
        self.reset()

    def end_goal(self):
        self.reset()

    def observe(self, body, now):
        self.last_now = now
        signals = [body.signal_for(p) for p in PARTS]
        if not all(
            s.valid and s.connected and s.pose is not None and 0 <= now - s.timestamp < 0.5
            for s in signals
        ):
            self.base.error = "buffered actor lost fresh device feedback"
            return False
        current = state_target(body)
        if self.buffer is None:
            return self.base.observe(body, now)
        # Each device can report a different packet from the short issued history.
        recent = [pose for at, pose in self.issued if 0 <= now - at <= 0.3]
        if recent:
            errors = np.stack([pose_error(current, pose) for pose in recent])
            matched = (
                (
                    (np.linalg.norm(errors[:, :, :3], axis=2) <= 0.025)
                    & (np.linalg.norm(errors[:, :, 3:], axis=2) <= 0.10)
                )
                .any(axis=0)
                .all()
            )
        else:
            matched = not self.started
        if matched:
            self.diverged_at = None
        elif self.diverged_at is None:
            self.diverged_at = now
        elif now - self.diverged_at >= 0.2:
            self.base.error = "device feedback diverged from the buffered trajectory"
        return self.ready

    def _plan(self, buffer, first, goal, reference, end, cancel):
        knots = [first]
        state, previous = first.state, first.rates
        at = first.time
        while at < end - 1e-8 and not cancel.is_set():
            dt = min(1 / self.hz, end - at)
            current = buffer.pose(state)
            desired = reference(at + dt) if reference else goal
            obs = articulated_observation(current, desired, previous, dt, state.rotations)
            latent = self.base.actor.predict(obs)[0]
            joints = self.base.actor.decoder.decode(latent)
            following, _, rates, _ = self.base.rig.advance(state, joints, dt)
            target = buffer.pose(following)
            if (
                self.base.reference_floor is not None
                and vector(target)[:, 2].min() < self.base.reference_floor
            ):
                raise ValueError("buffered candidate crossed the declared tracking floor")
            knots.append(MotionKnot(at + dt, following, rates))
            state, previous, at = following, rates, at + dt
        return knots

    def step(self, body, goal, dt, *, remaining_s=None):
        now = self.last_now
        current = state_target(body)
        if self.job and self.job[0].done():
            future, epoch, anchor, cancel = self.job
            self.job = None
            try:
                knots = future.result()
                if epoch == self.epoch and not cancel.is_set():
                    if now <= anchor:
                        self.buffer.extend(knots)
                        self.horizons += 1
                    else:
                        self.underruns += 1
            except Exception as exc:
                if epoch == self.epoch:
                    self.base.error = str(exc)[:300]
        if not self.ready:
            return current, None, None
        if self.buffer is None:
            self.buffer = MotionBuffer(self.base.rig, self.base.state, current)
            self.buffer.knots = [MotionKnot(now, self.base.state, np.zeros(66))]
            self.filtered_states = [self.base.state] * 3
            self.filtered_at = now
            self.origin, self.measured = self.buffer.knots[0], current
        end = now + min(self.horizon_s, remaining_s if remaining_s is not None else self.horizon_s)
        if self.job is None and now >= self.next_submit and self.buffer.knots[-1].time - now < 0.25:
            anchor = now + 0.15
            if anchor < end:
                first = self.buffer.sample(anchor)
                first = MotionKnot(anchor, first.state, first.rates.copy())
                cancel = Event()
                buffer, reference = self.buffer, self.reference
                future = self.base.submit(
                    lambda: self._plan(buffer, first, goal, reference, end, cancel)
                )
                self.job = future, self.epoch, anchor, cancel
                self.next_submit = now + 0.05
        knot = self.buffer.sample(now)
        output_state = knot.state
        if self.output_filter_s:
            elapsed = min(0.05, max(0.0, now - self.filtered_at))
            alpha = -math.expm1(-elapsed / self.output_filter_s)
            # Three causal convex stages attenuate velocity/acceleration jumps
            # without inventing tracker targets outside the actor's joint path.
            for i, previous_state in enumerate(self.filtered_states):
                output_state = blend_state(self.rig, previous_state, output_state, alpha)
                self.filtered_states[i] = output_state
            self.filtered_at = now
        action = self.buffer.pose(output_state)
        if (
            self.base.reference_floor is not None
            and vector(action)[:, 2].min() < self.base.reference_floor
        ):
            self.base.error = "interpolated candidate crossed the declared tracking floor"
            return current, None, None
        if len(self.buffer.knots) > 1 and now >= self.buffer.knots[0].time:
            self.started = True
        if now - self.buffer.knots[-1].time > 0.5:
            self.base.error = "motion horizon producer timed out"
            return current, None, None
        self.issued.append((now, action))
        self.base.previous = knot.rates.copy()
        self.base.steps_executed += 1
        return action, None, None

    def trajectory(self):
        """Only the task owner may publish this after an actual actor step."""
        from .actuation import PoseTarget
        from .joint_trajectory import JointSample, JointTrajectory

        if self.buffer is None or not self.ready:
            return None
        now = self.last_now
        first = self.buffer.sample(now)
        knots = [MotionKnot(now, first.state, first.rates)]
        knots += [k for k in self.buffer.knots if now < k.time <= now + 0.5]
        return JointTrajectory(
            epoch=self.epoch,
            rig=self.rig,
            origin=JointSample.from_knot(self.origin),
            measured=PoseTarget.from_target(self.measured),
            knots=tuple(JointSample.from_knot(k) for k in knots),
            filter_s=self.output_filter_s,
            floor=self.base.reference_floor,
        )

    def replay_observation(self, *args, **kwargs):
        return None

    def status(self):
        return {
            **self.base.status(),
            "execution_mode": "buffered",
            "actor_hz": self.hz,
            "output_filter_s": self.output_filter_s,
            "horizons": self.horizons,
            "missed_horizons": self.underruns,
            "horizon_pending": self.job is not None,
            "buffer_remaining_s": max(0, self.buffer.knots[-1].time - self.last_now)
            if self.buffer is not None and self.last_now is not None
            else 0,
            "replay_scope": "predicted_horizon_not_confirmed_training",
        }

    def close(self):
        self.reset()
        self.base.close()
