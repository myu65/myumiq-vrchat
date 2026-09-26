"""Small replaceable procedural policy. No locomotion or inferred avatar IK."""

import math
from typing import Protocol

from .body import (
    EXTRA_PARTS,
    ActuationTarget,
    BodyState,
    Controls,
    Frozen,
    HandTarget,
    Number,
    Pose,
    WorldState,
    qmul,
    rest_target,
)
from .calibration import ResidualPolicy
from .cognition import Goal
from .gaze import VisualGazePolicy, image_target, yaw_pitch
from .reach_policy import LearnedReachPolicy
from .unity_reach import integrate_reach


class MotionCommand(Frozen):
    goal: Goal
    elapsed_s: Number


class MotorPolicy(Protocol):
    def step(
        self, body: BodyState, command: MotionCommand, world: WorldState, dt: float
    ) -> ActuationTarget: ...


def approach_pose(
    previous: Pose, desired: Pose, dt: float, speed: float = 0.6, angular_speed: float = 1.5
) -> Pose:
    distance = math.dist(previous.position, desired.position)
    f = min(1.0, speed * dt / max(distance, 1e-12))
    position = tuple(a + (b - a) * f for a, b in zip(previous.position, desired.position))
    a, b = previous.orientation, desired.orientation
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = tuple(-x for x in b), -dot
    angle = math.acos(max(-1.0, min(1.0, dot)))
    f = min(1.0, angular_speed * dt / max(2 * angle, 1e-12))
    if angle < 1e-6:
        q = a
    else:
        q = tuple(
            (math.sin((1 - f) * angle) * x + math.sin(f * angle) * y) / math.sin(angle)
            for x, y in zip(a, b)
        )
    return Pose(position=position, orientation=q)


class ProceduralMotor:
    def __init__(
        self,
        rest: ActuationTarget | None = None,
        residual: ResidualPolicy | None = None,
        gaze_policy: VisualGazePolicy | None = None,
        reach_policy: LearnedReachPolicy | None = None,
    ):
        self.rest = rest or rest_target()
        self.residual = residual
        self.previous = self.rest
        self.gaze_policy = gaze_policy
        self._gaze_sample = None
        self._gaze_angles = (0.0, 0.0)
        self.reach_policy = reach_policy
        self._reach_velocity = (0.0, 0.0, 0.0)
        self._reach_key = None
        self._reach_elapsed = 0.0
        self.last_motor_observation = None
        self.last_motor_action = None

    def reach_observation(self, body: BodyState, target) -> tuple[float, ...]:
        if not body.right.valid or body.right.pose is None:
            raise ValueError("learned reach requires hand observation")
        current = body.right.pose.position
        shoulder = (0.05, -0.2, self.rest.head.position[2] - 0.25)
        return (
            *(a - b for a, b in zip(target, current)),
            *(a - b for a, b in zip(current, shoulder)),
            *self._reach_velocity,
        )

    def step(
        self, body: BodyState, command: MotionCommand, world: WorldState, dt: float
    ) -> ActuationTarget:
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("motor dt must be finite and within (0, 0.1]")
        goal = command.goal
        self.last_motor_observation = None
        self.last_motor_action = None
        if goal.skill == "WAIT" or not 0 <= command.elapsed_s < goal.duration_s:
            # A completed request releases inputs but does not request a pose reset.
            # Read the observed pose, including supported full-body trackers.
            def pose(part):
                signal = body.signal_for(part)
                if not signal.valid or not signal.connected or signal.pose is None:
                    raise ValueError("pose hold requires valid device feedback")
                return signal.pose

            extra = {
                part: pose(part)
                for part in EXTRA_PARTS
                if getattr(self.previous, part) is not None or body.signal_for(part).valid
            }
            self.previous = ActuationTarget(
                head=pose("head"),
                left=HandTarget(pose=pose("left_hand")),
                right=HandTarget(pose=pose("right_hand")),
                **extra,
            )
            self._gaze_sample = self._reach_key = None
            self._reach_velocity = (0.0, 0.0, 0.0)
            return self.previous
        goal.validate_world(world)
        head, left, right = self.rest.head, self.rest.left, self.rest.right
        # Only an active RETURN_TO_REST requests the neutral trajectory.
        if 0 <= command.elapsed_s < goal.duration_s:
            if goal.skill == "LOOK_AT":
                target = world.locate(goal.target)
                delta = tuple(a - b for a, b in zip(target, head.position))
                yaw = max(-0.8, min(0.8, math.atan2(delta[1], delta[0])))
                pitch = max(-0.5, min(0.5, -math.atan2(delta[2], math.hypot(*delta[:2]))))
                visual = next(o for o in world.objects if o.name == goal.target).source == "vision"
                if self.gaze_policy is not None or visual:
                    if not body.head.valid or body.head.pose is None:
                        raise ValueError("learned gaze requires observed head pose")
                    observed = image_target(world, goal.target, body.head.timestamp)
                    sample = (observed.name, observed.last_seen)
                    if sample != self._gaze_sample:
                        if self.gaze_policy is not None:
                            self._gaze_angles = self.gaze_policy.desired_angles(
                                body.head.pose,
                                observed.image_position,
                            )
                        else:
                            current_yaw, current_pitch = yaw_pitch(body.head.pose)
                            dx, dy = observed.image_position
                            self._gaze_angles = (
                                max(-0.8, min(0.8, current_yaw - max(-0.12, min(0.12, 0.3 * dx)))),
                                max(
                                    -0.5, min(0.5, current_pitch + max(-0.12, min(0.12, 0.3 * dy)))
                                ),
                            )
                        self._gaze_sample = sample
                    # Sample-and-hold: never reintegrate one visual error at the
                    # motor frame rate while waiting for the next camera frame.
                    yaw, pitch = self._gaze_angles
                head = Pose(
                    position=head.position,
                    orientation=qmul(
                        (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                        (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0),
                    ),
                )
            elif goal.skill in ("WAVE", "REACH"):
                side = 1.0 if goal.hand == "left" else -1.0
                if goal.skill == "WAVE":
                    desired = Pose(
                        position=(
                            0.25,
                            side * (0.32 + 0.06 * math.sin(5 * command.elapsed_s)),
                            self.rest.head.position[2] - 0.1,
                        )
                    )
                    curls = (0.1, 0.05, 0.05, 0.05, 0.05)
                else:
                    pos = world.locate(goal.target)
                    shoulder = (0.05, side * 0.2, self.rest.head.position[2] - 0.25)
                    # Reach is a bounded nearby gesture, with no root translation.
                    if math.dist(pos, shoulder) > 0.65 or pos[2] < 0.6:
                        raise ValueError("REACH target is outside the supported local workspace")
                    desired, curls = Pose(position=pos), (0.2,) * 5
                    if self.reach_policy is not None and goal.hand == "right":
                        if not body.right.valid or body.right.pose is None:
                            raise ValueError("learned reach requires hand observation")
                        if abs(self.rest.head.position[2] - 1.6) > 1e-6:
                            raise ValueError(
                                "learned reach requires its calibrated 1.6m body frame"
                            )
                        if (
                            self._reach_key != goal.target
                            or command.elapsed_s < self._reach_elapsed
                        ):
                            self._reach_velocity = (0.0, 0.0, 0.0)
                        self._reach_key, self._reach_elapsed = goal.target, command.elapsed_s
                        current = body.right.pose.position
                        observation = self.reach_observation(body, pos)
                        velocity_action = self.reach_policy.predict(observation)
                        self.last_motor_observation = observation
                        self.last_motor_action = velocity_action
                        position, self._reach_velocity = integrate_reach(
                            current,
                            self._reach_velocity,
                            velocity_action,
                            dt,
                        )
                        desired = Pose(position=position)
                hand = HandTarget(pose=desired, controls=Controls(curls=curls))
                if (
                    self.residual
                    and self.residual.skill == goal.skill
                    and self.residual.hand == goal.hand
                ):
                    hand = hand.model_copy(update={"pose": self.residual.apply(hand.pose)})
                if goal.hand == "left":
                    left = hand
                else:
                    right = hand
        if goal.skill != "LOOK_AT" or command.elapsed_s >= goal.duration_s:
            self._gaze_sample = None
        if goal.skill != "REACH" or goal.hand != "right" or command.elapsed_s >= goal.duration_s:
            self._reach_key = None
            self._reach_velocity = (0.0, 0.0, 0.0)
        result = ActuationTarget(
            **{part: getattr(self.rest, part) for part in EXTRA_PARTS},
            head=approach_pose(self.previous.head, head, dt, speed=0.2),
            left=HandTarget(
                pose=approach_pose(self.previous.left.pose, left.pose, dt), controls=left.controls
            ),
            right=HandTarget(
                pose=approach_pose(self.previous.right.pose, right.pose, dt),
                controls=right.controls,
            ),
        )
        self.previous = result
        return result
