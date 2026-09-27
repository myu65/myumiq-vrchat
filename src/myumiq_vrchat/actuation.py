"""Independent pose/input lifetimes, composed only by the existing output owner.

Canonical tracking-space poses never integrate controller root displacement.
Locomotion values are normalized input demands, not measured metres per second.
"""

import math

from pydantic import model_validator

from .body import EXTRA_PARTS, BodyTarget, Controls, Frozen, HandTarget, Pose, SignedUnit


class PoseTarget(Frozen):
    head: Pose
    left: Pose
    right: Pose
    chest: Pose | None = None
    pelvis: Pose | None = None
    left_elbow: Pose | None = None
    right_elbow: Pose | None = None
    left_knee: Pose | None = None
    right_knee: Pose | None = None
    left_foot: Pose | None = None
    right_foot: Pose | None = None

    @classmethod
    def from_target(cls, target):
        return cls(
            head=target.head,
            left=target.left.pose,
            right=target.right.pose,
            **{name: getattr(target, name) for name in EXTRA_PARTS},
        )


class LocomotionCommand(Frozen):
    forward: SignedUnit = 0.0
    strafe: SignedUnit = 0.0
    turn: SignedUnit = 0.0


class HandInputCommand(Frozen):
    left: Controls = Controls()
    right: Controls = Controls()

    @model_validator(mode="after")
    def no_navigation_ownership(self):
        for control in (self.left, self.right):
            if (
                control.sticks[1] != (0.0, 0.0)
                or control.stick_touches[1]
                or control.stick_clicks[1]
            ):
                raise ValueError("thumbstick 1 belongs to locomotion")
        return self


def split_target(target):
    """Compatibility boundary for old producers; no hidden lifetime renewal."""

    def hands(control):
        return control.model_copy(
            update={
                "sticks": tuple((0.0, 0.0) if i == 1 else x for i, x in enumerate(control.sticks)),
                "stick_touches": tuple(
                    False if i == 1 else x for i, x in enumerate(control.stick_touches)
                ),
                "stick_clicks": tuple(
                    False if i == 1 else x for i, x in enumerate(control.stick_clicks)
                ),
            }
        )

    # Legacy thumbstick touch/click and right Y are retained by the legacy frame
    # path below. The typed navigation command owns only its three explicit axes.
    return (
        PoseTarget.from_target(target),
        LocomotionCommand(
            forward=target.left.controls.sticks[1][1],
            strafe=target.left.controls.sticks[1][0],
            turn=target.right.controls.sticks[1][0],
        ),
        HandInputCommand(left=hands(target.left.controls), right=hands(target.right.controls)),
    )


class ActuatorCompositor:
    """No I/O, neural models or autonomous clock. Called by OutputSupervisor."""

    def __init__(self, input_ttl_s=0.15, pose_ttl_s=0.5):
        if not all(math.isfinite(t) and t > 0 for t in (input_ttl_s, pose_ttl_s)):
            raise ValueError("positive finite actuator leases required")
        self.input_ttl_s, self.pose_ttl_s = input_ttl_s, pose_ttl_s
        self.pose = self.locomotion = self.hands = self.legacy_controls = None
        self.stamps = {}
        self.joint_servo = None

    def _stamp(self, channel, stamp, now, ttl):
        if (
            not math.isfinite(stamp)
            or not 0 <= now - stamp < ttl
            or stamp < self.stamps.get(channel, -math.inf)
        ):
            raise ValueError("actuator command is stale or out of order")
        self.stamps[channel] = stamp

    def publish_pose(self, pose, stamp, now):
        self._stamp("pose", stamp, now, self.pose_ttl_s)
        self.pose = pose
        self.joint_servo = None

    def publish_trajectory(self, target, command, stamp, now):
        from .joint_trajectory import JointServo

        if command.knots[-1].time > stamp + 0.51 or command.knots[0].time > stamp + 0.01:
            raise ValueError("trajectory outside the producer's finite horizon")
        servo = self.joint_servo
        self.publish_frame(target, stamp, now)
        if servo is None or servo.command.epoch != command.epoch:
            servo = JointServo(command, now)
        else:
            servo.update(command)
        self.joint_servo = servo

    def publish_locomotion(self, command, stamp, now):
        self._stamp("locomotion", stamp, now, self.input_ttl_s)
        self.locomotion = command
        self.legacy_controls = None

    def publish_hands(self, command, stamp, now):
        self._stamp("hands", stamp, now, self.input_ttl_s)
        self.hands = command
        self.legacy_controls = None

    def publish_frame(self, target, stamp, now):
        pose, locomotion, hands = split_target(target)
        self.publish_pose(pose, stamp, now)
        self.publish_locomotion(locomotion, stamp, now)
        self.publish_hands(hands, stamp, now)
        self.legacy_controls = (target.left.controls, target.right.controls)

    def compose(self, now):
        pose_at = self.stamps.get("pose")
        if self.pose is None or not 0 <= now - pose_at < self.pose_ttl_s:
            self.joint_servo = None
            return None
        pose = self.joint_servo.sample(now) if self.joint_servo is not None else self.pose

        def fresh(key):
            return key in self.stamps and 0 <= now - self.stamps[key] < self.input_ttl_s

        hands = self.hands if fresh("hands") else HandInputCommand()
        motion = self.locomotion if fresh("locomotion") else LocomotionCommand()
        if self.legacy_controls is not None and fresh("hands") and fresh("locomotion"):
            left, right = self.legacy_controls
        else:

            def combine(control, stick):
                active = stick != (0.0, 0.0)
                return control.model_copy(
                    update={
                        "sticks": tuple(
                            stick if i == 1 else v for i, v in enumerate(control.sticks)
                        ),
                        "stick_touches": tuple(
                            active if i == 1 else v for i, v in enumerate(control.stick_touches)
                        ),
                    }
                )

            left = combine(hands.left, (motion.strafe, motion.forward))
            right = combine(hands.right, (motion.turn, 0.0))
        return BodyTarget(
            head=pose.head,
            left=HandTarget(pose=pose.left, controls=left),
            right=HandTarget(pose=pose.right, controls=right),
            **{name: getattr(pose, name) for name in EXTRA_PARTS},
        )
