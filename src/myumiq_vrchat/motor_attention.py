"""Bounded, disjoint gaze/gesture overlays; no model inference in motor frames."""

import math

from .body import BodyGoal, BodyTask, Pose, qmul
from .cognition import Goal
from .gaze import image_target, yaw_pitch
from .motor import MotionCommand, ProceduralMotor
from .whole_body import bounded_step, state_target


class AttentionMotor:
    def __init__(self):
        self.sample = None
        self.orientation = None
        self.gesture_key = None
        self.gesture_motor = None

    def gaze(self, body, target, world, policy=None):
        obj = next((o for o in world.objects if o.name == target), None)
        if obj is None or not body.head.valid:
            return None
        if obj.source == "vision":
            obj = image_target(world, target, body.head.timestamp)
            sample = (target, obj.last_seen)
            if sample != self.sample:
                if policy:
                    yaw, pitch = policy.desired_angles(body.head.pose, obj.image_position)
                else:
                    yaw, pitch = yaw_pitch(body.head.pose)
                    dx, dy = obj.image_position
                    yaw = max(-0.8, min(0.8, yaw - max(-0.12, min(0.12, 0.3 * dx))))
                    pitch = max(-0.5, min(0.5, pitch + max(-0.12, min(0.12, 0.3 * dy))))
                self.orientation = qmul(
                    (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                    (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0),
                )
                self.sample = sample
            return self.orientation
        delta = [a - b for a, b in zip(obj.position, body.head.pose.position)]
        yaw = max(-0.8, min(0.8, math.atan2(delta[1], delta[0])))
        pitch = max(-0.5, min(0.5, -math.atan2(delta[2], math.hypot(*delta[:2]))))
        return qmul(
            (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
            (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0),
        )

    def apply(self, owner, body, target, now, dt):
        tasks = []
        occupied = set()
        attention = owner.attention
        intent = owner.choice[2]
        if (
            (not attention or now >= attention[1])
            and intent.skill == "LOOK_AT"
            and now - owner.choice[1] < intent.duration_s
        ):
            attention = (intent.target, now + dt)
        if attention and now < attention[1]:
            try:
                rotation = self.gaze(body, attention[0], owner.world, owner.gaze_policy)
                if rotation:
                    target = target.model_copy(
                        update={"head": Pose(position=target.head.position, orientation=rotation)}
                    )
                    occupied.add("head")
                    tasks.append(
                        BodyTask(
                            id="attention",
                            kind="gaze",
                            effectors=("head",),
                            target=attention[0],
                            priority=90,
                        )
                    )
            except ValueError:
                self.sample = None
        if owner.gesture:
            gesture, started = owner.gesture
            if now - started < gesture.duration_s:
                if self.gesture_key != started:
                    self.gesture_key = started
                    self.gesture_motor = ProceduralMotor(target)
                self.gesture_motor.previous = state_target(body)
                action = self.gesture_motor.step(
                    body,
                    MotionCommand(
                        goal=Goal(
                            skill="WAVE", hand=gesture.hand, duration_s=min(10, gesture.duration_s)
                        ),
                        elapsed_s=now - started,
                    ),
                    owner.world,
                    dt,
                )
                target = target.model_copy(update={gesture.hand: getattr(action, gesture.hand)})
                part = gesture.hand + "_hand"
                occupied.add(part)
                tasks.append(
                    BodyTask(
                        id="conversation_gesture", kind="gesture", effectors=(part,), priority=80
                    )
                )
        if tasks:
            base = []
            for task in owner.goal.tasks:
                remaining = tuple(p for p in task.effectors if p not in occupied)
                if remaining:
                    base.append(task.model_copy(update={"effectors": remaining}))
            owner.goal = BodyGoal(tasks=tuple(base + tasks), duration_s=owner.goal.duration_s)
        owner.intent_metadata["concurrent_tasks"] = [t.model_dump(mode="json") for t in tasks]
        return bounded_step(state_target(body), target, dt)
