"""Fixed-length FK diagnostic poses; predicted targets, never contact observations.

One shared 1.6m body generates all eleven points. These are calibration fixtures,
not imitation results or physics-stable motions. standing_up is a transition
to standing via the motor's bounded_step, not a separate static pose.
"""

import math

from .body import BodyTarget, HandTarget, Pose, compose


def pitch(angle):
    return (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


def posture_target(name: str) -> BodyTarget:
    if name not in ("standing", "crouching", "sitting_floor", "lying"):
        raise ValueError("unknown diagnostic posture")
    thigh = {
        "standing": 0.0,
        "lying": 0.0,
        "crouching": math.pi / 3,
        "sitting_floor": math.acos(-0.3 / 0.4),
    }[name]
    shin = {
        "standing": 0.0,
        "lying": 0.0,
        "crouching": -math.pi / 3,
        "sitting_floor": math.acos(0.37 / 0.4),
    }[name]
    pelvis_z = 0.08 + 0.4 * (math.cos(thigh) + math.cos(shin))
    pelvis = Pose(position=(0.0, 0.0, pelvis_z))
    chest = Pose(position=(0.0, 0.0, pelvis_z + 0.42))
    poses = {"pelvis": pelvis, "chest": chest, "head": Pose(position=(0.0, 0.0, pelvis_z + 0.72))}
    for side, sign in (("left", 1), ("right", -1)):
        knee = (0.4 * math.sin(thigh), sign * 0.09, pelvis_z - 0.4 * math.cos(thigh))
        foot = (knee[0] + 0.4 * math.sin(shin), knee[1], knee[2] - 0.4 * math.cos(shin))
        poses[side + "_knee"] = Pose(position=knee, orientation=pitch(-shin))
        poses[side + "_foot"] = Pose(position=foot)
        # Shoulder is a latent FK joint, elbows/hands are schema landmarks.
        shoulder = Pose(position=(0.0, sign * 0.22, pelvis_z + 0.48))
        arm_angle = -0.9 if name == "sitting_floor" else 0.0
        elbow = compose(
            shoulder,
            Pose(
                position=(0.28 * math.sin(-arm_angle), 0.0, -0.28 * math.cos(arm_angle)),
                orientation=pitch(arm_angle),
            ),
        )
        hand = compose(elbow, Pose(position=(0.0, 0.0, -0.25)))
        poses[side + "_elbow"], poses[side + "_hand"] = elbow, hand
    if name == "lying":
        # Supine, rotating about the pelvis so the body does not translate away.
        root = Pose(position=(pelvis_z, 0.0, 0.14), orientation=pitch(-math.pi / 2))
        poses = {key: compose(root, value) for key, value in poses.items()}
    return BodyTarget(
        head=poses.pop("head"),
        left=HandTarget(pose=poses.pop("left_hand")),
        right=HandTarget(pose=poses.pop("right_hand")),
        **poses,
    )
