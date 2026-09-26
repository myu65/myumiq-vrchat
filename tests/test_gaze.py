"""Synthetic dynamics tests; none of these establish a live VRChat learning result."""

import math

import pytest

from myumiq_vrchat.body import Pose, WorldObject, WorldState, qmul, rest_target, simulated_body
from myumiq_vrchat.cognition import Decision, Goal
from myumiq_vrchat.gaze import VisualGazePolicy, train_visual_gaze, yaw_pitch
from myumiq_vrchat.motor import MotionCommand, ProceduralMotor
from myumiq_vrchat.replay import Observation, ReplayBuffer, Transition

np = pytest.importorskip("numpy")


def pose(yaw, pitch):
    return Pose(
        position=(0.0, 0.0, 1.6),
        orientation=qmul(
            (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
            (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0),
        ),
    )


def observation(now, angles, pixel, source="openvr_raw"):
    body = simulated_body(rest_target(), now)
    body = body.model_copy(
        update={
            "head": body.head.model_copy(
                update={
                    "pose": pose(*angles),
                    "source": source,
                }
            )
        }
    )
    world = WorldState(
        objects=(
            WorldObject(
                name="lamp",
                position=(2.0, 0.0, 1.6),
                source="vision",
                kind="object",
                image_position=tuple(float(x) for x in pixel),
                last_seen=now,
            ),
        ),
        timestamp=now,
    )
    return Observation(timestamp=now, body=body, world=world)


def replay(tmp_path, *, source="openvr_raw", one_axis=False):
    random = np.random.default_rng(17)
    response = np.array([[1.6, 0.12], [0.08, -2.1]])
    buffer = ReplayBuffer(100)
    goal = Goal(skill="LOOK_AT", target="lamp", duration_s=2)
    for index in range(30):
        motion = random.uniform(-0.08, 0.08, 2)
        if one_axis:
            motion[1] = 0
            motion[0] = 0.04 + index * 0.001
        old = observation(index * 2.0, (0, 0), (0.2, -0.2), source)
        new = observation(
            index * 2.0 + 1, motion, np.array((0.2, -0.2)) + response @ motion, source
        )
        action = rest_target().model_copy(update={"head": new.body.head.pose})
        transition = Transition(
            observation=old,
            next_observation=new,
            decision=Decision(goal=goal, source="fixed"),
            command=MotionCommand(goal=goal, elapsed_s=1),
            action=action,
            reward=math.hypot(*old.world.objects[0].image_position)
            - math.hypot(*new.world.objects[0].image_position),
            outcome="visual_feedback",
        )
        buffer.add(transition.model_dump_json())
    path = tmp_path / "experience.jsonl"
    buffer.save_state(path)
    return path, response


def test_fitted_gaze_uses_both_real_observation_channels(tmp_path):
    path, response = replay(tmp_path)
    report = train_visual_gaze(path, tmp_path / "policy.json")
    policy = VisualGazePolicy.model_validate_json((tmp_path / "policy.json").read_text())
    assert report["live_improvement_verified"] is False
    assert report["holdout_image_rmse"] < 1e-10
    error = np.array((0.1, -0.1))
    action = policy.desired_angles(pose(0, 0), tuple(error))
    assert np.linalg.norm(error + response @ action) < np.linalg.norm(error) / 2


@pytest.mark.parametrize("source,one_axis", [("simulated", False), ("openvr_raw", True)])
def test_rejects_missing_provenance_and_unexcited_axis(tmp_path, source, one_axis):
    path, _ = replay(tmp_path, source=source, one_axis=one_axis)
    with pytest.raises(ValueError):
        train_visual_gaze(path, tmp_path / "policy.json")
    assert not (tmp_path / "policy.json").exists()


@pytest.mark.parametrize("learned", [False, True])
def test_motor_holds_one_visual_correction_and_rejects_stale_frame(learned):
    policy = (
        VisualGazePolicy(inverse_jacobian=((0.5, 0.0), (0.0, -0.5)), training_samples=20)
        if learned
        else None
    )
    motor = ProceduralMotor(gaze_policy=policy)
    goal = Goal(skill="LOOK_AT", target="lamp", duration_s=5)
    first = observation(0.0, (0, 0), (0.4, 0.0))
    action = motor.step(first.body, MotionCommand(goal=goal, elapsed_s=0), first.world, 0.1)
    angle = yaw_pitch(action.head)[0]
    assert angle == pytest.approx(-0.12)
    body = first.body.model_copy(
        update={
            "head": first.body.head.model_copy(
                update={
                    "pose": action.head,
                    "timestamp": 0.1,
                }
            )
        }
    )
    second = motor.step(body, MotionCommand(goal=goal, elapsed_s=0.1), first.world, 0.1)
    assert yaw_pitch(second.head)[0] == pytest.approx(angle)
    stale = body.model_copy(update={"head": body.head.model_copy(update={"timestamp": 1.0})})
    with pytest.raises(ValueError, match="fresh"):
        motor.step(stale, MotionCommand(goal=goal, elapsed_s=1), first.world, 0.1)
    uncertain = first.world.model_copy(
        update={"objects": (first.world.objects[0].model_copy(update={"confidence": 0.2}),)}
    )
    with pytest.raises(ValueError, match="confident"):
        motor.step(first.body, MotionCommand(goal=goal, elapsed_s=0), uncertain, 0.1)


def test_template_target_rejects_ambiguous_matches(tmp_path):
    cv2 = pytest.importorskip("cv2")

    from myumiq_vrchat.vision import TemplateTargetDetector

    patch = np.random.default_rng(71).integers(0, 255, (24, 24, 3), dtype=np.uint8)
    path = tmp_path / "patch.png"
    assert cv2.imwrite(str(path), patch)
    detector = TemplateTargetDetector(path, name="fixture")
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[40:64, 70:94] = patch
    found = detector.detect(image, 0)
    assert len(found) == 1
    assert found[0].image_position == pytest.approx((2 * 82 / 300 - 1, 2 * 52 / 200 - 1))
    image[120:144, 200:224] = patch
    assert detector.detect(image, 1) == []
