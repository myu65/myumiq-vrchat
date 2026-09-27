import json

import numpy as np
import pytest
from test_joint_limits import bounded_fixture

from myumiq_vrchat.actuation import ActuatorCompositor, PoseTarget
from myumiq_vrchat.body import Controls
from myumiq_vrchat.joint_trajectory import JointSample, JointServo, JointTrajectory
from myumiq_vrchat.motion_buffer import MotionKnot
from myumiq_vrchat.whole_body import vector


def trajectory():
    rig, states = bounded_fixture()
    start = states[0]
    pose = rig.forward(start)
    rates = np.zeros(rig.action_size)
    rates[0] = 0.2
    end = start
    for _ in range(3):
        end, _, _, _ = rig.advance(end, rates, 0.1)
    samples = tuple(
        JointSample.from_knot(MotionKnot(t, s, np.zeros(66))) for t, s in ((1.0, start), (1.3, end))
    )
    return pose, JointTrajectory(
        epoch=1,
        rig=rig,
        origin=samples[0],
        measured=PoseTarget.from_target(pose),
        knots=samples,
        filter_s=0.035,
        floor=0.0,
    )


def test_servo_advances_between_producer_packets_without_renewing_inputs_or_pose():
    pose, command = trajectory()
    pose = pose.model_copy(
        update={
            "left": pose.left.model_copy(
                update={
                    "controls": Controls(sticks=((0.0, 0.0), (0.0, 0.4), (0.0, 0.0), (0.0, 0.0)))
                }
            )
        }
    )
    compositor = ActuatorCompositor()
    compositor.publish_trajectory(pose, command, 1.0, 1.0)
    values = [compositor.compose(1 + i / 60) for i in range(29)]
    assert values[1].left.controls.sticks[1][1] == 0.4
    assert values[10].left.controls.sticks[1][1] == 0
    assert values[18].head.position[0] > values[6].head.position[0]
    assert compositor.stamps["pose"] == 1.0
    assert compositor.compose(1.51) is None
    assert compositor.joint_servo is None


def test_ordinary_frame_cancels_future_motion_and_stale_trajectory_cannot_resume():
    pose, command = trajectory()
    compositor = ActuatorCompositor()
    compositor.publish_trajectory(pose, command, 1.0, 1.0)
    held = compositor.compose(1.1)
    compositor.publish_frame(held, 1.11, 1.11)
    assert compositor.compose(1.4) == held
    with pytest.raises(ValueError, match="stale"):
        compositor.publish_trajectory(pose, command, 1.0, 1.2)
    assert compositor.joint_servo is None


def test_servo_joint_limits_residual_and_fixed_epoch_context():
    pose, command = trajectory()
    # Round-trip through the exact bounded JSON IPC representation.
    text = command.model_dump_json()
    assert (
        len(
            json.dumps(
                {"trajectory": json.loads(text), "target": pose.model_dump(mode="json")}
            ).encode()
        )
        < 32768
    )
    command = JointTrajectory.model_validate_json(text)
    servo = JointServo(command, 1.0)
    for i in range(30):
        servo.sample(1 + i / 60)
        command.rig.validate_limits(servo.stages[-1])
    before = servo.stages[-1].root.copy()
    servo.update(command)
    np.testing.assert_allclose(before, servo.stages[-1].root)
    with pytest.raises(ValueError, match="immutable"):
        servo.update(command.model_copy(update={"filter_s": 0.05}))
    assert np.min(vector(servo.buffer.pose(servo.stages[-1]))[:, 2]) >= 0


def test_servo_checks_the_emitted_floor_not_just_knots():
    _, command = trajectory()
    servo = JointServo(command.model_copy(update={"floor": 10.0}), 1.0)
    with pytest.raises(ValueError, match="floor"):
        servo.sample(1.0)


def test_full_precision_twelve_knot_horizon_fits_wire_and_preserves_motion():
    from myumiq_vrchat.backends.supervisor import trajectory_wire

    pose, command = trajectory()
    state = command.origin.state()
    rng = np.random.default_rng(76)
    samples = []
    for i in range(12):
        state, _, _, _ = command.rig.advance(
            state, rng.uniform(-0.5, 0.5, command.rig.action_size), 0.04
        )
        samples.append(JointSample.from_knot(MotionKnot(600000 + i * 0.04, state, np.zeros(66))))
    command = command.model_copy(update={"knots": tuple(samples)})
    packet = dict(
        kind="trajectory",
        token="a" * 64,
        sequence=123456789,
        timestamp=600000.123456789,
        target=pose.model_dump(mode="json"),
        trajectory=trajectory_wire(command),
    )
    data = json.dumps(packet, allow_nan=False, separators=(",", ":"))
    assert len(data.encode()) <= 32768
    restored = JointTrajectory.model_validate_json(json.dumps(json.loads(data)["trajectory"]))
    assert len(restored.knots) == 12
    for before, after in zip(command.knots, restored.knots):
        assert after.time == pytest.approx(before.time, abs=1e-9)
        np.testing.assert_allclose(
            vector(command.rig.forward(before.state())),
            vector(restored.rig.forward(after.state())),
            atol=1e-7,
            rtol=0,
        )


def test_horizon_rejects_excessive_duration_and_future_start():
    pose, command = trajectory()
    bad = command.model_dump(mode="json")
    bad["knots"][-1]["time"] = 10.0
    with pytest.raises(ValueError, match="horizon"):
        JointTrajectory.model_validate_json(json.dumps(bad))
    with pytest.raises(ValueError, match="horizon"):
        ActuatorCompositor().publish_trajectory(pose, command, 0.9, 0.9)
