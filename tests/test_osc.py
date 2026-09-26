import math
import struct

import pytest
from pythonosc.osc_packet import OscPacket

from myumiq_vrchat.backends.osc import DeviceOutput, LiveConfig, hmd_packet, to_openvr
from myumiq_vrchat.backends.vmt import StopPolicy
from myumiq_vrchat.body import ActuationTarget, Controls, HandTarget, Pose, qmul, rest_target


def configuration(vmt_port=39570, hmd_port=4242):
    return LiveConfig(
        vmt_port=vmt_port,
        hmd_port=hmd_port,
        calibrated=True,
        console_validated=True,
        vmt_from_stage=Pose(position=(0.0, 0.0, 0.0)),
        hmd_from_stage=Pose(position=(0.0, 0.0, 0.0)),
        safe_target=rest_target(),
    )


def decode(packets):
    return [(m.message.address, m.message.params) for p in packets for m in OscPacket(p).messages]


def test_full_frame_has_correct_controller_roles_types_and_curls():
    sent = []
    output = DeviceOutput(configuration(), lambda data, port: sent.append((port, data)))
    target = rest_target()
    c = Controls(
        buttons=(False, True) + (False,) * 16,
        triggers=(0.75,) + (0.0,) * 8,
        sticks=((0.0, 0.0), (0.4, -0.6), (0.0, 0.0), (0.0, 0.0)),
        curls=(1.0, 0.75, 0.5, 0.25, 0.0),
    )
    target = ActuationTarget(
        head=target.head, left=target.left, right=HandTarget(pose=target.right.pose, controls=c)
    )
    output.validate_target(target)
    output.send_target(target)
    messages = decode([data for port, data in sent if port == 39570])
    poses = [p for a, p in messages if a == "/VMT/Raw/Driver"]
    assert [(p[0], p[1]) for p in poses] == [(1, 5), (2, 6)]
    assert all(type(p[0]) is int and type(p[2]) is float for p in poses)
    assert poses[1][3:6] == pytest.approx([0.25, 1.15, -0.2])
    finger = [p for a, p in messages if a == "/VMT/Skeleton/Scalar" and p[0] == 2]
    assert [p[2] for p in finger] == pytest.approx([1.0, 0.0, 0.25, 0.5, 0.75, 1.0])
    assert any(a == "/VMT/Skeleton/Apply" and p[0] == 2 for a, p in messages)
    assert all(len(data) <= 1200 for _, data in sent)
    assert len(sent[-1][1]) == 48 and sent[-1][0] == 4242


def test_neutral_releases_entire_profile_and_stop_owns_only_two_devices():
    sent = []
    output = DeviceOutput(configuration(), lambda data, port: sent.append((port, data)))
    output.neutralize()
    messages = decode([d for p, d in sent if p == 39570])
    for i in (1, 2):
        inputs = [(a, p) for a, p in messages if p[0] == i and "/Input/" in a]
        assert len(inputs) == 18 * 2 + 9 * 3 + 4 * 3
        assert all(all(x == 0 for x in p[3:]) for _, p in inputs)
    sent.clear()
    output.stop_pose(StopPolicy.DISABLE_OWNED)
    messages = decode([d for p, d in sent if p == 39570])
    assert [(p[0], p[1]) for a, p in messages if a == "/VMT/Raw/Driver"] == [(1, 0), (2, 0)]
    assert not any(a == "/VMT/Reset" for a, _ in messages)
    with pytest.raises(ValueError):
        output.stop_pose(StopPolicy.RESET_ALL)


@pytest.mark.parametrize(
    "angles",
    [
        (0.0, 0.0, 0.0),
        (0.6, -0.4, 0.2),
        (1.57, 0.0, -1.0),
        (0.0, math.pi / 2, 0.3),
        (0.4, -math.pi / 2, -0.2),
    ],
)
def test_hmd_packet_reconstructs_the_requested_openvr_rotation(angles):
    # Independent driver reconstruction uses its exact Rz(Roll) Ry(-Yaw) Rx(Pitch).
    a, b, c = angles
    q = qmul(
        qmul(
            (math.cos(c / 2), 0.0, 0.0, math.sin(c / 2)),
            (math.cos(b / 2), 0.0, math.sin(b / 2), 0.0),
        ),
        (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0),
    )
    # Pose is canonical; helper conversion deliberately exercises axis remapping too.
    from myumiq_vrchat.backends.osc import from_openvr

    p = from_openvr(Pose(position=(0.1, 0.2, -0.3), orientation=q))
    x, z, y, yaw, pitch, roll = struct.unpack("<6d", hmd_packet(p, Pose(position=(0.0, 0.0, 0.0))))
    assert (x, y, z) == pytest.approx((10.0, 20.0, -30.0))
    a, b, c = map(math.radians, (pitch, -yaw, roll))
    reconstructed = qmul(
        qmul(
            (math.cos(c / 2), 0.0, 0.0, math.sin(c / 2)),
            (math.cos(b / 2), 0.0, math.sin(b / 2), 0.0),
        ),
        (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0),
    )
    assert abs(
        sum(x * y for x, y in zip(reconstructed, to_openvr(p).orientation))
    ) == pytest.approx(1.0)


def test_stop_head_is_attempted_even_when_vmt_sends_fail():
    attempted = []

    def fail_vmt(data, port):
        attempted.append(port)
        if port == 39570:
            raise OSError("injected VMT failure")

    output = DeviceOutput(configuration(), fail_vmt)
    with pytest.raises(OSError):
        output.stop_pose(StopPolicy.DISABLE_OWNED)
    assert attempted[-1] == 4242


def test_live_requires_calibration_and_full_safe_target():
    data = configuration().model_dump()
    data["calibrated"] = False
    with pytest.raises(ValueError):
        LiveConfig(**data)
    target = rest_target()
    invalid = ActuationTarget(
        head=Pose(position=(10.0, 0.0, 1.6)), left=target.left, right=target.right
    )
    with pytest.raises(ValueError):
        DeviceOutput(configuration(), lambda *_: None).validate_target(invalid)


def test_live_config_rejects_global_reset():
    data = configuration().model_dump()
    data["stop_policy"] = StopPolicy.RESET_ALL
    with pytest.raises(ValueError, match="never resets"):
        LiveConfig(**data)
