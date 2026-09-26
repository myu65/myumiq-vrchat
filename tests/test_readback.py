import math
from types import SimpleNamespace

import pytest

from myumiq_vrchat.backends.readback import OpenVRReadback
from myumiq_vrchat.body import Pose, compose, inverse, q_from_matrix


def readback_fixture(devices, assigned=4294967295):
    sensor = OpenVRReadback(None, "hmd")
    sensor.vr = SimpleNamespace(
        k_unTrackedDeviceIndexInvalid=4294967295,
        k_unMaxTrackedDeviceCount=8,
        TrackedDeviceClass_Controller=2,
        TrackedDeviceClass_GenericTracker=3,
        Prop_SerialNumber_String=1,
        Prop_ControllerRoleHint_Int32=2,
    )
    sensor.system = SimpleNamespace(
        getTrackedDeviceIndexForControllerRole=lambda role: assigned,
        getTrackedDeviceClass=lambda i: 2 if i in devices else 0,
        getStringTrackedDeviceProperty=lambda i, prop: devices[i][0],
        getInt32TrackedDeviceProperty=lambda i, prop: devices[i][1],
    )
    return sensor


def test_unavailable_legacy_role_does_not_hide_owned_controller():
    sensor = readback_fixture({1: ("VMT_1", 1), 2: ("VMT_2", 2)})
    assert sensor._controller_index(1, "VMT_1") == 1
    assert sensor._controller_index(2, "VMT_2") == 2
    assert sensor._controller_index(2, "VMT_9") == 4294967295


@pytest.mark.parametrize(
    "devices,assigned,match",
    [
        ({1: ("someone-else", 1), 2: ("VMT_1", 1)}, 1, "outside"),
        ({1: ("VMT_1", 2)}, 4294967295, "role hint"),
        ({1: ("VMT_1", 1), 2: ("VMT_1", 1)}, 4294967295, "duplicate"),
    ],
)
def test_controller_identity_ambiguity_remains_a_hard_error(devices, assigned, match):
    with pytest.raises(RuntimeError, match=match):
        readback_fixture(devices, assigned)._controller_index(1, "VMT_1")


def test_serial_lookup_still_marks_invalid_tracking_unavailable():
    sensor = readback_fixture({1: ("VMT_1", 1), 2: ("VMT_2", 2)})
    frame = Pose(position=(0.0, 0.0, 0.0))
    sensor.config = SimpleNamespace(
        left_index=1, right_index=2, hmd_from_stage=frame, vmt_from_stage=frame, trackers=()
    )
    sensor.vr.TrackingUniverseRawAndUncalibrated = 2
    sensor.vr.TrackedControllerRole_LeftHand = 1
    sensor.vr.TrackedControllerRole_RightHand = 2
    sensor.system.getStringTrackedDeviceProperty = lambda i, p: "hmd" if i == 0 else f"VMT_{i}"
    matrix = SimpleNamespace(m=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.6], [0.0, 0.0, 1.0, 0.0]])
    poses = [
        SimpleNamespace(
            bDeviceIsConnected=True,
            bPoseIsValid=i != 2,
            eTrackingResult=200 if i != 2 else 201,
            mDeviceToAbsoluteTracking=matrix,
        )
        for i in range(3)
    ]
    sensor.system.getDeviceToAbsoluteTrackingPose = lambda *args: poses
    body = sensor.observe()
    assert body.head.valid and body.left.valid
    assert not body.right.valid and body.right.pose is None
    assert body.right.connected and body.right.tracking_result == 201


def test_calibration_translation_rotation_is_invertible():
    frame = Pose(position=(0.3, -1.0, 0.2), orientation=(0.5, 0.5, 0.5, 0.5))
    pose = Pose(position=(0.2, -0.4, 1.6), orientation=(1.0, 0.0, 0.0, 0.0))
    back = compose(inverse(frame), compose(frame, pose))
    assert back.position == pytest.approx(pose.position)
    assert back.orientation == pytest.approx(pose.orientation)


def test_full_body_readback_removes_mount_offset_and_marks_lost_tracker():
    from myumiq_vrchat.backends.osc import TrackerBinding

    sensor = readback_fixture({1: ("VMT_1", 1), 2: ("VMT_2", 2)})
    identity = Pose(position=(0.0, 0.0, 0.0))
    sensor.config = SimpleNamespace(
        left_index=1,
        right_index=2,
        hmd_from_stage=identity,
        vmt_from_stage=identity,
        trackers=(
            TrackerBinding(
                part="pelvis", index=3, body_from_tracker=Pose(position=(0.0, 0.0, 0.1))
            ),
            TrackerBinding(part="chest", index=4),
        ),
    )
    sensor.vr.TrackingUniverseRawAndUncalibrated = 2
    sensor.vr.TrackedControllerRole_LeftHand = 1
    sensor.vr.TrackedControllerRole_RightHand = 2
    sensor.vr.TrackedDeviceClass_GenericTracker = 3
    scans = []
    classes = {1: 2, 2: 2, 3: 3, 4: 3}
    sensor.system.getTrackedDeviceClass = lambda i: scans.append(i) or classes.get(i, 0)
    sensor.system.getStringTrackedDeviceProperty = lambda i, p: "hmd" if i == 0 else f"VMT_{i}"
    matrix = SimpleNamespace(m=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    poses = [
        SimpleNamespace(
            bDeviceIsConnected=True,
            bPoseIsValid=True,
            eTrackingResult=200,
            mDeviceToAbsoluteTracking=matrix,
        )
        for _ in range(5)
    ]
    sensor.system.getDeviceToAbsoluteTrackingPose = lambda *args: poses
    assert sensor.observe().pelvis.pose.position == pytest.approx((0, 0, 0.9))
    assert scans == list(range(sensor.vr.k_unMaxTrackedDeviceCount))
    scans.clear()
    poses[3].bPoseIsValid = False
    observed = sensor.observe()
    assert not observed.pelvis.valid and observed.pelvis.pose is None and observed.chest.valid
    assert scans == list(range(sensor.vr.k_unMaxTrackedDeviceCount))
    # Each observation refreshes identity; later hot-plug duplicates are rejected.
    classes[5] = 3
    sensor.system.getStringTrackedDeviceProperty = lambda i, p: (
        "hmd" if i == 0 else "VMT_4" if i == 5 else f"VMT_{i}"
    )
    with pytest.raises(RuntimeError, match="duplicate owned tracker identity"):
        sensor.observe()


def test_current_role_identity_is_checked_even_with_an_inventory():
    devices = {1: ("VMT_1", 1), 2: ("VMT_2", 2)}
    sensor = readback_fixture(devices, assigned=1)
    inventory = sensor._device_inventory()
    devices[1] = ("replacement-controller", 1)
    with pytest.raises(RuntimeError, match="outside"):
        sensor._controller_index(1, "VMT_1", inventory)


@pytest.mark.parametrize(
    "matrix,expected",
    [
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], (1.0, 0.0, 0.0, 0.0)),
        ([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], (0.0, 1.0, 0.0, 0.0)),
        ([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], (0.0, 0.0, 1.0, 0.0)),
        ([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], (0.0, 0.0, 0.0, 1.0)),
        (
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0),
        ),
    ],
)
def test_matrix_readback_handles_half_turns(matrix, expected):
    assert q_from_matrix(matrix) == pytest.approx(expected)
