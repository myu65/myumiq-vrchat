import pytest
from pythonosc.osc_packet import OscPacket

from myumiq_vrchat.backends.osc import DeviceOutput, LiveConfig, TrackerBinding
from myumiq_vrchat.backends.vmt import StopPolicy
from myumiq_vrchat.body import (
    EXTRA_PARTS,
    BodyConstraint,
    BodyGoal,
    BodyTarget,
    BodyTask,
    Pose,
    rest_target,
    simulated_body,
)


def full_target():
    # Transport fixture, not a anatomically calibrated posture.
    return rest_target().model_copy(
        update={part: Pose(position=(0.0, 0.0, 1.0)) for part in EXTRA_PARTS}
    )


def configuration():
    return LiveConfig(
        calibrated=True,
        console_validated=True,
        vmt_from_stage=Pose(position=(0.0, 0.0, 0.0)),
        hmd_from_stage=Pose(position=(0.0, 0.0, 0.0)),
        safe_target=full_target(),
        trackers=tuple(
            TrackerBinding(part=part, index=i + 3) for i, part in enumerate(EXTRA_PARTS)
        ),
        full_body_envelope=2.0,
    )


def test_old_record_has_unavailable_extra_body_not_fabricated_tracking():
    old = '{"head":{"position":[0,0,1.6]},"left":{"pose":{"position":[0,0,1]}},"right":{"pose":{"position":[0,0,1]}}}'
    target = BodyTarget.model_validate_json(old)
    body = simulated_body(target, 1.0)
    assert not target.is_full_body
    assert body.head.valid and not body.pelvis.valid and body.pelvis.pose is None
    assert body.signal_for("left_hand") == body.left


def test_walk_look_hold_and_free_hand_can_coexist_without_conflicting_ownership():
    tasks = (
        BodyTask(id="walk", kind="locomotion", effectors=("pelvis", "left_foot", "right_foot")),
        BodyTask(id="look", kind="gaze", effectors=("head",), target="person"),
        BodyTask(id="carry", kind="hold", effectors=("left_hand",), target="cup"),
        BodyTask(id="wave", kind="gesture", effectors=("right_hand",)),
    )
    constraint = BodyConstraint(
        part="left_hand", kind="fixed_pose", pose=Pose(position=(0.3, 0.2, 1.1))
    )
    goal = BodyGoal(tasks=tasks, constraints=(constraint,), duration_s=5)
    assert len(goal.tasks) == 4
    with pytest.raises(ValueError, match="conflict"):
        BodyGoal(
            tasks=tasks + (BodyTask(id="other", kind="gesture", effectors=("left_hand",)),),
            duration_s=5,
        )


def test_all_owned_trackers_sent_disabled_and_no_global_reset():
    sent = []
    output = DeviceOutput(configuration(), lambda data, port: sent.append((data, port)))
    output.send_target(full_target())
    poses = [
        m.message.params
        for data, port in sent
        if port == 39570
        for m in OscPacket(data).messages
        if m.message.address == "/VMT/Raw/Driver"
    ]
    assert [(p[0], p[1]) for p in poses] == [(1, 5), (2, 6)] + [(i, 7) for i in range(3, 11)]
    assert sum(port == 4242 for _, port in sent) == 1
    sent.clear()
    output.stop_pose(StopPolicy.DISABLE_OWNED)
    messages = [m.message for data, port in sent if port == 39570 for m in OscPacket(data).messages]
    assert {m.params[0] for m in messages if m.address == "/VMT/Raw/Driver"} == set(range(1, 11))
    assert all(m.params[1] == 0 for m in messages if m.address == "/VMT/Raw/Driver")
    assert not any(m.address == "/VMT/Reset" for m in messages)
    sent.clear()
    with pytest.raises(ValueError, match="capabilities"):
        output.send_target(rest_target())
    assert not sent
