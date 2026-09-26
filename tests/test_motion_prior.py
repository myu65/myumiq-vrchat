import hashlib
import math

import numpy as np
import pytest

from myumiq_vrchat.body import qmul, rotate
from myumiq_vrchat.motion_prior import (
    FiniteImitation,
    MotionPlayback,
    MotionReference,
    load_motion,
)
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.whole_body import target_from_vector, vector


def model():
    frames = []
    for t in np.linspace(0, 1, 81):
        p = vector(posture_target("standing"))
        p[:, 0] += 0.4 * t
        p[3, 2] += 0.25 * math.sin(math.pi * t)
        frames.append((float(t), target_from_vector(p)))
    return FiniteImitation.fit(frames, 1.0, "finite-demo", count=32)


def test_navigation_speed_scales_reference_phase_without_integrating_world_translation():
    motion = model()
    full = MotionPlayback(motion, motion.sample(0), 1.0)
    half = MotionPlayback(motion, motion.sample(0), 1.0)
    held = MotionPlayback(motion, motion.sample(0), 1.0)
    for index in range(10):
        now = index * 0.02
        full.observe(full.target(), now, 0.02)
        half.observe(half.target(), now, 0.02, speed_scale=0.5)
        held.observe(held.target(), now, 0.02, speed_scale=0.0)
    assert half.phase == pytest.approx(full.phase / 2)
    assert held.phase == 0.0
    assert full.offset == half.offset == held.offset


def test_finite_prior_preserves_distinct_boundaries_and_loads(tmp_path):
    motion = model()
    loaded = load_motion(motion.model_dump_json())
    assert isinstance(loaded, FiniteImitation)
    np.testing.assert_allclose(vector(loaded.sample(2)), vector(motion.sample(1)))
    np.testing.assert_allclose(vector(loaded.sample(-1)), vector(motion.sample(0)))
    assert loaded.sample(1).pelvis.position[0] - loaded.sample(0).pelvis.position[0] > 0.39
    path = tmp_path / "motion.json"
    motion.save(path)
    reference = MotionReference(policy=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert reference.load(-0.2) == motion
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        reference.load(-0.2)


def test_anchor_is_shared_across_sequence_and_tracking_translation_is_retained():
    motion = model()
    initial = motion.sample(0)
    p = vector(initial)
    q = (math.cos(0.4), 0.0, 0.0, math.sin(0.4))
    for row in p:
        row[:3] = np.asarray(rotate(q, tuple(row[:3]))) + [1.0, -0.3, 0.0]
        row[3:] = qmul(q, tuple(row[3:]))
    current = target_from_vector(p)
    playback = MotionPlayback(motion, current, 1.0)
    np.testing.assert_allclose(vector(playback.target()), p, atol=1e-7)
    playback.phase = 1.0
    difference = np.asarray(playback.target().pelvis.position) - current.pelvis.position
    expected = rotate(
        q, tuple(np.asarray(motion.sample(1).pelvis.position) - initial.pelvis.position)
    )
    np.testing.assert_allclose(difference, expected, atol=1e-7)


def test_no_phase_progress_from_wrong_duplicate_or_stale_feedback():
    motion = model()
    playback = MotionPlayback(motion, motion.sample(0), 1.0)
    for i in range(100):
        playback.observe(motion.sample(1), i * 0.02, 0.02)
    assert playback.phase == 0 and not playback.completed
    playback.observe(playback.target(), 3.0, 0.02)
    phase = playback.phase
    playback.observe(playback.target(), 3.0, 0.02)
    playback.observe(playback.target(), 2.0, 0.02)
    assert playback.phase == phase
    playback.observe(playback.target(), 100.0, 100.0)
    assert playback.phase - phase <= 1 / 32  # no phase catch-up after a long gap


def test_session_origin_does_not_integrate_previous_endpoint_error():
    motion = model()
    current = motion.sample(0)
    origin = current.pelvis.position[:2]
    for _ in range(50):
        playback = MotionPlayback(motion, current, 1.0, origin_xy=origin)
        np.testing.assert_allclose(playback.target().pelvis.position[:2], origin, atol=1e-8)
        # An imperfect finish is feedback, not a new room-space origin.
        p = vector(playback.target())
        p[:, 0] += 0.08
        current = target_from_vector(p)
    relative = MotionPlayback(motion, current, 1.0)
    assert relative.target().pelvis.position[0] == pytest.approx(origin[0] + 0.08)


def test_bent_pelvis_keeps_a_usable_heading_without_resetting_observed_pose():
    from myumiq_vrchat.articulated_tasks import anchored_goal

    motion = model()
    current = motion.sample(0)
    p = vector(current)
    yaw = (math.cos(0.3), 0.0, 0.0, math.sin(0.3))
    pitch = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
    p[2, 3:] = qmul(yaw, pitch)
    bent = target_from_vector(p)
    playback = MotionPlayback(motion, bent, 1.0)
    goal = anchored_goal(current, bent)
    np.testing.assert_allclose(
        rotate(playback.target().pelvis.orientation, (1.0, 0.0, 0.0)),
        rotate(yaw, (1.0, 0.0, 0.0)),
        atol=1e-7,
    )
    np.testing.assert_allclose(vector(goal), vector(playback.target()), atol=1e-7)
    np.testing.assert_allclose(vector(bent), p)


def test_complete_finite_motion_requires_observed_phase_coverage_and_never_wraps():
    motion = model()
    playback = MotionPlayback(motion, motion.sample(0), 1.0)
    for i in range(70):
        playback.observe(playback.target(), i * 0.02, 0.02)
    assert playback.completed and playback.phase == 1.0
    assert playback.evidence()["phase_bins_observed"] == 16
    np.testing.assert_allclose(vector(playback.target()), vector(motion.sample(1)), atol=1e-7)
