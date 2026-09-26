from myumiq_vrchat.body import Pose
from myumiq_vrchat.calibration import ResidualPolicy


def test_residual_policy_applies_only_translation():
    policy = ResidualPolicy(
        skill="WAVE", hand="right", correction=(0.1, -0.2, 0.3), training_samples=50
    )
    pose = Pose(position=(1.0, 2.0, 3.0))
    assert policy.apply(pose).position == (1.1, 1.8, 3.3)
    assert policy.apply(pose).orientation == pose.orientation
    assert ResidualPolicy.model_validate_json(policy.model_dump_json()) == policy


def test_training_rejects_non_real_or_insufficient_replay(tmp_path):
    from myumiq_vrchat.learning import train_residual_policy

    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    try:
        train_residual_policy(path, tmp_path / "policy.json")
    except ValueError as exc:
        assert "50 active real-device" in str(exc)
    else:
        raise AssertionError("empty replay was accepted")
    assert not (tmp_path / "policy.json").exists()
