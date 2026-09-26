import numpy as np
import pytest

from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.tracker_action import integrate_tracker_action
from myumiq_vrchat.tracker_basis import RateBasis
from myumiq_vrchat.whole_body import vector


def basis():
    rng = np.random.default_rng(41)
    return RateBasis.fit(rng.uniform(-0.5, 0.5, (100, 66)), 12)


def test_zero_latent_holds_arbitrary_pose_and_normalization_preserves_coordination():
    model = basis()
    pose = posture_target("sitting_floor")
    zero = model.decode(np.zeros(12))
    np.testing.assert_array_equal(zero, np.zeros(66))
    np.testing.assert_allclose(
        vector(integrate_tracker_action(pose, zero.reshape(11, 6), 0.05)), vector(pose)
    )
    rates = model.decode(np.ones(12))
    matrix = np.asarray(model.rows)
    np.testing.assert_allclose(rates, (rates @ matrix.T) @ matrix, atol=1e-10)
    assert np.linalg.norm(rates.reshape(11, 6)[:, :3], axis=1).max() <= 1 + 1e-12
    assert np.linalg.norm(rates.reshape(11, 6)[:, 3:], axis=1).max() <= 1 + 1e-12


def test_basis_rejects_empty_motion_and_invalid_latent():
    with pytest.raises(ValueError, match="nonzero"):
        RateBasis.fit(np.zeros((100, 66)), 12)
    with pytest.raises(ValueError):
        basis().decode(np.full(12, np.nan))
    with pytest.raises(ValueError):
        basis().decode(np.zeros(66))


def test_latent_export_preserves_actual_latent_and_physical_actions(tmp_path):
    torch = pytest.importorskip("torch")
    sac = pytest.importorskip("stable_baselines3").SAC
    from myumiq_vrchat.tracker_env import TrackerGoalEnv
    from myumiq_vrchat.tracker_policy import TrackerActor, export_actor

    torch.set_num_threads(1)
    decoder = basis()
    env = TrackerGoalEnv(
        [posture_target("standing"), posture_target("crouching")], decoder=decoder, recording=True
    )
    model = sac(
        "MlpPolicy", env, device="cpu", buffer_size=10, policy_kwargs={"net_arch": [16, 16]}
    )
    path = tmp_path / "latent.pt"
    export_actor(model, path, decoder)
    actor = TrackerActor(path)
    obs, _ = env.reset(seed=12)
    expected = model.predict(obs, deterministic=True)[0]
    rates = actor.predict(obs)
    np.testing.assert_allclose(actor.last_latent, expected, atol=1e-6)
    np.testing.assert_allclose(rates, decoder.decode(expected), atol=1e-6)
    env.step(expected)
    record = env.replay.get_data()[0]
    assert record.intent_metadata["decoder_id"] == actor.manifest["decoder_id"] == decoder.identity
    np.testing.assert_allclose(record.intent_metadata["latent_action"], expected)
    np.testing.assert_allclose(record.learning.rates, rates, atol=1e-6)
