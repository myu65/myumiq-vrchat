import numpy as np
import pytest

pytest.importorskip("mujoco")
from myumiq_vrchat.humanoid_env import WholeBodyTaskEnv


def test_goal_change_does_not_reset_articulated_state():
    env = WholeBodyTaskEnv()
    try:
        observation, _ = env.reset(seed=42)
        before = env.unwrapped.data.qpos.copy()
        env.set_task(0.2, -0.1, 1.35)
        assert np.array_equal(before, env.unwrapped.data.qpos)
        after, reward, terminated, truncated, info = env.step(np.zeros(17))
        assert after.shape == observation.shape == env.observation_space.shape
        assert np.isfinite(after).all() and np.isfinite(reward)
        assert reward == pytest.approx(sum(info["reward_components"].values()))
        assert info["motor_contract"] == env.contract
        assert not info["vrchat_transfer_verified"]
        with pytest.raises(ValueError):
            env.step(np.full(17, np.nan))
    finally:
        env.close()
