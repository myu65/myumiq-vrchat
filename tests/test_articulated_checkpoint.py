# ruff: noqa: E402 -- optional training dependencies
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
SAC = pytest.importorskip("stable_baselines3").SAC
from test_articulated_body import fixture

from myumiq_vrchat.articulated_actor import ArticulatedActor
from myumiq_vrchat.articulated_env import ArticulatedGoalEnv
from myumiq_vrchat.articulated_frame import JOINT_WORLD_FRAME
from myumiq_vrchat.articulated_policy import policy_for_frame
from myumiq_vrchat.tracker_basis import RateBasis


def test_checkpoint_restores_frame_updates_and_optimizer_for_further_learning(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_model_based_actor import save_checkpoint

    torch.set_num_threads(1)
    rig, states = fixture()
    decoder = RateBasis(
        action_size=rig.action_size,
        rows=tuple(map(tuple, np.eye(rig.action_size))),
        variance_fraction=1.0,
    )
    env = ArticulatedGoalEnv(rig, states, decoder=decoder)
    policy, kwargs = policy_for_frame(JOINT_WORLD_FRAME, rig)

    def model():
        return SAC(
            policy,
            env,
            device="cpu",
            buffer_size=10,
            policy_kwargs={"net_arch": [32, 32], **kwargs},
        )

    original = model()
    optimizer = torch.optim.Adam(original.actor.parameters(), lr=0.0002)
    obs, _ = env.reset(options={"start": states[0], "goal": states[1]})
    values = torch.tensor(obs[None])

    def update(actor, opt):
        loss = (actor(values, deterministic=True) - 0.2).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    update(original.actor, optimizer)
    original.model_based_updates, original.prior_updates = 100, 4000
    path = save_checkpoint(original, optimizer, rig, decoder, tmp_path, "a" * 64, {"horizon": 1})
    metadata = json.loads((path / "checkpoint.json").read_text("utf-8"))
    assert metadata["model_based_updates"] == 100 and metadata["bc_updates"] == 4000
    assert (
        metadata["source_sha256"] == "a" * 64
        and not metadata["evaluated"]
        and not metadata["promoted"]
    )
    exported = ArticulatedActor(path / "candidate-actor.pt")
    assert exported.manifest["policy_frame"] == JOINT_WORLD_FRAME
    np.testing.assert_allclose(
        exported.predict(obs)[0], original.predict(obs, deterministic=True)[0], atol=1e-6
    )
    state = torch.load(path / "training-state.pt", map_location="cpu", weights_only=True)
    resumed = model()
    resumed.actor.load_state_dict(state["actor"])
    resumed_optimizer = torch.optim.Adam(resumed.actor.parameters())
    resumed_optimizer.load_state_dict(state["optimizer"])
    update(original.actor, optimizer)
    update(resumed.actor, resumed_optimizer)
    for key, expected in original.actor.state_dict().items():
        torch.testing.assert_close(resumed.actor.state_dict()[key], expected, atol=0, rtol=0)
    assert not list(path.glob("*.tmp"))
    env.close()


def test_failed_training_state_write_preserves_the_previous_file(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from types import SimpleNamespace

    import train_model_based_actor as trainer

    path = tmp_path / "training-state.pt"
    path.write_bytes(b"previous complete state")

    def fail_save(value, stream):
        stream.write(b"incomplete")
        raise OSError("interrupted write")

    monkeypatch.setattr(trainer.torch, "save", fail_save)
    actor = SimpleNamespace(state_dict=lambda: {})
    model = SimpleNamespace(actor=actor, model_based_updates=100)
    with pytest.raises(OSError, match="interrupted"):
        trainer.save_training_state(model, actor, path)
    assert path.read_bytes() == b"previous complete state"
