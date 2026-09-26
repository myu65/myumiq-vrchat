"""Inference-only learned REACH actor; the LLM never supplies these motor actions."""

import json
import math
from pathlib import Path


class LearnedReachPolicy:
    def __init__(self, path: Path):
        import torch

        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (
            manifest.get("contract") != "myumiq-reach-v1"
            or manifest.get("hand") != "right"
            or manifest.get("observation_size") != 9
            or manifest.get("action_size") != 3
        ):
            raise ValueError("unsupported learned reach policy contract")
        self._torch = torch
        self._model = torch.jit.load(str(path), map_location="cpu").eval()
        self.predict((0.0,) * 9)

    def predict(self, observation: tuple[float, ...]) -> tuple[float, float, float]:
        if len(observation) != 9 or not all(math.isfinite(x) and abs(x) <= 2 for x in observation):
            raise ValueError("reach observations outside the trained interface")
        with self._torch.inference_mode():
            tensor = self._torch.tensor([observation], dtype=self._torch.float32)
            action = self._model(tensor).reshape(-1).tolist()
        if len(action) != 3 or not all(math.isfinite(x) and -1 <= x <= 1 for x in action):
            raise ValueError("learned reach policy emitted invalid action")
        return tuple(action)


def export_reach_actor(model, output: Path) -> None:
    """Export the deterministic SAC actor without critics or a trainer at runtime."""
    import numpy as np
    import torch

    actor = model.policy.actor
    network = (
        torch.nn.Sequential(
            actor.features_extractor,
            actor.latent_pi,
            actor.mu,
            torch.nn.Tanh(),
        )
        .cpu()
        .eval()
    )
    observations = np.random.default_rng(1).uniform(-0.6, 0.6, (100, 9)).astype(np.float32)
    expected, _ = model.predict(observations, deterministic=True)
    traced = torch.jit.trace(network, torch.zeros(1, 9))
    with torch.inference_mode():
        actual = traced(torch.from_numpy(observations)).numpy()
    if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise ValueError("exported reach actor disagrees with the trained policy")
    traced.save(str(output))
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "contract": "myumiq-reach-v1",
                "hand": "right",
                "observation_size": 9,
                "action_size": 3,
                "algorithm": "SAC",
                "training_environment": "unity",
                "vrchat_transfer_verified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
