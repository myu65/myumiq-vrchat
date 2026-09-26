"""State/goal-conditioned whole-body actor; controller locomotion is separate."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .body import BodyState, BodyTarget
from .tracker_action import ACTION_SHAPE, CONTRACT, integrate_tracker_action
from .whole_body import state_target, vector

OBSERVATION_CONTRACT = "myumiq-tracker-goal-v1"
OBSERVATION_SIZE = 210  # 11 * (goal error 6 + current pose 7 + previous rate 6) + dt
SEGMENTS = ((0, 1), (1, 2), (1, 5), (1, 6), (5, 3), (6, 4), (2, 7), (2, 8), (7, 9), (8, 10))


def pose_error(current: BodyTarget, goal: BodyTarget) -> np.ndarray:
    """Tracking-axis position and shortest-arc rotation vectors, not Euler angles."""
    a, b = vector(current), vector(goal)
    qa, qb = a[:, 3:], b[:, 3:]
    w = np.sum(qa * qb, axis=1)
    # q_goal * conjugate(q_current), in tracking-space axes.
    v = -qb[:, :1] * qa[:, 1:] + qa[:, :1] * qb[:, 1:] - np.cross(qb[:, 1:], qa[:, 1:])
    signs = np.where(w < 0, -1.0, 1.0)
    w, v = w * signs, v * signs[:, None]
    norm = np.linalg.norm(v, axis=1)
    angle = 2 * np.arctan2(norm, np.clip(w, 0, 1))
    rotvec = v * (angle / np.maximum(norm, 1e-12))[:, None]
    return np.concatenate((b[:, :3] - a[:, :3], rotvec), axis=1)


def observation(current: BodyTarget, goal: BodyTarget, previous_rates, dt: float) -> np.ndarray:
    if not np.isfinite(dt) or not 0 < dt <= 0.1:
        raise ValueError("invalid policy timestep")
    previous = np.asarray(previous_rates, dtype=np.float64)
    if previous.size != 66 or not np.isfinite(previous).all() or np.any(np.abs(previous) > 1):
        raise ValueError("invalid previous tracker rates")
    poses = vector(current)
    # q and -q encode the same pose, including in the network's state input.
    poses[:, 3:] *= np.where(poses[:, 3:4] < 0, -1.0, 1.0)
    error = pose_error(current, goal) / np.array([0.6, 0.6, 0.6, 2.0, 2.0, 2.0])
    result = np.concatenate((error.ravel(), poses.ravel(), previous.ravel(), [dt / 0.05]))
    if (
        result.size != OBSERVATION_SIZE
        or not np.isfinite(result).all()
        or np.abs(result).max() > 10
    ):
        raise ValueError("tracker observation outside supported interface")
    return result.astype(np.float32)


def distance(current, goal):
    error = pose_error(current, goal)
    return float(
        np.mean(np.linalg.norm(error[:, :3], axis=1))
        + 0.15 * np.mean(np.linalg.norm(error[:, 3:], axis=1))
    )


def worst_tracker_distance(current, goal):
    """Largest endpoint tolerance ratio, expressed on the position-error scale."""
    error = pose_error(current, goal)
    return 0.12 * max(
        float(np.linalg.norm(error[:, :3], axis=1).max()) / 0.12,
        float(np.linalg.norm(error[:, 3:], axis=1).max()) / 0.35,
    )


def reward_components(before, after, goal, rates, previous):
    old, new = distance(before, goal), distance(after, goal)
    a, b = vector(after), vector(goal)
    distortion = np.mean(
        [
            abs(np.linalg.norm(a[i, :3] - a[j, :3]) - np.linalg.norm(b[i, :3] - b[j, :3]))
            for i, j in SEGMENTS
        ]
    )
    return {
        "progress": 10 * (old - new),
        "pose_error": -0.2 * new,
        "segment_distortion": -0.1 * float(distortion),
        "effort": -0.002 * float(np.mean(rates**2)),
        "action_change": -0.01 * float(np.mean((rates - previous) ** 2)),
    }


class TrackerActor:
    """Inference-only exported actor; loading it does not authorize live promotion."""

    def __init__(self, path: Path):
        import torch

        manifest = json.loads(path.with_suffix(".json").read_text("utf-8"))
        if (
            manifest.get("contract") != CONTRACT
            or manifest.get("observation_contract") != OBSERVATION_CONTRACT
            or manifest.get("observation_size") != OBSERVATION_SIZE
            or manifest.get("action_size") != 66
            or manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("unsupported or changed whole-body actor")
        self.manifest, self.torch = manifest, torch
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.previous = np.zeros(66, dtype=np.float32)
        self.last_latent = None

    def predict(self, values):
        values = np.asarray(values, dtype=np.float32)
        if (
            values.shape != (OBSERVATION_SIZE,)
            or not np.isfinite(values).all()
            or np.abs(values).max() > 10
        ):
            raise ValueError("invalid whole-body actor observation")
        with self.torch.inference_mode():
            output = self.model(self.torch.from_numpy(values[None]))
            if self.manifest.get("latent_action_size"):
                output, latent = output
                self.last_latent = latent.reshape(-1).numpy().copy()
                if (
                    self.last_latent.shape != (self.manifest["latent_action_size"],)
                    or not np.isfinite(self.last_latent).all()
                    or np.abs(self.last_latent).max() > 1
                ):
                    raise ValueError("invalid latent actor action")
            result = output.reshape(-1).numpy().copy()
        if result.shape != (66,) or not np.isfinite(result).all() or np.abs(result).max() > 1:
            raise ValueError("invalid whole-body actor rates")
        return result

    def step(self, body: BodyState, goal: BodyTarget, dt: float):
        current = state_target(body)
        obs = observation(current, goal, self.previous, dt)
        action = self.predict(obs)
        target = integrate_tracker_action(current, action.reshape(ACTION_SHAPE), dt)
        self.previous = action
        return target, obs, action


def export_actor(model, output: Path, decoder=None):
    import torch

    if decoder is not None and decoder.action_size != 66:
        raise ValueError("tracker export requires a 66-rate decoder")
    actor = model.policy.actor
    network = (
        torch.nn.Sequential(actor.features_extractor, actor.latent_pi, actor.mu, torch.nn.Tanh())
        .cpu()
        .eval()
    )
    samples = np.random.default_rng(42).uniform(-1, 1, (64, OBSERVATION_SIZE)).astype(np.float32)
    expected, _ = model.predict(samples, deterministic=True)
    latent_expected = expected
    if decoder is not None:
        from .tracker_basis import torch_decoder

        class DecodedActor(torch.nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor
                self.decoder = torch_decoder(decoder)

            def forward(self, values):
                latent = self.actor(values)
                return self.decoder(latent), latent

        network = DecodedActor(network).eval()
        expected = np.stack([decoder.decode(action) for action in expected])
    traced = torch.jit.trace(network, torch.zeros(1, OBSERVATION_SIZE))
    with torch.inference_mode():
        actual = traced(torch.from_numpy(samples))
        if decoder is not None:
            actual, latent = actual
            if not np.allclose(latent.numpy(), latent_expected, atol=1e-6, rtol=1e-6):
                raise ValueError("exported latent action differs from SAC")
        actual = actual.numpy()
    if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise ValueError("whole-body actor export differs from SAC")
    traced.save(str(output))
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "contract": CONTRACT,
                "observation_contract": OBSERVATION_CONTRACT,
                "observation_size": OBSERVATION_SIZE,
                "action_size": 66,
                "latent_action_size": len(decoder.rows) if decoder else None,
                "decoder_id": decoder.identity if decoder else None,
                "decoder": decoder.model_dump(mode="json") if decoder else None,
                "algorithm": getattr(model, "algorithm_label", "SAC"),
                "sac_updates": int(model._n_updates),
                "motion_bc_updates": int(getattr(model, "prior_updates", 0)),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "training_scope": "tracker_geometry",
                "promoted": False,
                "avatar_verified": False,
                "world_locomotion_verified": False,
            },
            indent=2,
        ),
        "utf-8",
    )
