"""SAC actor with a general zero-goal-error equilibrium and separate inference export."""

import hashlib
import json

import numpy as np
import torch
from stable_baselines3.sac.policies import Actor, SACPolicy

from .articulated_actor import ArticulatedActor as ArticulatedActor
from .articulated_env import CONTRACT
from .articulated_frame import (
    FRAME,
    JOINT_WORLD_FRAME,
    JointWorldFrame,
    canonical_observation,
    tracking_action,
)
from .joint_limits import CONTRACT as LIMIT_CONTRACT


def zero_goal(values):
    return torch.cat((torch.zeros_like(values[:, :66]), values[:, 66:]), dim=1)


class AnchoredActor(Actor):
    def get_action_dist_params(self, obs):
        if self.use_sde:
            raise ValueError("anchored actor uses diagonal Gaussian exploration")
        mean, log_std, kwargs = super().get_action_dist_params(obs)
        reference = self.extract_features(zero_goal(obs), self.features_extractor)
        return mean - self.mu(self.latent_pi(reference)), log_std, kwargs


class AnchoredSACPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return AnchoredActor(**kwargs).to(self.device)


class CanonicalAnchoredActor(AnchoredActor):
    def forward(self, obs, deterministic=False):
        if not deterministic:
            raise ValueError("body-frame actor supports deterministic optimization only")
        canonical, turn = canonical_observation(obs)
        return tracking_action(super().forward(canonical, deterministic=True), turn)

    def action_log_prob(self, obs):
        raise ValueError("body-frame actor does not implement a stochastic SAC density")


class CanonicalJointPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CanonicalAnchoredActor(**kwargs).to(self.device)


class JointWorldActor(CanonicalAnchoredActor):
    def __init__(self, *args, rig, **kwargs):
        super().__init__(*args, **kwargs)
        self.body_frame = JointWorldFrame(rig)

    def forward(self, obs, deterministic=False):
        if not deterministic:
            raise ValueError("body-frame actor supports deterministic optimization only")
        features, turn, parent_q = self.body_frame.encode(obs)
        action = AnchoredActor.forward(self, features, deterministic=True)
        return self.body_frame.decode(action, turn, parent_q)


class JointWorldPolicy(SACPolicy):
    def __init__(self, *args, rig, **kwargs):
        self.rig_definition = rig
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return JointWorldActor(**kwargs, rig=self.rig_definition).to(self.device)


def policy_for_frame(frame, rig):
    if frame == "tracking":
        return AnchoredSACPolicy, {}
    if frame == FRAME:
        return CanonicalJointPolicy, {}
    if frame == JOINT_WORLD_FRAME:
        return JointWorldPolicy, {"rig": rig}
    raise ValueError("unsupported training policy frame")


class ExportedJointNetwork(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.net = torch.nn.Sequential(actor.features_extractor, actor.latent_pi, actor.mu)
        self.canonical_frame = isinstance(actor, CanonicalAnchoredActor)
        self.body_frame = actor.body_frame if isinstance(actor, JointWorldActor) else None

    def forward(self, values):
        if self.body_frame is not None:
            features, turn, parent_q = self.body_frame.encode(values)
            action = torch.tanh(self.net(features) - self.net(zero_goal(features)))
            return self.body_frame.decode(action, turn, parent_q)
        if self.canonical_frame:
            canonical, turn = canonical_observation(values)
            return tracking_action(
                torch.tanh(self.net(canonical) - self.net(zero_goal(canonical))), turn
            )
        return torch.tanh(self.net(values) - self.net(zero_goal(values)))


def export_articulated(model, path, rig, decoder):
    count = 210 + 4 * len(rig.names)
    if decoder.action_size != rig.action_size or model.observation_space.shape != (count,):
        raise ValueError("actor, rig and decoder do not agree")
    canonical = isinstance(model.actor, CanonicalAnchoredActor)
    if canonical and not np.array_equal(np.asarray(decoder.rows), np.eye(rig.action_size)):
        raise ValueError("body-frame actor requires an identity joint decoder")
    network = ExportedJointNetwork(model.actor).cpu().eval()
    samples = np.random.default_rng(41).uniform(-1, 1, (64, count)).astype(np.float32)
    expected, _ = model.predict(samples, deterministic=True)
    traced = torch.jit.trace(network, torch.zeros(1, count))
    with torch.inference_mode():
        actual = traced(torch.from_numpy(samples)).numpy()
    if not np.allclose(actual, expected, atol=1e-6):
        raise ValueError("exported articulated actor differs from SAC")
    traced.save(str(path))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "contract": CONTRACT,
                "observation_size": count,
                "policy_frame": (
                    JOINT_WORLD_FRAME
                    if isinstance(model.actor, JointWorldActor)
                    else FRAME
                    if canonical
                    else "tracking"
                ),
                "net_arch": model.actor.net_arch,
                "latent_action_size": len(decoder.rows),
                "rig": rig.model_dump(mode="json"),
                "rig_sha256": hashlib.sha256(rig.model_dump_json().encode()).hexdigest(),
                "joint_constraint_contract": LIMIT_CONTRACT if rig.joint_limits else None,
                "decoder": decoder.model_dump(mode="json"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "algorithm": getattr(model, "algorithm_label", "SAC")
                + "; goal-anchored deterministic mean",
                "gradient_updates": model._n_updates,
                "motion_bc_updates": int(getattr(model, "prior_updates", 0)),
                "model_based_updates": int(getattr(model, "model_based_updates", 0)),
                "reference_floor": getattr(model, "reference_floor", None),
                "reference_floor_weight": getattr(model, "floor_weight", 20.0),
                "reference_floor_power": getattr(model, "floor_power", 2),
                "worst_tracker_weight": getattr(model, "worst_tracker_weight", 0.0),
                "scope": "ideal_articulated_tracker_actuator_not_avatar",
                "promoted": False,
                "avatar_verified": False,
            },
            indent=2,
        ),
        "utf-8",
    )
