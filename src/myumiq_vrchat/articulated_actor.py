"""Inference-only articulated actor loader; no trainer or Gymnasium dependency."""

import hashlib
import json

import numpy as np
import torch

from .articulated_body import OBSERVATION_CONTRACT as CONTRACT


class ArticulatedActor:
    """Reads an explicit candidate; does not fit live body state or authorize its use."""

    def __init__(self, path):
        from .articulated_body import ArticulatedRig
        from .joint_limits import CONTRACT as LIMIT_CONTRACT
        from .tracker_basis import RateBasis

        self.manifest = json.loads(path.with_suffix(".json").read_text("utf-8"))
        if (
            self.manifest.get("contract") != CONTRACT
            or self.manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("unsupported or changed articulated actor")
        self.rig = ArticulatedRig.model_validate_json(json.dumps(self.manifest["rig"]))
        expected_contract = LIMIT_CONTRACT if self.rig.joint_limits else None
        if self.manifest.get("joint_constraint_contract") != expected_contract or (
            self.rig.joint_limits
            and self.manifest.get("rig_sha256")
            != hashlib.sha256(self.rig.model_dump_json().encode()).hexdigest()
        ):
            raise ValueError("articulated actor joint constraint contract or rig digest differs")
        self.decoder = RateBasis.model_validate_json(json.dumps(self.manifest["decoder"]))
        self.size = 210 + len(self.rig.names) * 4
        if (
            self.manifest.get("observation_size") != self.size
            or self.manifest.get("latent_action_size") != len(self.decoder.rows)
            or self.decoder.action_size != self.rig.action_size
        ):
            raise ValueError("inconsistent articulated actor interfaces")
        self.model = torch.jit.load(str(path), map_location="cpu").eval()

    def predict(self, obs, deterministic=True):
        values = np.asarray(obs, dtype=np.float32)
        if (
            not deterministic
            or values.shape != (self.size,)
            or not np.isfinite(values).all()
            or np.abs(values).max() > 10
        ):
            raise ValueError("invalid articulated inference request")
        with torch.inference_mode():
            rates = self.model(torch.from_numpy(values[None])).numpy().reshape(-1).copy()
        if (
            rates.shape != (len(self.decoder.rows),)
            or not np.isfinite(rates).all()
            or np.abs(rates).max() > 1
        ):
            raise ValueError("invalid articulated actor output")
        return rates, None
