"""Atomic actor and optimizer snapshot for model-based refinement."""

import os

import torch


def save_training_state(model, optimizer, path):
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as stream:
        torch.save(
            {
                "actor": model.actor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_based_updates": model.model_based_updates,
                "prior_updates": int(getattr(model, "prior_updates", 0)),
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
