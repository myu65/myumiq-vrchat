"""Retarget and fit one CC0 periodic clip; no live output or automatic promotion."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from myumiq_vrchat.motion import GltfMotion
from myumiq_vrchat.motion_prior import FiniteImitation
from myumiq_vrchat.whole_body import PeriodicImitation, vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gltf", type=Path)
    parser.add_argument("--license", required=True, type=Path)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--author", default=None)
    parser.add_argument("--clip", default="Walk_Loop")
    parser.add_argument("--kind", choices=("periodic", "finite"), default="periodic")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--retarget", type=Path, help="explicit cross-skeleton reference profile")
    args = parser.parse_args()
    license_text = args.license.read_text(encoding="utf-8")
    if "CC0" not in license_text:
        raise ValueError("this initial pipeline requires a supplied CC0 license")
    args.out.mkdir(parents=True, exist_ok=False)
    motion = GltfMotion(args.gltf)
    if args.retarget:
        from .motion_retarget import RetargetProfile

        profile = RetargetProfile.model_validate_json(args.retarget.read_text("utf-8"))
        frames = profile.retarget(motion, args.clip, hz=60)
    else:
        frames = motion.retarget(args.clip, hz=60)
    duration = motion.duration(args.clip)
    # Interleaved held-out times measure interpolation of this clip, NOT transfer
    # to other motions/avatars. Exclude duplicate periodic endpoint from training.
    if args.kind == "finite":
        # Both boundaries belong to training; withheld interior times remain unseen.
        train = [f for i, f in enumerate(frames) if i % 4 != 0 or i in (0, len(frames) - 1)]
        heldout = [f for i, f in enumerate(frames) if i % 4 == 0 and i not in (0, len(frames) - 1)]
        policy = FiniteImitation.fit(train, duration, args.clip)
    else:
        train = [f for i, f in enumerate(frames) if i % 4 != 0 and f[0] < duration - 1e-5]
        heldout = [f for i, f in enumerate(frames) if i % 4 == 0]
        policy = PeriodicImitation.fit(train, duration, args.clip)
    policy.save(args.out / "policy.json")
    baseline = np.mean(np.stack([vector(t) for _, t in train]), axis=0)
    position_errors, baseline_errors, angles = [], [], []
    for t, target in heldout:
        a, b = vector(target), vector(policy.sample(t / duration))
        position_errors.extend(np.linalg.norm(a[:, :3] - b[:, :3], axis=1))
        baseline_errors.extend(np.linalg.norm(a[:, :3] - baseline[:, :3], axis=1))
        angles.extend(
            np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(a[:, 3:] * b[:, 3:], axis=1)), 0, 1)))
        )
    for name, sequence in (
        ("teacher", frames),
        ("learned", [(t, policy.sample(t / duration)) for t, _ in frames]),
    ):
        with (args.out / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            for t, target in sequence:
                stream.write(
                    json.dumps({"time_s": t, "target": target.model_dump(mode="json")}) + "\n"
                )
    files = motion.source_files + [args.license] + ([args.retarget] if args.retarget else [])
    report = {
        "source_url": args.source_url,
        "author": args.author,
        "license": "CC0-1.0",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "profile": (
            "explicit destination-rig reference transfer"
            if args.retarget
            else "Quaternius Godot glTF, head height 1.6m, uniform source proportions"
        ),
        "clip": args.clip,
        "duration_s": duration,
        "train_frames": len(train),
        "heldout_frames": len(heldout),
        "kind": args.kind,
        "method": ("Fourier" if args.kind == "periodic" else "Gaussian basis")
        + " phase-conditioned ridge behavior cloning",
        "position_mean_cm": float(np.mean(position_errors) * 100),
        "position_max_cm": float(np.max(position_errors) * 100),
        "baseline_position_mean_cm": float(np.mean(baseline_errors) * 100),
        "rotation_mean_deg": float(np.mean(angles)),
        "rotation_max_deg": float(np.max(angles)),
        "evaluation_scope": "held-out times from same motion; no physics/contact/VRChat validation",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
