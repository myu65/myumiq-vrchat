"""Build a new constrained corpus identity from verified training motions only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from myumiq_vrchat.articulated_body import ArticulatedRig, JointState
from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.joint_limits import CONTRACT, from_reference, violation
from myumiq_vrchat.motion import GltfMotion


def prepare(source, gltf, output, margin=.1, reference_clips=()):
    report = json.loads((source / "result.json").read_text("utf-8"))
    if report.get("license") != "CC0":
        raise ValueError("verified CC0 corpus required")
    motion = GltfMotion(gltf)
    paths = [gltf] + [gltf.parent / b["uri"] for b in motion.data["buffers"]]
    for path in paths:
        if report.get("source_files", {}).get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("reference motion differs from the source corpus")
    rig = ArticulatedRig.model_validate_json((source / "rig.json").read_text("utf-8"))
    if rig.joint_limits:
        raise ValueError("source already has a joint envelope; use its original corpus")
    rows = json.loads((source / "poses.json").read_text("utf-8"))
    clips = sorted({r["clip"] for r in rows if r["split"] == "train"} | set(reference_clips))
    heldout = {r["clip"] for r in rows if r["split"] == "heldout"}
    if not clips or set(clips) & heldout:
        raise ValueError("training and held-out clips must be disjoint")
    states = [rig.sample(motion, clip, float(t)) for clip in clips
              for t in np.linspace(0, motion.duration(clip), int(np.ceil(motion.duration(clip) * 20)) + 1)]
    limits = from_reference(np.stack([s.rotations for s in states]), margin=margin)
    data = rig.model_dump(mode="json") | {"joint_limits": [x.model_dump(mode="json") if x else None for x in limits]}
    constrained = ArticulatedRig.model_validate_json(json.dumps(data))
    audit = {}
    for split in ("train", "heldout"):
        selected = [JointState(np.array(r["joint_state"]["root"]), np.array(r["joint_state"]["local_orientations"]))
                    for r in rows if r["split"] == split]
        errors = [float(violation(s.rotations, limits).max()) for s in selected]
        audit[split] = {"samples": len(selected), "maximum_violation_rad": max(errors, default=0.),
                        "outside": sum(x > 1e-6 for x in errors)}
    if audit["train"]["outside"]:
        raise ValueError(f"corpus exceeds reference envelope; do not widen from held-out samples: {audit}")
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "rig.json").write_text(constrained.model_dump_json(indent=2), "utf-8")
    (output / "poses.json").write_bytes((source / "poses.json").read_bytes())
    envelope_report = {"contract": CONTRACT, "reference_clips": clips, "reference_samples": len(states),
                       "reference_hz": 20, "margin_rad": margin, "audit": audit,
                       "rig_sha256": hashlib.sha256(constrained.model_dump_json().encode()).hexdigest(),
                       "source_rig_sha256": hashlib.sha256(rig.model_dump_json().encode()).hexdigest(),
                       "scope": "motion_support_not_complete_human_anatomy", "avatar_verified": False}
    (output / "result.json").write_text(json.dumps({"license": "CC0", "source_files": report["source_files"],
        "source": str(source.resolve()), "joint_envelope": envelope_report, "promoted": False}, indent=2), "utf-8")
    return envelope_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gltf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=.1)
    parser.add_argument("--reference-clips", nargs="*", default=[],
                        help="Additional explicitly selected envelope references; must not overlap held-out clips")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.gltf, args.output, args.margin, args.reference_clips)))


if __name__ == "__main__":
    main()
