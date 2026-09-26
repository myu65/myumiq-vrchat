"""Offline canonical skeleton comparison, not an observed VRChat avatar."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from myumiq_vrchat.body import BodyTarget
from myumiq_vrchat.postures import posture_target

EDGES = (("head", "chest"), ("chest", "pelvis"),
         ("chest", "left_elbow"), ("left_elbow", "left_hand"),
         ("chest", "right_elbow"), ("right_elbow", "right_hand"),
         ("pelvis", "left_knee"), ("left_knee", "left_foot"),
         ("pelvis", "right_knee"), ("right_knee", "right_foot"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--postures", action="store_true")
    args = parser.parse_args()
    names = ("standing", "crouching", "sitting_floor", "lying")
    if args.postures:
        teacher = [{"target": posture_target(n).model_dump(mode="json")} for n in names]
        learned = None
    else:
        teacher = [json.loads(s) for s in (args.run / "teacher.jsonl").read_text().splitlines()]
        learned = [json.loads(s) for s in (args.run / "learned.jsonl").read_text().splitlines()]
    fig = plt.figure(figsize=(12, 4))
    for col, fraction in enumerate((0, 0.25, 0.5, 0.75), 1):
        ax = fig.add_subplot(1, 4, col, projection="3d")
        index = col - 1 if args.postures else round(fraction * (len(teacher)-1))
        for rows, color, style in ((teacher, "#1265b0", "-"), (learned, "#ec7320", "--")):
            if rows is None:
                continue
            target = BodyTarget.model_validate_json(json.dumps(rows[index]["target"]))
            for start, end in EDGES:
                a, b = target.pose_for(start).position, target.pose_for(end).position
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], style, color=color)
        ax.set(xlim=(-0.9, 0.9), ylim=(-0.7, 0.7), zlim=(0, 1.8),
               xlabel="Forward (m)", ylabel="Left (m)",
               title=names[index] if args.postures else f"Phase {fraction:.2f}")
        ax.set_box_aspect((1.4, 1.4, 1.8))
        ax.view_init(elev=15, azim=-55)
    fig.suptitle("Predicted FK diagnostic poses (not VRChat observations)" if args.postures else
                 "Offline Walk_Loop: CC0 teacher (blue), fitted policy (orange dashed)")
    fig.tight_layout()
    fig.savefig(args.run / ("postures.png" if args.postures else "comparison.png"), dpi=160)


if __name__ == "__main__":
    main()
