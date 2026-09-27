"""glTF humanoid motion sampling into canonical targets; no device output."""

import json
import struct
from pathlib import Path

import numpy as np

from .body import BodyTarget, HandTarget, Pose, q_from_matrix

QUATERNIUS = {
    "head": "DEF-head",
    "chest": "DEF-spine.003",
    "pelvis": "DEF-hips",
    "left_hand": "DEF-hand.L",
    "right_hand": "DEF-hand.R",
    "left_elbow": "DEF-forearm.L",
    "right_elbow": "DEF-forearm.R",
    "left_knee": "DEF-shin.L",
    "right_knee": "DEF-shin.R",
    "left_foot": "DEF-foot.L",
    "right_foot": "DEF-foot.R",
}
# Position stays at the elbow joint, orientation follows the upper arm, matching
# the segment used by VRChat's elbow trackers (mount offsets remain calibrated).
ORIENTATION_NODES = {"left_elbow": "DEF-upper_arm.L", "right_elbow": "DEF-upper_arm.R"}


def rotation(q):
    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-8:
        raise ValueError("invalid rotation quaternion")
    x, y, z, w = np.asarray(q) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def interpolate(a, b, fraction, quaternion=False):
    if quaternion:
        dot = float(np.dot(a, b))
        if dot < 0:
            b, dot = -b, -dot
        if dot < 0.9995:
            angle = np.arccos(np.clip(dot, -1, 1))
            value = (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(
                angle
            )
        else:
            value = a + fraction * (b - a)
        return value / np.linalg.norm(value)
    return a + fraction * (b - a)


class GltfMotion:
    def __init__(self, path: Path):
        payload = path.read_bytes()
        embedded = None
        if payload[:4] == b"glTF":
            if len(payload) < 20 or struct.unpack_from("<II", payload, 4) != (2, len(payload)):
                raise ValueError("invalid GLB header")
            chunks, offset = {}, 12
            while offset < len(payload):
                if offset + 8 > len(payload):
                    raise ValueError("truncated GLB chunk")
                size, kind = struct.unpack_from("<II", payload, offset)
                offset += 8
                if offset + size > len(payload) or kind in chunks:
                    raise ValueError("invalid GLB chunk")
                chunks[kind] = payload[offset : offset + size]
                offset += size
            self.data = json.loads(chunks[0x4E4F534A])
            embedded = chunks.get(0x004E4942)
        else:
            self.data = json.loads(payload)
        self.buffers = []
        self.source_files = [path]
        for index, buffer in enumerate(self.data["buffers"]):
            if "uri" not in buffer:
                if index != 0 or embedded is None or len(embedded) < buffer["byteLength"]:
                    raise ValueError("missing embedded GLB buffer")
                self.buffers.append(embedded)
                continue
            resolved = (path.parent / buffer["uri"]).resolve()
            if not resolved.is_relative_to(path.parent.resolve()):
                raise ValueError("buffer must be a local dataset file")
            self.buffers.append(resolved.read_bytes())
            self.source_files.append(resolved)
        self.nodes = self.data["nodes"]
        self.parents = {}
        for i, node in enumerate(self.nodes):
            for child in node.get("children", []):
                if child in self.parents:
                    raise ValueError("node has multiple parents")
                self.parents[child] = i
        self.names = {n.get("name"): i for i, n in enumerate(self.nodes)}
        self.clips = {a["name"]: a for a in self.data["animations"]}

    def accessor(self, index):
        item = self.data["accessors"][index]
        if item["componentType"] != 5126 or "sparse" in item:
            raise ValueError("motion accessor must contain dense float32 values")
        width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[item["type"]]
        view = self.data["bufferViews"][item["bufferView"]]
        return np.ndarray(
            (item["count"], width),
            dtype="<f4",
            buffer=self.buffers[view["buffer"]],
            offset=view.get("byteOffset", 0) + item.get("byteOffset", 0),
            strides=(view.get("byteStride", width * 4), 4),
        ).copy()

    def duration(self, clip):
        return max(float(self.accessor(s["input"])[-1, 0]) for s in self.clips[clip]["samplers"])

    def world_matrices(self, clip, time_s):
        animated = {}
        animation = self.clips[clip]
        for channel in animation["channels"]:
            key = channel["target"]["path"]
            if key not in ("translation", "rotation", "scale"):
                continue
            sampler = animation["samplers"][channel["sampler"]]
            mode = sampler.get("interpolation", "LINEAR")
            if mode not in ("LINEAR", "STEP"):
                raise ValueError(f"unsupported animation interpolation: {mode}")
            times = self.accessor(sampler["input"])[:, 0]
            values = self.accessor(sampler["output"])
            if (
                not len(times)
                or len(times) != len(values)
                or not np.all(np.isfinite(times))
                or not np.all(np.isfinite(values))
                or np.any(np.diff(times) <= 0)
            ):
                raise ValueError("animation samples must be finite with increasing times")
            i = max(0, min(len(times) - 1, int(np.searchsorted(times, time_s, side="right")) - 1))
            j = min(i + 1, len(times) - 1)
            fraction = (
                0.0
                if i == j or mode == "STEP"
                else float(np.clip((time_s - times[i]) / (times[j] - times[i]), 0, 1))
            )
            animated[channel["target"]["node"], key] = interpolate(
                values[i], values[j], fraction, key == "rotation"
            )
        result, visiting = {}, set()

        def world(i):
            if i in result:
                return result[i]
            if i in visiting:
                raise ValueError("cyclic skeleton")
            visiting.add(i)
            node = self.nodes[i]
            if "matrix" in node:
                if any((i, key) in animated for key in ("translation", "rotation", "scale")):
                    raise ValueError("animated matrix nodes are unsupported")
                local = np.array(node["matrix"]).reshape(4, 4).T
            else:
                t = animated.get((i, "translation"), node.get("translation", (0, 0, 0)))
                r = animated.get((i, "rotation"), node.get("rotation", (0, 0, 0, 1)))
                s = animated.get((i, "scale"), node.get("scale", (1, 1, 1)))
                local = np.eye(4)
                local[:3, :3] = rotation(r) @ np.diag(s)
                local[:3, 3] = t
            result[i] = world(self.parents[i]) @ local if i in self.parents else local
            visiting.remove(i)
            return result[i]

        for i in range(len(self.nodes)):
            world(i)
        return result

    def retarget(self, clip, hz=30, head_height=1.6):
        if not 1 <= hz <= 240 or not 0.5 <= head_height <= 2.5:
            raise ValueError("invalid sampling rate or body scale")
        # This explicit profile is for Quaternius Godot glTF: +Z forward,
        # +X anatomical left, +Y up. Other skeletons need their own profile.
        basis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        reference = self.world_matrices("A_TPose", 0)
        ids = {part: self.names[name] for part, name in QUATERNIUS.items()}
        scale = head_height / float(reference[ids["head"]][1, 3])
        frames = []
        duration = self.duration(clip)
        for time_s in np.unique(np.append(np.arange(0, duration, 1 / hz), duration)):
            world = self.world_matrices(clip, float(time_s))
            poses = {}
            for part, index in ids.items():
                rotation_index = (
                    self.names[ORIENTATION_NODES[part]] if part in ORIENTATION_NODES else index
                )
                # Remove tiny exporter scale errors, retain proper rotations.
                u, _, vt = np.linalg.svd(world[rotation_index][:3, :3])
                ru, _, rvt = np.linalg.svd(reference[rotation_index][:3, :3])
                orientation = basis @ (u @ vt) @ (ru @ rvt).T @ basis.T
                poses[part] = Pose(
                    position=tuple(float(v) for v in basis @ world[index][:3, 3] * scale),
                    orientation=q_from_matrix(orientation),
                )
            target = BodyTarget(
                head=poses.pop("head"),
                left=HandTarget(pose=poses.pop("left_hand")),
                right=HandTarget(pose=poses.pop("right_hand")),
                **poses,
            )
            frames.append((float(time_s), target))
        return frames
