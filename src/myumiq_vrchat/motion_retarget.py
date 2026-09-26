"""Explicit cross-skeleton reference transfer for offline motion demonstrations."""

import numpy as np
from pydantic import Field, model_validator

from .articulated_body import ArticulatedRig, JointState, inverse
from .body import Frozen, Number, q_from_matrix, qmul
from .motion import interpolate, rotation


class RetargetProfile(Frozen):
    rig: ArticulatedRig
    reference_root: tuple[Number, Number, Number]
    reference_rotations: tuple[tuple[Number, Number, Number, Number], ...]
    source_nodes: dict[str, int]
    source_reference_clip: str = Field(min_length=1)
    source_reference_time: Number = Field(default=0.0, ge=0)
    source_root: int = Field(ge=0)
    basis: tuple[tuple[Number, Number, Number], ...]
    translation_scale: Number = Field(gt=0)
    motion_scale: Number = Field(default=1.0, gt=0, le=1)

    @model_validator(mode="after")
    def valid(self):
        basis = np.asarray(self.basis)
        if (
            basis.shape != (3, 3)
            or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(basis), 1.0)
            or set(self.source_nodes) != set(self.rig.names)
            or any(n < 0 for n in self.source_nodes.values())
        ):
            raise ValueError("complete node mapping and proper coordinate rotation required")
        self.rig.forward(
            JointState(np.asarray(self.reference_root), np.asarray(self.reference_rotations))
        )
        return self

    def retarget(self, motion, clip, hz=60):
        if not 1 <= hz <= 240 or max(self.source_root, *self.source_nodes.values()) >= len(
            motion.nodes
        ):
            raise ValueError("invalid retarget sampling or source node")
        reference = motion.world_matrices(self.source_reference_clip, self.source_reference_time)
        basis = np.asarray(self.basis)
        target_global = []
        for parent, local in zip(self.rig.parents, self.reference_rotations):
            target_global.append(qmul(target_global[parent], local) if parent >= 0 else local)

        def proper(matrix):
            u, _, vt = np.linalg.svd(matrix[:3, :3])
            result = u @ vt
            if np.linalg.det(result) < 0:
                raise ValueError("reflected source skeleton unsupported")
            return result

        refs = {n: proper(reference[n]) for n in set(self.source_nodes.values())}
        duration = motion.duration(clip)
        times = np.unique(np.append(np.arange(0, duration, 1 / hz), duration))
        frames = []
        for time_s in times:
            world = motion.world_matrices(clip, float(time_s))
            global_q = []
            for i, name in enumerate(self.rig.names):
                n = self.source_nodes[name]
                base = rotation((*target_global[i][1:], target_global[i][0]))
                delta = basis @ proper(world[n]) @ refs[n].T @ basis.T
                full = q_from_matrix(delta @ base)
                global_q.append(
                    tuple(
                        interpolate(
                            np.asarray(target_global[i]),
                            np.asarray(full),
                            self.motion_scale,
                            quaternion=True,
                        )
                    )
                )
            local = [
                qmul(inverse(global_q[p]), q) if p >= 0 else q
                for p, q in zip(self.rig.parents, global_q)
            ]
            displacement = world[self.source_root][:3, 3] - reference[self.source_root][:3, 3]
            root = (
                np.asarray(self.reference_root)
                + basis @ displacement * self.translation_scale * self.motion_scale
            )
            frames.append((float(time_s), self.rig.forward(JointState(root, np.asarray(local)))))
        return frames
