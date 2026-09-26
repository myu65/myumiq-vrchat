"""Kinematic joint-rate actuator using an explicit imported skeleton, without physics."""

from dataclasses import dataclass

import numpy as np
from pydantic import model_serializer, model_validator

from .body import Frozen, Number, q_from_matrix, qmul, rotate
from .joint_limits import JointLimit, project, violation
from .motion import ORIENTATION_NODES, QUATERNIUS
from .tracker_policy import observation as tracker_observation
from .tracker_policy import pose_error
from .whole_body import PARTS, target_from_vector

CANONICAL_BASIS = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
OBSERVATION_CONTRACT = "myumiq-articulated-goal-v1"


def articulated_observation(current, goal, previous, dt, rotations):
    q = np.asarray(rotations).copy()
    q *= np.where(q[:, :1] < 0, -1, 1)
    return np.concatenate((tracker_observation(current, goal, previous, dt), q.ravel())).astype(
        np.float32
    )


def inverse(q):
    return (q[0], -q[1], -q[2], -q[3])


def rotation_vectors(current, goal):
    qa, qb = np.asarray(current), np.asarray(goal)
    w = np.sum(qa * qb, axis=1)
    v = -qb[:, :1] * qa[:, 1:] + qa[:, :1] * qb[:, 1:] - np.cross(qb[:, 1:], qa[:, 1:])
    sign = np.where(w < 0, -1.0, 1.0)
    w, v = w * sign, v * sign[:, None]
    norm = np.linalg.norm(v, axis=1)
    return v * (2 * np.arctan2(norm, np.clip(w, 0, 1)) / np.maximum(norm, 1e-12))[:, None]


def increment(rotations, rates, dt):
    norm = np.linalg.norm(rates, axis=1)
    angle = norm * dt
    xyz = rates * (np.sin(angle / 2) / np.maximum(norm, 1e-12))[:, None]
    w = np.cos(angle / 2)[:, None]
    qw, qv = rotations[:, :1], rotations[:, 1:]
    result = np.concatenate(
        (w * qw - np.sum(xyz * qv, axis=1)[:, None], w * qv + qw * xyz + np.cross(xyz, qv)), axis=1
    )
    return result / np.linalg.norm(result, axis=1)[:, None]


@dataclass(frozen=True)
class JointState:
    root: np.ndarray
    rotations: np.ndarray


class ArticulatedRig(Frozen):
    names: tuple[str, ...]
    parents: tuple[int, ...]
    offsets: tuple[tuple[Number, Number, Number], ...]
    reference_rotations: tuple[tuple[Number, ...], ...]
    tracker_nodes: tuple[int, ...]
    orientation_nodes: tuple[int, ...]
    scale: Number
    joint_limits: tuple[JointLimit | None, ...] = ()

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        result = handler(self)
        if not self.joint_limits:
            result.pop("joint_limits", None)  # Keep legacy rig identities stable.
        return result

    @model_validator(mode="after")
    def structure(self):
        count = len(self.names)
        if (
            not 2 <= count <= 64
            or len(set(self.names)) != count
            or len(self.parents) != count
            or len(self.offsets) != count
            or len(self.reference_rotations) != count
            or self.scale <= 0
            or self.parents[0] != -1
            or any(not 0 <= p < i for i, p in enumerate(self.parents[1:], 1))
            or len(self.tracker_nodes) != 11
            or len(self.orientation_nodes) != 11
            or any(not 0 <= n < count for n in self.tracker_nodes + self.orientation_nodes)
            or any(len(r) != 9 for r in self.reference_rotations)
            or self.joint_limits
            and (
                len(self.joint_limits) != count
                or self.joint_limits[0] is not None
                or any(limit is None for limit in self.joint_limits[1:])
            )
        ):
            raise ValueError("invalid articulated rig")
        return self

    @classmethod
    def from_motion(cls, motion, head_height=1.6):
        if not 0.5 <= head_height <= 2.5:
            raise ValueError("invalid reference height")
        root = motion.names["DEF-hips"]
        selected = {root}
        for name in QUATERNIUS.values():
            node = motion.names[name]
            while node != root:
                selected.add(node)
                node = motion.parents[node]

        def depth(node):
            result = 0
            while node != root:
                result += 1
                node = motion.parents[node]
            return result

        ids = sorted(selected, key=lambda n: (depth(n), n))
        index = {node: i for i, node in enumerate(ids)}
        ref = motion.world_matrices("A_TPose", 0)
        scale = head_height / float(ref[motion.names["DEF-head"]][1, 3])
        points = [CANONICAL_BASIS @ ref[n][:3, 3] * scale for n in ids]
        parents = [-1 if n == root else index[motion.parents[n]] for n in ids]
        rotations = []
        for n in ids:
            u, _, vt = np.linalg.svd(ref[n][:3, :3])
            rotations.append(tuple(float(x) for x in (u @ vt).ravel()))
        return cls(
            names=tuple(motion.nodes[n]["name"] for n in ids),
            parents=tuple(parents),
            offsets=tuple(
                tuple(float(x) for x in (points[i] - points[p] if p >= 0 else np.zeros(3)))
                for i, p in enumerate(parents)
            ),
            reference_rotations=tuple(rotations),
            scale=scale,
            tracker_nodes=tuple(index[motion.names[QUATERNIUS[p]]] for p in PARTS),
            orientation_nodes=tuple(
                index[motion.names[ORIENTATION_NODES.get(p, QUATERNIUS[p])]] for p in PARTS
            ),
        )

    @property
    def action_size(self):
        return 3 + 3 * len(self.names)

    def sample(self, motion, clip, time_s):
        world = motion.world_matrices(clip, time_s)
        global_q = []
        for name, reference in zip(self.names, self.reference_rotations):
            u, _, vt = np.linalg.svd(world[motion.names[name]][:3, :3])
            delta = (
                CANONICAL_BASIS
                @ (u @ vt)
                @ np.asarray(reference).reshape(3, 3).T
                @ CANONICAL_BASIS.T
            )
            global_q.append(q_from_matrix(delta))
        local = [
            qmul(inverse(global_q[p]), q) if p >= 0 else q for p, q in zip(self.parents, global_q)
        ]
        return JointState(
            CANONICAL_BASIS @ world[motion.names[self.names[0]]][:3, 3] * self.scale,
            np.array(local),
        )

    def forward(self, state):
        if (
            state.root.shape != (3,)
            or state.rotations.shape != (len(self.names), 4)
            or not np.isfinite(state.root).all()
            or not np.isfinite(state.rotations).all()
            or not np.allclose(np.linalg.norm(state.rotations, axis=1), 1, atol=1e-6)
        ):
            raise ValueError("invalid joint state")
        positions, rotations = [], []
        for i, parent in enumerate(self.parents):
            local = tuple(state.rotations[i])
            if parent < 0:
                positions.append(state.root)
                rotations.append(local)
            else:
                positions.append(
                    positions[parent] + np.asarray(rotate(rotations[parent], self.offsets[i]))
                )
                rotations.append(qmul(rotations[parent], local))
        return target_from_vector(
            np.array(
                [
                    (*positions[p], *rotations[q])
                    for p, q in zip(self.tracker_nodes, self.orientation_nodes)
                ]
            )
        )

    def rates_between(self, before, after, dt):
        return (
            np.concatenate(
                (
                    (after.root - before.root) / 0.6,
                    (rotation_vectors(before.rotations, after.rotations) / 2).ravel(),
                )
            )
            / dt
        )

    def validate_limits(self, state):
        if float(violation(state.rotations, self.joint_limits).max()) > 1e-6:
            raise ValueError("joint state is outside the configured joint envelope")

    def advance(self, state, action, dt):
        action = np.asarray(action, dtype=float)
        if (
            action.shape != (self.action_size,)
            or not np.isfinite(action).all()
            or np.abs(action).max() > 1
            or not np.isfinite(dt)
            or not 0 < dt <= 0.1
        ):
            raise ValueError("invalid articulated action")
        before = self.forward(state)
        self.validate_limits(state)
        scale = 1.0 / max(1.0, float(np.linalg.norm(action.reshape(-1, 3), axis=1).max()))
        for _ in range(12):
            proposal = JointState(
                state.root + action[:3] * (0.6 * dt * scale),
                project(
                    increment(state.rotations, action[3:].reshape(-1, 3) * (2 * scale), dt),
                    self.joint_limits,
                ),
            )
            target = self.forward(proposal)
            physical = pose_error(before, target) / np.array([0.6, 0.6, 0.6, 2, 2, 2]) / dt
            joint_rates = self.rates_between(state, proposal, dt)
            maximum = max(
                float(np.linalg.norm(physical.reshape(-1, 3), axis=1).max()),
                float(np.linalg.norm(joint_rates.reshape(-1, 3), axis=1).max()),
            )
            if maximum <= 1.0:
                return proposal, target, physical.ravel(), scale
            scale *= 0.999 / maximum
        raise ValueError("could not satisfy coordinated tracker speed limit")
