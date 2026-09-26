"""Validated observed starts for candidate-only articulated policy refinement."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .articulated_body import OBSERVATION_CONTRACT, JointState
from .articulated_controller import close_enough, feedback_matches
from .body import BodyTarget
from .replay import WholeBodyTransition
from .whole_body import PARTS, state_target, vector


@dataclass(frozen=True)
class ExperienceCase:
    session: str
    action: str
    evidence: str
    root: np.ndarray
    joints: np.ndarray
    current: np.ndarray
    goal: np.ndarray
    previous: np.ndarray
    dt: float


def experience_case(record, actor):
    """Keep device observations distinct from the fitted, latent joint estimate."""
    step, meta = record.learning, record.intent_metadata
    if (
        record.environment != "vrchat"
        or record.outcome != "device_feedback"
        or step is None
        or step.reward_scope != "tracker_geometry"
        or step.observation_contract != OBSERVATION_CONTRACT
        or record.policy_id != "articulated:" + actor.manifest["sha256"][:16]
        or meta.get("decoder_id") != actor.decoder.identity
        or meta.get("joint_constraint_contract") != actor.manifest.get("joint_constraint_contract")
        or meta.get("joint_state_source") != "fitted_tracker_estimate"
        or meta.get("intent_generation") is None
    ):
        raise ValueError("unconfirmed or incompatible articulated experience")
    evidence = step.evidence_ids[0]
    if ":readback:" not in evidence:
        raise ValueError("device readback evidence required")
    for observation in (record.observation, record.next_observation):
        signals = [observation.body.signal_for(part) for part in PARTS]
        # Existing recorder stamps the loop before readback; the capture stamps
        # can be a few milliseconds later. Bound the complete capture interval.
        captured = max(observation.timestamp, *(s.timestamp for s in signals))
        if captured - observation.timestamp >= 0.5 or not all(
            s.valid
            and s.connected
            and s.source == "openvr_raw"
            and s.pose is not None
            and 0 <= captured - s.timestamp < 0.5
            for s in signals
        ):
            raise ValueError("fresh device observations required")
    data = meta["joint_state"]
    state = JointState(np.asarray(data["root"]), np.asarray(data["local_orientations"]))
    predicted = actor.rig.forward(
        state
    )  # Also validates dimensions, finiteness and unit quaternions.
    actor.rig.validate_limits(state)
    current = state_target(record.observation.body)
    if not close_enough(current, predicted):
        raise ValueError("fitted joints do not explain observed trackers")
    expected = BodyTarget.model_validate_json(json.dumps(meta["expected_tracker_pose"]))
    if not feedback_matches(expected, state_target(record.next_observation.body), current):
        raise ValueError("following feedback does not confirm this command")
    goal = BodyTarget.model_validate_json(json.dumps(meta["pose_goal"]))
    if not goal.is_full_body:
        raise ValueError("full body goal required")
    previous = np.asarray(meta["previous_rates"])
    if previous.shape != (66,) or not np.isfinite(previous).all() or np.abs(previous).max() > 1:
        raise ValueError("invalid previous tracker rates")
    floor = actor.manifest.get("reference_floor")
    if floor is not None and min(vector(current)[9:, 2].min(), vector(goal)[9:, 2].min()) < floor:
        raise ValueError("experience below the declared tracking floor")
    session = evidence.split(":readback:", 1)[0]
    return ExperienceCase(
        session,
        f"{session}:{meta['intent_generation']}",
        evidence,
        state.root,
        state.rotations,
        vector(current),
        vector(goal),
        previous,
        step.dt,
    )


def load_experience_cases(path, actor, *, per_action=16, max_actions=128):
    """Deterministic reservoir per attempt; never split neighbouring frames for evaluation."""
    if not 1 <= per_action <= 64 or not 1 <= max_actions <= 128:
        raise ValueError("invalid experience sampling bounds")
    if path.stat().st_size > 2_000_000_000:
        raise ValueError("replay file exceeds the bounded import size")
    digest, groups, counts, rejected = hashlib.sha256(), {}, Counter(), Counter()
    seen = set()
    rng = np.random.default_rng(43)
    with path.open("rb") as stream:
        while line := stream.readline(131073):
            digest.update(line)
            if len(line) > 131072:
                raise ValueError("oversized replay record")
            try:
                raw = json.loads(line)
                if not raw.get("learning"):
                    rejected["no_confirmed_learning"] += 1
                    continue
                case = experience_case(WholeBodyTransition.model_validate_json(line), actor)
            except (ValueError, KeyError, TypeError) as exc:
                rejected[type(exc).__name__] += 1
                continue
            if case.evidence in seen:
                rejected["duplicate_readback"] += 1
                continue
            seen.add(case.evidence)
            if case.action not in groups:
                if len(groups) >= max_actions:
                    raise ValueError("replay has too many independent actions for one training job")
                groups[case.action] = []
            bucket = groups[case.action]
            counts[case.action] += 1
            if len(bucket) < per_action:
                bucket.append(case)
            else:
                index = int(rng.integers(counts[case.action]))
                if index < per_action:
                    bucket[index] = case
    cases = [case for bucket in groups.values() for case in bucket]
    if not cases:
        raise ValueError("no compatible confirmed articulated experience")
    return cases, dict(
        replay_sha256=digest.hexdigest(),
        sessions=sorted({c.session for c in cases}),
        actions=len(groups),
        selected=len(cases),
        confirmed=sum(counts.values()),
        rejected=dict(rejected),
    )


def require_disjoint_sessions(training, evaluation):
    if {c.session for c in training} & {c.session for c in evaluation}:
        raise ValueError("training and evaluation must use independent replay sessions")
