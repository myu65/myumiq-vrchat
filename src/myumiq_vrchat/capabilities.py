"""Capability inventory, verified compositions and evidence-scoped statistics."""

import re
from dataclasses import asdict, dataclass, field


def motion_capability(name):
    return name in ("WAVE", "WALK_IN_PLACE") or bool(
        re.fullmatch(r"(?:MOTION|POSTURE)_[A-Z][A-Z0-9_]{0,39}", name)
    )


def posture_capability(name):
    """A persistent whole-body endpoint, distinct from a cyclic gesture."""
    return name in ("STAND", "CROUCH", "SIT", "LIE", "RETURN_TO_REST") or bool(
        re.fullmatch(r"POSTURE_[A-Z][A-Z0-9_]{0,39}", name)
    )


@dataclass
class Capability:
    name: str
    available: bool
    method: str = "needs_design"
    prerequisites: tuple[str, ...] = ()
    status: str = "unavailable"
    successes: int = 0
    failures: int = 0
    unknown: int = 0
    evidence_scope: str = "device_execution_only"
    policy: str | None = None
    outcomes_by_scope: dict = field(default_factory=dict)
    supported_hands: tuple[str, ...] | None = None
    duration_s: float = 5.0
    description: str = ""
    target_sources: tuple[str, ...] | None = None

    def summary(self):
        measured = self.successes + self.failures
        return {
            "name": self.name,
            "description": self.description,
            "available": self.available,
            "status": self.status,
            "prerequisites": list(self.prerequisites),
            "success_rate": self.successes / measured if measured else None,
            "measured_trials": measured,
            "unknown_trials": self.unknown,
            "scope": self.evidence_scope,
            "outcomes_by_scope": self.outcomes_by_scope,
        }


@dataclass
class Resolution:
    actions: list = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    @property
    def route(self):
        return (
            "learning"
            if self.missing
            else (
                "blocked" if self.blockers else "composition" if len(self.actions) > 1 else "skill"
            )
        )


class CapabilityRegistry:
    def __init__(self, walking=False):
        self.items = {}
        for name in (
            "WAIT",
            "WAVE",
            "LOOK_AT",
            "REACH",
            "RETURN_TO_REST",
            "CROUCH",
            "SIT",
            "LIE",
            "STAND",
        ):
            self.items[name] = Capability(name, True, status="procedural")
        self.items["REACH"].method = "vrchat_replay"
        self.items["SIT"].method = "reference_free_rl"
        self.items["WALK_IN_PLACE"] = Capability(
            "WALK_IN_PLACE",
            walking,
            method="imitation",
            status="learned" if walking else "unavailable",
        )
        self.items["HOLD_HAND"] = Capability(
            "HOLD_HAND", True, prerequisites=("REACH",), status="composition"
        )
        self.items["EXPLORE"] = Capability(
            "EXPLORE",
            True,
            prerequisites=("LOOK_AT",),
            status="composition",
            evidence_scope="visual_attention_not_navigation",
        )
        self.items["TALK"] = Capability(
            "TALK", False, status="unavailable", evidence_scope="local_submission_only"
        )
        self.items["EXPLORE_HOME"] = Capability(
            "EXPLORE_HOME",
            False,
            status="requires_private_home_visual_gate",
            evidence_scope="image_change_not_metric_displacement",
        )
        for name, description in (
            ("MOVE_FORWARD", "向きを変えず短く前進し、止まって景色の変化を確認する"),
            ("TURN_LEFT", "短く左へ体ごと旋回する"),
            ("TURN_RIGHT", "短く右へ体ごと旋回する"),
        ):
            self.items[name] = Capability(
                name,
                False,
                description=description,
                status="requires_private_home_visual_gate",
                evidence_scope="image_change_not_metric_displacement",
            )
        for name, method, prerequisites in (
            ("FOLLOW", "visual_navigation", ("TARGET_IDENTITY", "NAVIGATION", "OBSTACLE_DISTANCE")),
            (
                "APPROACH",
                "reference_free_rl",
                ("ROOT_LOCALIZATION", "NAVIGATION", "OBSTACLE_DISTANCE"),
            ),
            (
                "HANDSHAKE",
                "imitation",
                ("REACH", "HOLD_HAND", "GRASP", "CONTACT_FEEDBACK", "PARTNER_CONSENT"),
            ),
            (
                "PAT_HEAD",
                "reference_free_rl",
                (
                    "TARGET_IDENTITY",
                    "TARGET_HEAD_POSE",
                    "REACH",
                    "CONTACT_FEEDBACK",
                    "PARTNER_CONSENT",
                ),
            ),
        ):
            self.items[name] = Capability(name, False, method=method, prerequisites=prerequisites)
        self.seen = {}

    def get(self, name):
        return self.items.setdefault(name, Capability(name, False))

    def observe(self, name, outcome, scope="device_execution_only"):
        item = self.get(name)
        counts = item.outcomes_by_scope.setdefault(
            scope, {"success": 0, "failure": 0, "unknown": 0}
        )
        counts["success" if outcome is True else "failure" if outcome is False else "unknown"] += 1
        item.evidence_scope = (
            scope if len(item.outcomes_by_scope) == 1 else "mixed_see_scope_counts"
        )
        if outcome is True:
            item.successes += 1
        elif outcome is False:
            item.failures += 1
        else:
            item.unknown += 1

    def summary(self):
        summaries = []
        for item in list(self.items.values())[:64]:
            row = item.summary()
            missing = self.missing_prerequisites(item.name)
            row["available"] = self.is_available(item.name)
            if missing:
                row["unavailable_prerequisites"] = missing
                if item.available:
                    row["status"] = "requires_unavailable_capabilities"
            summaries.append(row)
        return summaries

    def is_available(self, name, visiting=frozenset()):
        item = self.items.get(name)
        return bool(
            item
            and item.available
            and name not in visiting
            and all(
                self.is_available(required, visiting | {name}) for required in item.prerequisites
            )
        )

    def missing_prerequisites(self, name):
        item = self.items.get(name)
        return (
            [
                required
                for required in item.prerequisites
                if not self.is_available(required, frozenset({name}))
            ]
            if item
            else []
        )

    def restore_statistics(self, data):
        # Availability comes from executable adapters, never persisted claims.
        for name, stats in data.items():
            item = self.get(name)
            for field_name in ("successes", "failures", "unknown"):
                value = stats.get(field_name, 0)
                if type(value) is not int or not 0 <= value <= 1_000_000:
                    raise ValueError("invalid capability statistics")
                setattr(item, field_name, value)
            item.outcomes_by_scope = stats.get("outcomes_by_scope", {})
            item.evidence_scope = stats.get("evidence_scope", item.evidence_scope)

    def dump(self):
        return {name: asdict(item) for name, item in self.items.items()}

    def resolve(self, purpose, world):
        from .autonomous_body import Intent
        from .gaze import gaze_targets

        result = Resolution()
        for step in purpose.steps:
            item = self.get(step.capability)
            missing_prereqs = self.missing_prerequisites(item.name)
            if step.learn or not self.is_available(item.name):
                result.missing.append(item.name)
                result.blockers.extend(missing_prereqs)
                continue
            if item.name == "EXPLORE":
                if step.hand is not None or step.speech or step.target is not None:
                    result.blockers.append(
                        "EXPLORE selects visible targets itself; use LOOK_AT/TALK for explicit arguments"
                    )
                    continue
                objects = sorted(
                    gaze_targets(world), key=lambda o: (self.seen.get(o.name, 0), o.name)
                )[:3]
                if not objects:
                    result.blockers.append("no_visible_target")
                for obj in objects:
                    result.actions.append(
                        (
                            "LOOK_AT",
                            Intent(skill="LOOK_AT", target=obj.name, duration_s=step.duration_s),
                        )
                    )
                continue
            skill = {"HOLD_HAND": "REACH", "TALK": "WAIT"}.get(item.name, item.name)
            try:
                intent = Intent(
                    skill=skill,
                    hand=step.hand,
                    target=step.target,
                    duration_s=step.duration_s,
                    speech=step.speech,
                )
                intent.validate_world(world)
                if item.target_sources is not None and not any(
                    o.name == intent.target and o.source in item.target_sources
                    for o in world.objects
                ):
                    raise ValueError("target source is unsupported by the current motor")
                if item.supported_hands is not None and intent.hand not in item.supported_hands:
                    raise ValueError("no learned motion for the requested hand")
                if skill == "LOOK_AT" and intent.target not in {
                    o.name for o in gaze_targets(world)
                }:
                    raise ValueError("target lacks fresh confident gaze feedback")
                if item.name == "TALK" and not intent.speech:
                    raise ValueError("TALK requires speech")
                result.actions.append((item.name, intent))
            except ValueError as exc:
                result.blockers.append(f"{item.name}: {str(exc)[:120]}")
        if len(result.actions) > 12:
            result.blockers.append("plan_too_long")
        # Do not execute a prefix of a contact/missing-capability plan as if complete.
        if result.missing or result.blockers:
            result.actions.clear()
        return result
