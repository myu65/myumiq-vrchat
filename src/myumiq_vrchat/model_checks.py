"""Exercise real cognitive contracts with synthetic data and no device authority."""

import os
import time
from dataclasses import asdict

from .autonomy import Drives
from .body import WorldState
from .capabilities import CapabilityRegistry
from .decision import Candidate, DecisionInput, make_scorer
from .decision_selection import LocalSelection
from .purposes import request_purpose
from .shared_dialogue import request_dialogue


def _synthetic_visual():
    """A bounded PNG fixture without camera access or optional vision dependencies."""
    import base64
    import struct
    import zlib

    from .visual_context import VisualContext

    def chunk(kind, data):
        return (
            struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
        )

    rows = (b"\0" + bytes((255, 0, 0)) * 64 + bytes((0, 0, 255)) * 64) * 96
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", 128, 96, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return VisualContext(
        WorldState(), base64.b64encode(png).decode(), time.perf_counter(), image_mime="image/png"
    )


def _credentials(profile):
    if getattr(profile, "adapter", None) == "openrouter_chat" or getattr(
        profile, "backend", None
    ) in ("jev", "jev_openrouter"):
        return profile.api_key_env if not os.environ.get(profile.api_key_env) else None
    return None


def _speech(config, use_image=False):
    context = {"working": {"turns": []}, "commitment": None}
    replies, metadata = [], []
    checks = ["Japanese", "current_turn"]
    if use_image:
        result = request_dialogue(
            config,
            WorldState(),
            asdict(Drives()),
            {},
            context,
            "画像の左右の色を教えて。",
            visual=_synthetic_visual(),
        )
        if "赤" not in result.reply or "青" not in result.reply:
            raise ValueError("dialogue did not describe the synthetic image")
        metadata.append(result._inference)
        checks.append("image_input")
    for token in ("あさがお", "ひまわり"):
        text = f"今の合言葉を『{token}』に変更します。今の合言葉だけを日本語で返して。"
        result = request_dialogue(config, WorldState(), asdict(Drives()), {}, context, text)
        if token not in result.reply or (replies and "あさがお" in result.reply):
            raise ValueError("dialogue did not answer the current synthetic turn")
        replies.append(result.reply)
        metadata.append(result._inference)
        context["working"]["turns"].extend(
            [{"role": "user", "text": text}, {"role": "assistant", "text": result.reply}]
        )
    return {"replies": replies, "inference": metadata, "checks": checks}


def _thought(config, use_image=False):
    registry = CapabilityRegistry()
    result = request_purpose(
        config,
        WorldState(),
        Drives(),
        {},
        [],
        registry.summary(),
        [],
        {"working": {"turns": []}, "commitment": None},
        **({"visual": _synthetic_visual()} if use_image else {}),
    )
    return {
        "description": result.description,
        "steps": [step.capability for step in result.steps],
        "inference": result._inference,
        "checks": ["purpose_schema"],
        "executed": False,
    }


def _decision(settings):
    selections, metadata = [], []
    for skill, utterance in (("CROUCH", "しゃがんで"), ("STAND", "今度は立って")):
        visual = (
            _synthetic_visual() if settings.selection and settings.selection.use_image else None
        )
        request = DecisionInput(
            captured_at=time.perf_counter(),
            image_base64=visual.image_base64 if visual else None,
            image_mime="image/png",
            candidates=tuple(
                Candidate(id=name.lower(), description=description, intent={"skill": name})
                for name, description in (
                    ("WAIT", "現在の姿勢を保持する"),
                    ("CROUCH", "MyuMIQがしゃがむ"),
                    ("STAND", "MyuMIQが立ち上がる"),
                )
            ),
            state={
                "latest_user_utterance": utterance,
                "conversation": [{"role": "user", "text": utterance, "latest_utterance": True}],
                "capabilities": [
                    {"name": n, "available": True} for n in ("WAIT", "CROUCH", "STAND")
                ],
                "body": {"posture": "standing" if skill == "CROUCH" else "crouching"},
                "previous_action": None
                if skill == "CROUCH"
                else {"skill": "CROUCH", "completed": True},
                "fixture": True,
            },
            instruction="相手の最新の依頼をMyuMIQが行う候補を選ぶ。前回の依頼は完了済み。"
            "現在姿勢を考慮し、新しい依頼に従う。これは接続確認用の仮想状態。",
        )
        if settings.selection:
            result = LocalSelection(
                settings.selection.llm,
                allow_state_only=settings.selection.allow_state_only,
                use_image=settings.selection.use_image,
                understand_requests=settings.selection.understand_requests,
            ).choose(request)
        else:
            scorer = make_scorer(settings.backend)
            try:
                result = scorer.score(request)
            finally:
                if hasattr(scorer, "close"):
                    scorer.close()
        chosen = result.select(request, time.perf_counter(), settings.max_age_s)
        if chosen is None or chosen.id != skill.lower():
            raise ValueError("decision did not follow the new request")
        selections.append(chosen.id)
        metadata.append(result.metadata)
    return {
        "selected": selections,
        "inference": metadata,
        "checks": ["candidate_identity", "changed_request", "snapshot_deadline"],
        "executed": False,
    }


def check_models(config):
    """A failed role does not skip the others or silently replace its backend."""
    decision = config.decision
    roles = [
        ("speech", config.llm, lambda: _speech(config.llm, config.dialogue.use_image)),
        (
            "thought",
            config.purpose.planner if config.purpose else None,
            lambda: _thought(config.purpose.planner, config.purpose.planner_use_image),
        ),
        (
            "decision",
            (
                (decision.selection.llm if decision.selection else decision.backend)
                if decision
                else None
            ),
            lambda: _decision(decision),
        ),
    ]
    reports = []
    for role, profile, probe in roles:
        started = time.perf_counter()
        report = {"role": role, "ok": False}
        if profile is None:
            report.update(error="not_configured", hint="Configure this independent model role.")
        elif variable := _credentials(profile):
            report.update(
                error="missing_credentials",
                environment_variable=variable,
                hint="Set the environment variable before starting this process.",
            )
        else:
            report.update(
                adapter=getattr(profile, "adapter", getattr(profile, "backend", None)),
                model=getattr(profile, "model", getattr(profile, "jev_model", None)),
            )
            try:
                report.update(probe(), ok=True)
            except Exception as exc:
                # Extension/provider exceptions may echo credentials or response bodies.
                report.update(
                    error=type(exc).__name__,
                    hint="Check model ID, endpoint/provider, JSON support, token limit and deadline.",
                )
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        reports.append(report)
    return {
        "ok": all(report["ok"] for report in reports),
        "roles": reports,
        "scope": "synthetic_model_contracts_only",
        "vrchat_tested": False,
        "audio_tested": False,
        "memory_read_or_written": False,
    }
