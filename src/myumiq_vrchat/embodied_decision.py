"""Independent decision-model body control over world and conversation history."""

from dataclasses import asdict
from hashlib import sha256

from .capabilities import posture_capability
from .decision import Candidate, DecisionInput, make_scorer
from .exploration import DIRECTED_MOVEMENT, NAVIGATION_SKILLS
from .gaze import gaze_targets
from .purposes import Purpose, SkillRequest


def candidate_id(intent):
    """Stable within the current target namespace, never based on list position."""
    identity = "body_" + intent.skill
    if intent.hand:
        identity += "_" + intent.hand
    if intent.target:
        identity += "_" + sha256(intent.target.encode("utf-8")).hexdigest()[:24]
    return identity


def target_geometry(world, name):
    if name is None:
        return None
    target = next((o for o in world.objects if o.name == name), None)
    return (
        target.model_dump(mode="json", include={"position", "image_position", "source", "kind"})
        if target
        else None
    )


def decision_outcome(item):
    summary = {k: item[k] for k in ("goal_id", "goal", "status", "action") if k in item}
    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        summary["evidence"] = {
            k: evidence[k]
            for k in (
                "capability",
                "success",
                "scope",
                "movement_m",
                "observations",
                "maximum_position_error_m",
                "maximum_rotation_error_rad",
                "actor_error",
                "avatar_verified",
                "execution",
            )
            if k in evidence
        }
    elif evidence is not None:
        summary["evidence"] = str(evidence)[:240]
    return summary


def candidates(registry, world):
    result = [
        Candidate(id="continue", description="今の動作・姿勢を続ける。新しい身体動作を始めない。")
    ]
    requests = [
        (
            label,
            SkillRequest(capability=skill, hand=hand, duration_s=registry.get(skill).duration_s),
        )
        for label, skill, hand in (
            ("現在の姿勢で止まって待つ", "WAIT", None),
            ("床に座る", "SIT", None),
            ("立ち上がる", "STAND", None),
            ("横になる", "LIE", None),
            ("しゃがむ", "CROUCH", None),
            ("標準の安静姿勢へ戻す", "RETURN_TO_REST", None),
            ("左手を振る", "WAVE", "left"),
            ("右手を振る", "WAVE", "right"),
            ("学習したその場歩行を行う", "WALK_IN_PLACE", None),
            ("Homeを少し移動して新しい景色を探索する", "EXPLORE_HOME", None),
        )
    ]
    requests += [
        (
            registry.get(name).description,
            SkillRequest(capability=name, duration_s=registry.get(name).duration_s),
        )
        for name in DIRECTED_MOVEMENT
    ]
    for name, item in registry.items.items():
        if name.startswith(("MOTION_", "POSTURE_")):
            requests.append(
                (
                    item.description or name,
                    SkillRequest(capability=name, duration_s=item.duration_s),
                )
            )
    for obj in gaze_targets(world):
        requests.append((obj.name + "を見る", SkillRequest(capability="LOOK_AT", target=obj.name)))
    for obj in world.objects:
        if obj.source in ("manual", "fixture"):
            for hand, label in (("left", "左手"), ("right", "右手")):
                requests.append(
                    (
                        obj.name + "へ" + label + "を伸ばす",
                        SkillRequest(capability="REACH", hand=hand, target=obj.name),
                    )
                )
    for label, step in requests:
        purpose = Purpose(
            description=label[:80],
            reason="世界と会話から次の身体動作を判断",
            success_description="動作の結果を観測する",
            steps=(step,),
        )
        resolved = registry.resolve(purpose, world)
        if resolved.actions and not resolved.blockers and not resolved.missing:
            intent = resolved.actions[0][1]
            result.append(
                Candidate(
                    id=candidate_id(intent),
                    description=label,
                    intent=intent.model_dump(mode="json"),
                )
            )
    return tuple(result)


def score(settings, request):
    import time

    if not 0 <= time.perf_counter() - request.captured_at <= settings.max_age_s:
        raise ValueError("body decision snapshot expired before inference")
    scorer = make_scorer(settings.backend)
    try:
        result = scorer.score(request)
        chosen = result.select(request, time.perf_counter(), settings.max_age_s)
        report = result.model_dump(mode="json")
        report.update(
            selected_id=chosen.id,
            candidates=[c.model_dump(mode="json") for c in request.candidates],
        )
        return chosen, report
    finally:
        if hasattr(scorer, "close"):
            scorer.close()


def evaluate(settings, request, session=None):
    if settings.selection is None:
        return score(settings, request)
    import time

    from .decision_selection import LocalSelection

    if not 0 <= time.perf_counter() - request.captured_at <= settings.max_age_s:
        raise ValueError("body decision snapshot expired before inference")
    profile = settings.selection
    result = LocalSelection(
        profile.llm,
        allow_state_only=profile.allow_state_only,
        use_image=profile.use_image,
        understand_requests=profile.understand_requests,
        session=session,
    ).choose(request)
    chosen = result.select(request, time.perf_counter(), settings.max_age_s)
    report = result.model_dump(mode="json")
    report.update(
        selected_id=chosen.id if chosen else None,
        candidates=[c.model_dump(mode="json") for c in request.candidates],
    )
    return chosen, report


class EmbodiedDecision:
    """Uses the PurposeRunner's scheduler, memory and outcome recorder."""

    def __init__(self, runner):
        self.runner = runner
        self.pending = None
        self.next_request = 0.0
        self.context_epoch = 0
        self.applied_epoch = -1
        self.observation_key = None
        self.observation_epoch = 0
        self.stimulus_key = None
        self.stimulus_epoch = 0
        self.fulfilled_utterance = None
        self.fulfilled = {}
        self.attempt_utterance = None
        self.attempted = set()
        self.maintained_posture = None
        self.handled_request = None
        self.request_assessment = None
        self.stop_latched = False
        self.stopped_utterance = None
        self.retry_after = 0.0
        self.generation_session = None
        self.retired = []

    def _preempt(self):
        if self.pending:
            self.retired.append((self.pending[0], self.generation_session))
            if self.generation_session:
                self.generation_session.cancel()
                self.generation_session.close()
            self.pending = self.generation_session = None
            self.runner.emit("body_decision_preempted")

    def close(self):
        self._preempt()
        for _, session in self.retired:
            if session:
                session.cancel()
                session.close()

    def clear_body_request(self):
        self.maintained_posture = None
        self.handled_request = None
        self.request_assessment = None
        self.stop_latched = False
        self.stopped_utterance = None
        self.runner.shared.working.pop("maintained_posture", None)
        self.runner.shared.working.pop("body_request", None)

    def _request_conflict(self, intent, report, utterance_id):
        new_request = (
            report.get("basis") == "latest_utterance"
            and report.get("request_status") not in ("none", "unsupported", "clarify")
            and utterance_id is not None
            and utterance_id != self.handled_request
        )
        held = self.maintained_posture
        if self.runner.owner.choice[2].skill == "BODY_GOAL":
            held = {"skill": "BODY_GOAL"}
        # A whole-body turn also moves planted feet and changes orientation
        # conditions. It is not a compatible gaze overlay on a condition goal.
        compatible_skills = ("WAIT", held["skill"]) if held else ()
        if held and held["skill"] != "BODY_GOAL":
            compatible_skills += ("LOOK_AT",)
        compatible = not held or intent.skill in compatible_skills
        return not compatible and not new_request, new_request

    def retry_available(self, intent, utterance_id):
        """One fresh-input attempt; failure memory never grants endless retries."""
        turns = self.runner.shared.working["turns"]
        latest = next((t["episode_id"] for t in reversed(turns) if t["role"] == "user"), None)
        if utterance_id is None or utterance_id != latest:
            return False
        if latest != self.attempt_utterance:
            self.attempt_utterance, self.attempted = latest, set()
        records = [h.get("action") or {} for h in self.runner.history]
        if self.runner.running:
            records.append(self.runner.running)
        self.attempted.update(
            a["candidate_id"]
            for a in records
            if a.get("context_utterance_id") == latest and a.get("candidate_id")
        )
        return candidate_id(intent) not in self.attempted

    def conversation_changed(self):
        self._preempt()
        self.context_epoch += 1
        self.next_request = 0.0
        self.retry_after = 0.0

    def _observe_changes(self):
        runner = self.runner
        commitment = runner.shared.commitment or {}
        last = runner.history[-1] if runner.history else {}
        key = (
            tuple(sorted((o.name, o.source, o.kind) for o in runner.owner.world.objects)),
            runner.goal_id if runner.origin != "body_decision" else None,
            tuple(commitment.get(k) for k in ("id", "description", "status", "progress")),
            (last.get("goal_id"), last.get("status")),
        )
        stimulus_key = (
            key[1],
            tuple(commitment.get(k) for k in ("id", "description", "criterion", "target")),
        )
        if stimulus_key != self.stimulus_key:
            self.stimulus_key = stimulus_key
            self.stimulus_epoch += 1
        if key != self.observation_key:
            self.observation_key = key
            self.observation_epoch += 1
            self.next_request = 0.0

    def _completed_in_context(self, chosen, epoch):
        runner, owner = self.runner, self.runner.owner
        # Exploration is an ongoing activity made of finite safe leases. Its
        # previous lease succeeding must not mark the entire activity fulfilled.
        # Explicit directed moves and one-shot gestures keep deduplication.
        if chosen.intent.get("skill") == "EXPLORE_HOME":
            return False
        last = runner.history[-1] if runner.history else {}
        action = last.get("action") or {}
        evidence = last.get("evidence") or {}
        return (
            last.get("status") == "plan_completed"
            and isinstance(evidence, dict)
            and evidence.get("success") is True
            and action.get("candidate_id") == chosen.id
            and action.get("conversation_epoch") == epoch
            and action.get("decision_stimulus_epoch") == self.stimulus_epoch
            and action.get("intent_generation") == owner.choice[0]
            and action.get("target_geometry")
            == target_geometry(owner.world, chosen.intent.get("target"))
        )

    def _fulfilled_for_utterance(self, utterance_id):
        if utterance_id != self.fulfilled_utterance:
            self.fulfilled_utterance, self.fulfilled = utterance_id, {}
        if utterance_id is not None:
            for item in self.runner.history:
                action, evidence = item.get("action") or {}, item.get("evidence") or {}
                intent = action.get("intent") or {}
                # Target-facing control and exploration respond continuously to the world.
                if (
                    action.get("context_utterance_id") == utterance_id
                    and item.get("status") == "plan_completed"
                    and isinstance(evidence, dict)
                    and evidence.get("success") is True
                    and not intent.get("target")
                    and intent.get("skill") not in ("WAIT", "EXPLORE_HOME")
                ):
                    self.fulfilled[action.get("candidate_id")] = item
        return self.fulfilled

    def tick(self, now):
        try:
            self._tick(now)
        except Exception as exc:
            self._failed(now, exc)

    def _failed(self, now, exc):
        self.runner.owner.health["decision"] = {
            "state": "unavailable",
            "role": "body_owner",
            "error": str(exc)[:200],
            "fallback": "current_finite_action_then_pose_hold",
        }
        self.runner.emit("body_decision_failed", error=str(exc)[:200])
        delay = (
            min(2.0, self.runner.settings.retry_s)
            if isinstance(exc, ValueError)
            else self.runner.settings.retry_s
        )
        self.next_request = now + delay
        self.retry_after = self.next_request

    def assess_request(self, report, utterance_id, now):
        status = report.get("request_status")
        if status is None or utterance_id is None:
            return
        previous = self.request_assessment
        if (
            previous
            and previous["utterance_id"] == utterance_id
            and previous.get("execution")
            in ("accepted", "running", "completed", "failed", "unverified")
        ):
            # Current capability availability cannot rewrite an accepted request's
            # history. Only execution observations update its factual outcome.
            return
        self.request_assessment = {
            "utterance_id": utterance_id,
            "status": status,
            "reason": report.get("request_reason", ""),
            "execution": "not_accepted",
            "requested_capabilities": report.get("requested_capabilities", []),
            "assessed_at": now,
        }
        self.runner.shared.working["body_request"] = dict(self.request_assessment)
        self.runner.emit("body_request_assessed", **self.request_assessment)
        if status == "stop":
            self.stop_latched = True
            self.stopped_utterance = utterance_id
        if (
            status in ("unsupported", "clarify")
            and self.runner.owner.choice[2].skill in NAVIGATION_SKILLS
        ):
            from .autonomous_body import Intent

            self.runner.finish("unresolved_body_request")
            self.runner.owner._choose(now, Intent(skill="WAIT"), "unresolved_body_request")
            self.runner.epoch = self.runner.owner.generation

    def mark_request(self, utterance_id, **result):
        if self.request_assessment and self.request_assessment["utterance_id"] == utterance_id:
            if result.get("execution") == "unavailable" and self.request_assessment.get(
                "execution"
            ) in ("accepted", "running", "completed", "failed", "unverified"):
                return  # A later inference error cannot erase execution evidence.
            self.request_assessment.update(result)
            self.runner.shared.working["body_request"] = dict(self.request_assessment)

    def _tick(self, now):
        from .autonomous_body import Intent
        from .purpose_runtime import action_summary, background, body_summary

        runner, owner = self.runner, self.runner.owner
        self.retired = [(future, session) for future, session in self.retired if not future.done()]
        if runner.running:
            runner._observe_step(now)
        self._observe_changes()
        if self.pending:
            future, generation, epoch, stimulus_epoch, utterance_id = self.pending
            if not future.done():
                return
            self.pending = None
            if self.generation_session:
                self.generation_session.close()
                self.generation_session = None
            if (
                generation != owner.generation
                or epoch != self.context_epoch
                or stimulus_epoch != self.stimulus_epoch
            ):
                self.next_request = 0.0
                runner.emit("body_decision_discarded", reason="context_changed")
            else:
                try:
                    chosen, report = future.result()
                    if not 0 <= now - report["captured_at"] <= owner.config.decision.max_age_s:
                        raise ValueError("body decision expired before activation")
                    owner.health["decision"] = {
                        "state": "running",
                        "result": report,
                        "role": "body_owner",
                    }
                    runner.emit("body_decision", context_epoch=epoch, **report)
                    self.assess_request(report, utterance_id, now)
                    if chosen is not None and chosen.intent:
                        # An observation/outcome update is not a blanket veto.
                        # Recheck this proposal against current evidence below.
                        intent = Intent.model_validate(chosen.intent)
                        if self.stop_latched and (
                            report.get("request_status") not in ("action", "stop")
                            or report.get("request_status") == "action"
                            and utterance_id == self.stopped_utterance
                        ):
                            runner.emit("body_decision_suppressed", reason="explicit_stop")
                            self.next_request = now + owner.config.decision.refresh_s
                            return
                        if runner.running and report.get("request_status") == "none":
                            runner.emit(
                                "body_decision_suppressed",
                                reason="conversation_preserves_body_goal",
                            )
                            self.next_request = now + owner.config.decision.refresh_s
                            return
                        conflict, new_request = self._request_conflict(intent, report, utterance_id)
                        if conflict:
                            runner.emit(
                                "body_decision_suppressed",
                                reason="maintained_posture",
                                candidate_id=chosen.id,
                            )
                            self.next_request = now + owner.config.decision.refresh_s
                            return
                        same = (intent.skill, intent.target, intent.hand) == (
                            owner.choice[2].skill,
                            owner.choice[2].target,
                            owner.choice[2].hand,
                        )
                        if same and runner.running and self.applied_epoch == epoch:
                            self.mark_request(utterance_id, execution="running", skill=intent.skill)
                        fulfilled = self._fulfilled_for_utterance(utterance_id).get(chosen.id)
                        completed = fulfilled or self._completed_in_context(chosen, epoch)
                        if completed:
                            self.mark_request(
                                utterance_id, execution="completed", skill=intent.skill
                            )
                            runner.emit(
                                "body_decision_suppressed",
                                reason="completed_same_context",
                                candidate_id=chosen.id,
                                action_id=(fulfilled or runner.history[-1])["action"]["action_id"],
                            )
                        if not (
                            completed or same and runner.running and self.applied_epoch == epoch
                        ):
                            step = SkillRequest(
                                capability=intent.skill,
                                hand=intent.hand,
                                target=intent.target,
                                duration_s=intent.duration_s,
                            )
                            purpose = Purpose(
                                description=chosen.description[:80],
                                reason="判断モデルが世界状態と会話履歴から選択",
                                success_description="動作の結果を観測する",
                                steps=(step,),
                            )
                            resolution = runner.registry.resolve(purpose, owner.world)
                            if resolution.blockers or resolution.missing:
                                raise ValueError(
                                    "body decision target/capability no longer available"
                                )
                            if intent.skill == "LOOK_AT":
                                obj = next(
                                    o for o in owner.world.objects if o.name == intent.target
                                )
                                if obj.source == "vision":
                                    from .gaze import image_target

                                    image_target(owner.world, intent.target, now)
                            runner.finish("superseded_by_body_decision")
                            runner.accept(
                                purpose,
                                now,
                                origin="body_decision",
                                retry_utterance_id=utterance_id,
                            )
                            if runner.goal is not None and runner.actions:
                                owner.purpose_metadata = {
                                    "goal_id": runner.goal_id,
                                    "purpose": purpose.description,
                                    "capability": intent.skill,
                                    "step": 0,
                                }
                                owner._choose(now, intent, "body_decision:" + report["backend"])
                                runner.epoch = owner.generation
                                body = owner.snapshot
                                start = {
                                    part: getattr(body, part).pose.position
                                    for part in ("head", "left", "right", "left_foot", "right_foot")
                                    if body and getattr(body, part).valid
                                }
                                runner.running = {
                                    "intent": intent,
                                    "capability": intent.skill,
                                    "started": now,
                                    "start": start,
                                    "movement": 0.0,
                                    "observations": 0,
                                    "intent_generation": owner.choice[0],
                                    "candidate_id": chosen.id,
                                    "conversation_epoch": epoch,
                                    "context_utterance_id": utterance_id,
                                    "target_geometry": target_geometry(owner.world, intent.target),
                                    "deferred": False,
                                    "condition": runner.condition(intent),
                                }
                                self.applied_epoch = epoch
                                self.mark_request(
                                    utterance_id, execution="accepted", skill=intent.skill
                                )
                                if report.get("request_status") == "action":
                                    self.stop_latched = False
                                if new_request:
                                    self.handled_request = utterance_id
                                    self.maintained_posture = (
                                        {"skill": intent.skill, "utterance_id": utterance_id}
                                        if posture_capability(intent.skill)
                                        or intent.skill in DIRECTED_MOVEMENT
                                        else None
                                    )
                                    runner.shared.working["maintained_posture"] = (
                                        self.maintained_posture
                                    )
                                if utterance_id is not None:
                                    self.retry_available(intent, utterance_id)
                                    self.attempted.add(chosen.id)
                                # This activation is already known; only subsequent changes wake us.
                                self._observe_changes()
                                runner.running["decision_stimulus_epoch"] = self.stimulus_epoch
                                self.next_request = max(
                                    self.next_request, now + owner.config.decision.refresh_s
                                )
                except Exception as exc:
                    self.mark_request(utterance_id, execution="unavailable", reason=str(exc)[:160])
                    self._failed(now, exc)
        if now < max(self.next_request, self.retry_after):
            return
        if len(self.retired) >= 2:
            owner.health["decision"] = {
                "state": "waiting_adapter_cancellation",
                "role": "body_owner",
            }
            return
        context = runner.shared.context()
        from .visual_context import visual_context

        visual = visual_context(
            runner.services.vision,
            owner.world,
            now,
            use_image=True,
            max_age_s=min(3.0, owner.config.decision.max_age_s),
        )
        world, image = visual.world, visual.fresh_image(now)
        captured = visual.captured_at if image else now
        image_error = visual.reason
        turns = [
            {k: t[k] for k in ("role", "episode_id", "partner", "submitted") if k in t}
            | {"text": t["text"][:160]}
            for t in context["working"]["turns"][-4:]
        ]
        utterance_id = next(
            (t.get("episode_id") for t in reversed(turns) if t["role"] == "user"), None
        )
        for turn in turns:
            turn["latest_utterance"] = turn.get("episode_id") == utterance_id
        fulfilled = self._fulfilled_for_utterance(utterance_id)
        available = tuple(c for c in candidates(runner.registry, world) if c.id not in fulfilled)
        recovery, blocked, permitted = [], [], []
        for candidate in available:
            if candidate.intent:
                intent = Intent.model_validate(candidate.intent)
                if runner.shared.repeated_failure(intent.skill, runner.condition(intent)):
                    if not self.retry_available(intent, utterance_id):
                        blocked.append(candidate.id)
                        continue
                    recovery.append(candidate.id)
            permitted.append(candidate)
        available = tuple(permitted)
        last = runner.history[-1] if runner.history else {}
        latest_actions = [
            h
            for h in runner.history
            if utterance_id is not None
            and (h.get("action") or {}).get("context_utterance_id") == utterance_id
        ]
        body = body_summary(owner.snapshot)
        if owner.snapshot:
            body["tracking_poses"] = {
                p: owner.snapshot.signal_for(p).pose.model_dump(mode="json")
                for p in ("head", "left", "right")
                if owner.snapshot.signal_for(p).valid
            }
        request = DecisionInput(
            world=world,
            image_base64=image,
            image_mime="image/jpeg",
            captured_at=captured,
            candidates=available,
            state={
                "conversation": turns,
                "body": body,
                "latest_utterance_id": utterance_id,
                "latest_user_utterance": next(
                    (t["text"] for t in reversed(turns) if t["role"] == "user"), None
                ),
                "stop_latched": self.stop_latched,
                "capabilities": [
                    {k: c[k] for k in ("name", "description", "available", "status")}
                    for c in runner.registry.summary()
                ],
                "maintained_posture": self.maintained_posture,
                "handled_body_request_utterance_id": self.handled_request,
                "fulfilled_candidates_for_latest_utterance": list(fulfilled),
                "fulfilled_request_capabilities": sorted(
                    {h["action"]["intent"]["skill"] for h in fulfilled.values()}
                ),
                "failed_candidates_requiring_new_request": recovery,
                "blocked_repeated_failures": blocked,
                "actions_for_latest_utterance": [decision_outcome(h) for h in latest_actions[-8:]],
                "last_outcome_matches_latest_utterance": bool(
                    utterance_id
                    and (last.get("action") or {}).get("context_utterance_id") == utterance_id
                ),
                "conversation_epoch": self.context_epoch,
                "last_action_conversation_epoch": self.applied_epoch,
                "observation_epoch": self.observation_epoch,
                "current_action": action_summary(runner.running, runner.goal_id, runner.index),
                "attention": context["working"]["attention"],
                "focus": context["working"]["focus"],
                "planner_proposal": context["working"].get("planner_proposal"),
                "drives": asdict(owner.drives),
                "current_intent": owner.choice[2].model_dump(mode="json"),
                "current_intent_active": owner.action_timing(now)["phase"]
                in ("preparing", "running"),
                "current_execution": owner.action_timing(now),
                "commitment": {
                    k: context["commitment"][k]
                    for k in ("id", "description", "criterion", "target", "progress", "status")
                    if k in context["commitment"]
                }
                if context["commitment"]
                else None,
                "knowledge": [k["text"][:120] for k in context.get("semantic", [])[:3]],
                "last_outcome": decision_outcome(last),
            },
            instruction="世界状態、身体状態、欲求、会話履歴を合わせて、今この動作を行う妥当性を判断する。"
            "会話の返答は別モデルが行う。相手自身の行動報告や質問を動作依頼と取り違えない。"
            "assistantの了承・返答は身体動作の実行証拠ではない。最新の発話への実行記録は"
            "actions_for_latest_utteranceにある。同じ文でも発話IDが異なれば新しい依頼。"
            "以前の発話に対する成功・失敗を、最新の依頼の実行済み判定に使わない。"
            "failed_candidates_requiring_new_requestは以前に失敗した動作で、最新の相手が改めてその動作を依頼した場合だけ一度再試行できる。自発行動では選ばない。"
            "latest_utterance=falseの過去の発話は新しい命令ではない。最新の「立って」の後に過去の「しゃがんで」を再実行しない。"
            "実行済みの依頼を繰り返さない。新しい依頼や観測による理由がなければ現在の動作を続ける。姿勢を自動的に戻さない。"
            "会話がなくても周囲と目的から行動できる。未観測の成功・接触・移動を仮定しない。",
        )
        runner.emit(
            "body_decision_request",
            context_epoch=self.context_epoch,
            observation_epoch=self.observation_epoch,
            state=request.state,
            world=world.model_dump(mode="json"),
            image_available=image is not None,
            image_error=image_error,
        )
        if owner.config.decision.selection is not None:
            from .generation import GenerationSession

            self.generation_session = GenerationSession(owner.config.decision.selection.llm)
        session = self.generation_session
        self.pending = (
            background(
                lambda: (
                    evaluate(owner.config.decision, request, session)
                    if session is not None
                    else evaluate(owner.config.decision, request)
                )
            ),
            owner.generation,
            self.context_epoch,
            self.stimulus_epoch,
            utterance_id,
        )
        self.next_request = now + owner.config.decision.refresh_s
        owner.health["decision"] = {"state": "thinking", "role": "body_owner"}
