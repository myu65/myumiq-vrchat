"""Evidence-bearing application memory, shared by dialogue, plans and learning."""

import json
import re
import sqlite3
import time
import uuid
from copy import deepcopy
from pathlib import Path


class AgentMemory:
    """Owned by the slow executive; motor frames only see immutable summaries."""

    def __init__(self, path: Path):
        if path.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError("agent memory must be outside repository")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episode
              (id TEXT PRIMARY KEY, time REAL, kind TEXT, partner TEXT, data TEXT);
            CREATE TABLE IF NOT EXISTS knowledge
              (key TEXT PRIMARY KEY, partner TEXT, text TEXT, evidence TEXT, scope TEXT);
            CREATE TABLE IF NOT EXISTS social
              (id TEXT PRIMARY KEY, turns INTEGER, sessions INTEGER, last_session TEXT, last_seen REAL);
            CREATE TABLE IF NOT EXISTS skill_condition
              (skill TEXT, condition TEXT, scope TEXT, success INTEGER, failure INTEGER,
               unknown INTEGER, PRIMARY KEY(skill, condition, scope));
        """)
        self.session = uuid.uuid4().hex
        row = self.db.execute("SELECT value FROM state WHERE key='commitment'").fetchone()
        self.commitment = json.loads(row[0]) if row else None
        if self.commitment and self.commitment["status"] in ("active", "suspended"):
            self.commitment["status"] = "needs_revalidation"
        self.working = dict(
            partner=None,
            focus=None,
            topic="",
            turns=[],
            plan=None,
            attention="environment",
            session=self.session,
        )
        self.revision = 0
        self.save()

    def save(self):
        self.revision += 1
        self.db.execute(
            "INSERT OR REPLACE INTO state VALUES ('commitment', ?)",
            (json.dumps(self.commitment, ensure_ascii=False),),
        )
        self.db.execute(
            "INSERT OR REPLACE INTO state VALUES ('working', ?)",
            (json.dumps(self.working, ensure_ascii=False),),
        )
        self.db.commit()

    def episode(self, kind, data, partner=None):
        identity = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO episode VALUES (?,?,?,?,?)",
            (identity, time.time(), kind, partner, json.dumps(data, ensure_ascii=False)),
        )
        # Compact semantic aggregates survive this bounded detailed journal.
        self.db.execute(
            "DELETE FROM episode WHERE id IN (SELECT id FROM episode ORDER BY time DESC LIMIT -1 OFFSET 2000) AND id NOT IN (SELECT evidence FROM knowledge)"
        )
        return identity

    def _update_commitment(self, purpose):
        prior = self.commitment
        replace = purpose.continuity == "replace"
        if prior is None or prior["status"] == "completed" or replace:
            if prior and replace:
                self.episode("commitment_replaced", prior)
            self.commitment = dict(
                id=uuid.uuid4().hex,
                description=purpose.description,
                criterion=purpose.criterion,
                target=purpose.focus,
                priority=0.5,
                progress=0.0,
                status="active",
                evidence=[],
                created=time.time(),
                updated=time.time(),
            )
        else:
            prior["status"] = "active"
            prior["updated"] = time.time()
        if purpose.focus:
            self.working["focus"] = purpose.focus

    def propose_goal(self, purpose):
        """A planner updates purpose, never the executing finite action/plan."""
        self._update_commitment(purpose)
        self.working["planner_proposal"] = purpose.model_dump(mode="json")
        self.episode("planner_proposal", self.working["planner_proposal"])
        self.save()

    def start_plan(self, purpose, plan_id, *, manage_commitment=True):
        if manage_commitment:
            self._update_commitment(purpose)
        self.working["plan"] = dict(
            id=plan_id,
            description=purpose.description,
            step=0,
            status="active",
            manages_commitment=manage_commitment,
        )
        self.episode(
            "plan_started",
            dict(
                plan_id=plan_id,
                purpose=purpose.model_dump(mode="json"),
                manages_commitment=manage_commitment,
                commitment_id=self.commitment["id"] if self.commitment else None,
            ),
        )
        self.save()

    def plan_result(self, status, evidence):
        plan = self.working.get("plan")
        if plan:
            plan["status"] = status
        if self.commitment and plan and plan.get("manages_commitment", True):
            self.commitment["updated"] = time.time()
            if status.startswith("blocked"):
                self.commitment["priority"] = max(0.1, self.commitment["priority"] - 0.1)
            if status == "interrupted":
                self.commitment["status"] = "suspended"
        self.episode("plan_result", dict(plan=plan, status=status, evidence=evidence))
        self.save()

    def progress(self, criterion, evidence, target=None):
        c = self.commitment
        if c is None or c["status"] == "completed" or c["criterion"] != criterion:
            return
        if c.get("target") and c["target"] != target:
            return
        c["evidence"] = (c["evidence"] + [evidence])[-8:]
        c["progress"] = 1.0
        c["status"] = "completed"
        c["updated"] = time.time()
        self.episode("commitment_completed", c)

    def hear(self, text, *, partner=None, source="asr", focus=None, input_context=None):
        text = text.strip()[:2000]
        if not text:
            raise ValueError("empty utterance")
        # partner must already be explicitly bound by the operator/identity adapter.
        if partner is not None and (not partner or len(partner) > 80):
            raise ValueError("invalid verified speaker identifier")
        previous = self.working["partner"]
        if previous != partner:
            self.working["turns"], self.working["topic"] = [], ""
        self.working.update(partner=partner, attention="conversation")
        if focus:
            self.working["focus"] = focus
        data = dict(
            text=text, source=source, identity="explicit_association" if partner else "anonymous"
        )
        if input_context is not None:
            data["input"] = dict(input_context)
        episode_id = self.episode("heard", data, partner)
        turn = dict(role="user", text=text, episode_id=episode_id, partner=partner)
        if input_context is not None:
            turn["input"] = dict(input_context)
        self.working["turns"] = (self.working["turns"] + [turn])[-10:]
        if partner:
            self.db.execute(
                """INSERT INTO social VALUES (?,1,1,?,?) ON CONFLICT(id) DO UPDATE SET
                turns=turns+1, sessions=sessions+(last_session!=excluded.last_session),
                last_session=excluded.last_session, last_seen=excluded.last_seen""",
                (partner, self.session, time.time()),
            )
            # Preserve short explicit self-reports without inventing a summary.
            # Questions and anonymous speech do not create personal knowledge.
            if "?" not in text and "？" not in text:
                clauses = [s.strip() for s in re.split("[。！!\n]", text)]
                facts = [
                    s
                    for s in clauses
                    if 2 <= len(s) <= 160
                    and any(
                        marker in s
                        for marker in (
                            "私は",
                            "好き",
                            "嫌い",
                            "苦手",
                            "趣味",
                            "暮らして",
                            "と呼んで",
                        )
                    )
                ][:2]
                for quote in facts:
                    self.db.execute(
                        "INSERT OR REPLACE INTO knowledge VALUES (?,?,?,?,?)",
                        (
                            f"reported:{partner}:{quote}",
                            partner,
                            quote,
                            episode_id,
                            "speaker_reported_not_verified",
                        ),
                    )
        self.save()
        return turn

    def reply(self, reply, topic, facts, heard, submitted):
        partner = heard["partner"]
        episode_id = self.episode(
            "reply",
            dict(
                text=reply,
                submitted=submitted,
                topic=topic,
                delivery="unverified",
                in_reply_to=heard["episode_id"],
            ),
            partner,
        )
        self.working["topic"] = topic[:120]
        self.working["turns"] = (
            self.working["turns"]
            + [dict(role="assistant", text=reply, episode_id=episode_id, submitted=submitted)]
        )[-10:]
        # Only exact evidence from this utterance is retained as a reported fact.
        # Anonymous speech stays working/episodic, never an attributed social fact.
        if partner:
            for quote in facts[:2]:
                if 2 <= len(quote) <= 160 and quote in heard["text"]:
                    if self.db.execute(
                        "SELECT 1 FROM knowledge WHERE evidence=? AND instr(text,?)>0",
                        (heard["episode_id"], quote),
                    ).fetchone():
                        continue
                    self.db.execute(
                        "INSERT OR REPLACE INTO knowledge VALUES (?,?,?,?,?)",
                        (
                            f"reported:{partner}:{quote}",
                            partner,
                            quote,
                            heard["episode_id"],
                            "speaker_reported_not_verified",
                        ),
                    )
        if submitted and sum(t["role"] == "user" for t in self.working["turns"]) >= 2:
            # Submission, not partner receipt: conversational continuity evidence only.
            if self.commitment and self.commitment["criterion"] == "interaction":
                self.commitment["progress"] = max(self.commitment["progress"], 0.5)
        self.save()

    def skill_outcome(self, skill, condition, outcome, scope, evidence, target=None):
        episode_id = self.episode(
            "skill_outcome",
            dict(skill=skill, condition=condition, success=outcome, scope=scope, evidence=evidence),
        )
        self.db.execute(
            """INSERT INTO skill_condition VALUES (?,?,?,?,?,?)
            ON CONFLICT(skill,condition,scope) DO UPDATE SET success=success+excluded.success,
            failure=failure+excluded.failure, unknown=unknown+excluded.unknown""",
            (
                skill,
                condition,
                scope,
                int(outcome is True),
                int(outcome is False),
                int(outcome is None),
            ),
        )
        if outcome is False:
            self.db.execute(
                "INSERT OR REPLACE INTO knowledge VALUES (?,?,?,?,?)",
                (
                    f"failure:{skill}:{condition}:{scope}",
                    None,
                    f"{skill}: failure observed under {condition}",
                    episode_id,
                    scope,
                ),
            )
        if outcome is True:
            if skill == "LOOK_AT" and scope in (
                "fresh_visual_alignment",
                "fresh_visual_body_heading",
            ):
                self.progress("observe_target", episode_id, target)
            elif skill in ("SIT", "LIE"):
                self.progress("rest", episode_id)
        if self.working["plan"]:
            self.working["plan"]["step"] += 1
        self.save()

    def learned(self, task):
        episode_id = self.episode("capability_learned", task)
        self.db.execute(
            "INSERT OR REPLACE INTO knowledge VALUES (?,?,?,?,?)",
            (
                "learned:" + task["capability"],
                None,
                task["capability"] + " policy registered",
                episode_id,
                task["result"].get("evaluation_scope", "unknown"),
            ),
        )
        self.progress("learn_capability", episode_id, task["capability"])
        self.save()

    def repeated_failure(self, skill, condition):
        failures = 0
        rows = self.db.execute(
            "SELECT data FROM episode WHERE kind='skill_outcome' ORDER BY rowid DESC"
        )
        for (data,) in rows:
            result = json.loads(data)
            if result["skill"] != skill or result["condition"] != condition:
                continue
            if result["success"] is True:
                return False
            if result["success"] is False:
                failures += 1
                if failures >= 2:
                    return True
        # The bounded episode journal may have aged out older evidence.
        row = self.db.execute(
            "SELECT SUM(failure),SUM(success) FROM skill_condition WHERE skill=? AND condition=?",
            (skill, condition),
        ).fetchone()
        return bool(row and (row[0] or 0) >= 2 and not (row[1] or 0))

    def retrieve(self, capabilities, query=""):
        partner = self.working["partner"]
        rows = self.db.execute(
            "SELECT id,kind,partner,data FROM episode WHERE partner IS ? OR partner IS NULL ORDER BY time DESC LIMIT 8",
            (partner,),
        ).fetchall()
        knowledge = self.db.execute(
            "SELECT text,evidence,scope FROM knowledge WHERE partner IS ? OR partner IS NULL ORDER BY rowid DESC LIMIT 10",
            (partner,),
        ).fetchall()
        social = (
            self.db.execute(
                "SELECT turns,sessions,last_seen FROM social WHERE id=?", (partner,)
            ).fetchone()
            if partner
            else None
        )
        conditions = self.db.execute(
            "SELECT skill,condition,scope,success,failure,unknown FROM skill_condition ORDER BY failure DESC, rowid DESC LIMIT 8"
        ).fetchall()
        # Literal query relevance works for Japanese without inventing an embedding model.
        if query:
            knowledge.sort(
                key=lambda r: -sum(query[i : i + 2] in r[0] for i in range(len(query) - 1))
            )
        episodes = []
        for row in rows[:4]:
            data = json.loads(row[3])
            compact = {
                k: v
                for k, v in data.items()
                if k
                in (
                    "status",
                    "skill",
                    "success",
                    "scope",
                    "text",
                    "topic",
                    "condition",
                    "submitted",
                    "capability",
                    "direction",
                    "image_change",
                    "novelty",
                    "outcome",
                )
            }
            if "text" in compact:
                compact["text"] = compact["text"][:160]
            episodes.append(dict(id=row[0], kind=row[1], partner=row[2], data=compact))
        return deepcopy(
            dict(
                revision=self.revision,
                working=self.working,
                commitment=self.commitment,
                episodic=episodes,
                semantic=[dict(text=r[0], evidence=r[1], scope=r[2]) for r in knowledge[:6]],
                social=dict(
                    identity=partner,
                    known=bool(social and social[0] > 1),
                    turns=social[0] if social else 0,
                    prior_sessions=max(0, social[1] - 1) if social else 0,
                    encounter=(
                        "anonymous"
                        if not social
                        else "returning"
                        if social[1] > 1
                        else "continuing"
                        if social[0] > 1
                        else "first"
                    ),
                    affinity=None,
                ),
                capabilities=capabilities,
                failure_conditions=[
                    dict(zip(("skill", "condition", "scope", "success", "failure", "unknown"), r))
                    for r in conditions
                ],
            )
        )

    def context(self, query=""):
        result = self.retrieve([], query)
        result.pop("capabilities")
        if result["commitment"]:
            result["commitment"].pop("evidence", None)
        return result

    def close(self):
        self.save()
        self.db.close()
