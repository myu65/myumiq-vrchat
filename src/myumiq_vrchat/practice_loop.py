"""Finite local-training / restored live-trial batches using existing motor ownership."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from pydantic import Field

from .articulated_tasks import ArticulatedTasks
from .body import Frozen, Number
from .body_console import atomic_json
from .cli import outside_repo
from .condition_validation import ConditionCase, starting_pose
from .practice_feedback import append_training, collect, digest, live_admissible


class PracticeConfig(Frozen):
    tasks: Path
    cases: Path
    prior: Path
    reference_corpus: Path
    # Explicit operator-controlled host command, not executable text from a model.
    adapter: tuple[str, ...] = Field(min_length=1, max_length=20)
    probe_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    rounds: int = Field(default=2, ge=1, le=8)
    updates: int = Field(default=100, ge=1, le=1000)
    learning_rate: Number = Field(default=1e-5, gt=0, le=0.001)
    reference_fraction: Number = Field(default=0.5, gt=0, lt=1)
    reference_anchor_weight: Number = Field(default=1.0, gt=0, le=1)
    training_timeout_s: Number = Field(default=600, ge=30, le=3600)
    live_timeout_s: Number = Field(default=600, ge=30, le=2100)
    cleanup_timeout_s: Number = Field(default=90, ge=10, le=180)
    total_timeout_s: Number = Field(default=3600, ge=120, le=14400)


def command(argv, timeout, log):
    with Path(log).open("w", encoding="utf-8") as output:
        subprocess.run(argv, timeout=timeout, check=True, stdout=output, stderr=subprocess.STDOUT)


class PracticeLoop:
    def __init__(self, config, out, runner=command):
        self.config, self.out, self.runner = config, outside_repo(out), runner
        self.out.mkdir(parents=True, exist_ok=False)
        self.deadline = time.monotonic() + config.total_timeout_s
        self.rounds = []

    def remaining(self, limit):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("practice batch budget exhausted")
        return min(limit, remaining)

    def event(self, stage, **details):
        row = dict(time=time.time(), stage=stage, **details)
        atomic_json(self.out / "status.json", row)
        with (self.out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def trial(self, folder, tasks, probes):
        folder.mkdir()
        task_bytes = tasks.read_bytes()
        tasks = folder / "tasks.json"
        tasks.write_bytes(task_bytes)
        request = folder / "request.json"
        report_path, cleanup_path = folder / "report.json", folder / "cleanup.json"
        atomic_json(
            request,
            dict(
                tasks=str(tasks),
                tasks_sha256=digest(tasks),
                actor_sha256=json.loads(tasks.read_text("utf-8"))["actor_sha256"],
                cases=[c.model_dump(mode="json") for c in probes],
                scope="fixed_actor_owned_home_device_practice",
            ),
        )
        self.event("live", folder=str(folder))
        try:
            self.runner(
                [
                    *self.config.adapter,
                    "--operation",
                    "run",
                    "--request",
                    str(request),
                    "--report",
                    str(report_path),
                ],
                self.remaining(self.config.live_timeout_s),
                folder / "adapter.log",
            )
        finally:
            # Cleanup gets its own budget even after the overall or stage deadline.
            self.runner(
                [
                    *self.config.adapter,
                    "--operation",
                    "cleanup",
                    "--request",
                    str(request),
                    "--report",
                    str(cleanup_path),
                ],
                self.config.cleanup_timeout_s,
                folder / "cleanup.log",
            )
            cleanup = json.loads(cleanup_path.read_text("utf-8"))
            if cleanup.get("restored") is not True or cleanup.get("errors"):
                raise RuntimeError("live adapter did not verify restoration; batch stopped")
        settings = ArticulatedTasks.model_validate_json(tasks.read_text("utf-8"))
        summary, starts = collect(
            json.loads(report_path.read_text("utf-8")),
            probes,
            settings.actor_sha256,
            settings.reference_floor,
        )
        atomic_json(folder / "measurements.json", summary)
        atomic_json(folder / "training-starts.json", [c.model_dump(mode="json") for c in starts])
        return summary, starts

    def run(self):
        config = self.config
        settings = ArticulatedTasks.model_validate_json(
            outside_repo(config.tasks).read_text("utf-8")
        )
        raw_cases = json.loads(outside_repo(config.cases).read_text("utf-8"))
        cases = [ConditionCase.model_validate_json(json.dumps(c)) for c in raw_cases]
        if (
            not 2 <= len(cases) <= 64
            or len({c.id for c in cases}) != len(cases)
            or {c.split for c in cases} != {"train", "heldout"}
        ):
            raise ValueError("unique training and held-out cases required")
        for case in cases:
            starting_pose(settings, case)
        by_id = {c.id: c for c in cases}
        probes = [by_id[key] for key in config.probe_ids]
        if len(set(config.probe_ids)) != len(probes):
            raise ValueError("duplicate probe identity")
        if {c.split for c in probes} != {"train", "heldout"}:
            raise ValueError("live probes need both training and held-out conditions")
        if any(
            c.world.objects
            or any(x.frame == "target" for x in c.goal.conditions)
            or not c.goal.conditions
            or not 1 <= c.goal.duration_s <= 20
            for c in probes
        ):
            raise ValueError(
                "first practice adapter contract requires fixed-frame condition probes"
            )
        prior = outside_repo(config.prior)
        if digest(prior / "candidate-actor.pt") != settings.actor_sha256:
            raise ValueError("prior checkpoint differs from tasks")
        baseline = outside_repo(settings.actor)
        if digest(baseline) != settings.actor_sha256:
            raise ValueError("configured actor content changed")
        corpus = outside_repo(config.reference_corpus)
        champion = self.out / "selected-tasks.json"
        atomic_json(champion, settings.model_dump(mode="json"))
        atomic_json(self.out / "config.json", config.model_dump(mode="json"))
        atomic_json(self.out / "frozen-cases.json", raw_cases)
        atomic_json(
            self.out / "manifest.json",
            dict(
                baseline_actor=str(baseline),
                baseline_actor_sha256=digest(baseline),
                tasks_sha256=digest(config.tasks),
                cases_sha256=digest(config.cases),
                reference_report_sha256=digest(corpus / "result.json"),
                avatar_verified=False,
                normal_settings_modified=False,
            ),
        )
        baseline_live, starts = self.trial(self.out / "baseline-live", champion, probes)
        champion_live = baseline_live
        cases = append_training(cases, starts)
        for number in range(1, config.rounds + 1):
            folder = self.out / f"round-{number:02d}"
            folder.mkdir()
            case_path, candidate_dir = folder / "cases.json", folder / "candidate"
            atomic_json(case_path, [c.model_dump(mode="json") for c in cases])
            self.event("training", round=number, cases=len(cases), prior=str(prior))
            self.runner(
                [
                    sys.executable,
                    "-m",
                    "myumiq_vrchat.condition_training",
                    "--tasks",
                    str(champion),
                    "--cases",
                    str(case_path),
                    "--prior",
                    str(prior),
                    "--baseline",
                    str(baseline),
                    "--reference-corpus",
                    str(corpus),
                    "--out",
                    str(candidate_dir),
                    "--updates",
                    str(config.updates),
                    "--learning-rate",
                    str(config.learning_rate),
                    "--reference-fraction",
                    str(config.reference_fraction),
                    "--reference-anchor-weight",
                    str(config.reference_anchor_weight),
                ],
                self.remaining(config.training_timeout_s),
                folder / "training.log",
            )
            trained = json.loads((candidate_dir / "result.json").read_text("utf-8"))
            if (
                trained["initial_actor_sha256"] != settings.actor_sha256
                or trained["baseline_actor_sha256"] != digest(baseline)
                or trained["cases_sha256"] != digest(case_path)
            ):
                raise ValueError("training result does not match batch inputs")
            eligible = trained["eligible_for_controlled_trial"] is True
            trial_tasks = folder / "trial-tasks.json"
            candidate_settings = settings
            if eligible:
                candidate_settings = settings.model_copy(
                    update={
                        "actor": candidate_dir / "candidate-actor.pt",
                        "actor_sha256": digest(candidate_dir / "candidate-actor.pt"),
                    }
                )
            atomic_json(trial_tasks, candidate_settings.model_dump(mode="json"))
            self.event("offline_evaluated", round=number, eligible=eligible)
            live, starts = self.trial(folder / "live", trial_tasks, probes)
            accepted = (
                eligible
                and live_admissible(champion_live, live)
                and live_admissible(baseline_live, live)
            )
            if accepted:
                settings, prior, champion_live = candidate_settings, candidate_dir, live
                atomic_json(champion, settings.model_dump(mode="json"))
            cases = append_training(cases, starts)
            result = dict(
                round=number,
                updates=trained["updates"],
                offline_eligible=eligible,
                live_actor_sha256=candidate_settings.actor_sha256,
                adopted_in_batch=accepted,
                selected_actor_sha256=settings.actor_sha256,
                next_case_count=len(cases),
                rejection=None
                if accepted
                else ("offline_regression" if not eligible else "live_evidence_or_regression"),
            )
            self.rounds.append(result)
            atomic_json(folder / "result.json", result)
            self.event("round_complete", **result)
        return dict(
            completed=True,
            rounds=self.rounds,
            selected_tasks=str(champion),
            selected_prior=str(prior),
            selected_actor_sha256=settings.actor_sha256,
            normal_settings_modified=False,
            avatar_verified=False,
            scope="local_actor_updates_and_observed_start_feedback_not_visual_reward",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = PracticeConfig.model_validate_json(outside_repo(args.config).read_text("utf-8"))
    loop = PracticeLoop(config, args.out)
    try:
        result = loop.run()
    except Exception as exc:
        atomic_json(
            loop.out / "result.json", dict(completed=False, rounds=loop.rounds, error=str(exc))
        )
        loop.event("failed", error=str(exc))
        raise
    atomic_json(loop.out / "result.json", result)
    loop.event("complete", selected_actor_sha256=result["selected_actor_sha256"])


if __name__ == "__main__":
    main()
