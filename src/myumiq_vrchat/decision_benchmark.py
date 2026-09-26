"""Reproducible independent-score evaluation from user-reviewed JSON cases."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from .decision import BackendConfig, DecisionInput, make_scorer


def benchmark(scorer, cases, repeats=5):
    output = {
        "quality": [],
        "latency": [],
        "method": {
            "warmups": 2,
            "repeats": repeats,
            "image_io_included": True,
            "quality": "top-1 membership in predeclared acceptable IDs",
            "order": "original, reverse, fixed-seed shuffle; score differences by stable ID",
        },
    }
    for case in cases:
        request = DecisionInput.model_validate_json(json.dumps(case["request"]))
        if case.get("image"):
            request = DecisionInput.with_image(
                Path(case["image"]), **request.model_dump(exclude={"image_base64"})
            )
        results = []
        for order in (
            list(request.candidates),
            list(reversed(request.candidates)),
            random.Random(23).sample(list(request.candidates), len(request.candidates)),
        ):
            changed = request.model_copy(update={"candidates": tuple(order)})
            result = scorer.score(changed)
            results.append(result)
        tops = [min(r.scores, key=lambda s: (-s.score, s.id)).id for r in results]
        baseline = {s.id: s.score for s in results[0].scores}
        output["quality"].append(
            {
                "case": case["id"],
                "acceptable": case["acceptable"],
                "top_ids": tops,
                "correct": tops[0] in case["acceptable"],
                "order_stable": len(set(tops)) == 1,
                "max_score_delta": max(
                    abs(s.score - baseline[s.id]) for r in results for s in r.scores
                ),
                "result": results[0].model_dump(mode="json"),
            }
        )
    case = cases[0]
    for n in (2, 4, 8, 16):
        samples = []
        # File read + base64 + validation + CPU preprocessing + GPU + scalar transfer/API.
        for index in range(repeats + 2):
            started = time.perf_counter()
            values = dict(case["request"], candidates=case["request"]["candidates"][:n])
            request = DecisionInput.model_validate_json(json.dumps(values))
            if case.get("image"):
                request = DecisionInput.with_image(
                    Path(case["image"]), **request.model_dump(exclude={"image_base64"})
                )
            if len(request.candidates) != n:
                raise ValueError("latency case must provide 16 distinct candidates")
            result = scorer.score(request)
            elapsed = (time.perf_counter() - started) * 1000
            if index >= 2:
                samples.append({"elapsed_ms": elapsed, "result": result.model_dump(mode="json")})
        latencies = sorted(s["elapsed_ms"] for s in samples)
        output["latency"].append(
            {
                "candidates": n,
                "median_ms": statistics.median(latencies),
                "max_ms": max(latencies),
                "samples": samples,
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    config = BackendConfig.model_validate_json(args.config.read_text("utf-8-sig"))
    scorer = make_scorer(config)
    try:
        results = benchmark(scorer, json.loads(args.cases.read_text("utf-8-sig")), args.repeats)
        results["config"] = config.model_dump(mode="json")
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
        print(
            json.dumps(
                {
                    "latency": [
                        {k: v for k, v in row.items() if k != "samples"}
                        for row in results["latency"]
                    ],
                    "quality": [
                        {k: v for k, v in row.items() if k != "result"}
                        for row in results["quality"]
                    ],
                },
                ensure_ascii=False,
            )
        )
    finally:
        if hasattr(scorer, "close"):
            scorer.close()


if __name__ == "__main__":
    main()
