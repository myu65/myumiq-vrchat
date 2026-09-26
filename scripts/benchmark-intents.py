"""Small repeatable Japanese structured-intent benchmark for a running local server."""

import argparse
import json
import statistics
import time
from pathlib import Path

from myumiq_vrchat.body import WorldObject, WorldState
from myumiq_vrchat.cognition import LLMConfig, decide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    world = WorldState(
        objects=(WorldObject(name="player_1", position=(1.0, 0.2, 1.6), source="manual", kind="player"),)
    )
    cases = [
        ("右手を振って", "WAVE", "right"),
        ("player_1を見て", "LOOK_AT", None),
        ("少し待って", "WAIT", None),
        ("左手を振って", "WAVE", "left"),
        ("player_1に右手を伸ばして", "REACH", "right"),
        ("踊って", "WAIT", None),
    ]
    config = LLMConfig(base_url=args.url, model=args.model, timeout_s=30)
    rows = []
    for prompt, skill, hand in cases:
        started = time.perf_counter()
        try:
            decision = decide(config, prompt, world)
            correct = decision.goal.skill == skill and (hand is None or decision.goal.hand == hand)
            rows.append(
                {
                    "prompt": prompt,
                    "expected": skill,
                    "goal": decision.goal.model_dump(),
                    "valid": True,
                    "correct": correct,
                    "latency_s": time.perf_counter() - started,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "prompt": prompt,
                    "expected": skill,
                    "valid": False,
                    "correct": False,
                    "latency_s": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    result = {
        "model": args.model,
        "valid_rate": sum(x["valid"] for x in rows) / len(rows),
        "accuracy": sum(x["correct"] for x in rows) / len(rows),
        "median_latency_s": statistics.median(x["latency_s"] for x in rows),
        "cases": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
