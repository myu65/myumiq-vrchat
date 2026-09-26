"""Small Japanese conversation/intent gate; replies require human semantic review."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from myumiq_vrchat import cognition
from myumiq_vrchat.body import WorldObject, WorldState
from myumiq_vrchat.cli import outside_repo
from myumiq_vrchat.cognition import LLMConfig, decide_conversation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, help="independent JSON cases frozen before evaluation")
    args = parser.parse_args()
    output = outside_repo(args.output)
    if output.exists():
        raise ValueError("output must be new")
    config = LLMConfig(base_url=args.url, model=args.model, timeout_s=30)
    world = WorldState(objects=(WorldObject(name="person", position=(0.4, -0.25, 1.2),
                                          source="fixture", kind="player"),))
    cases = [
        ("こんにちは。右手を振ってください。", "WAVE", "right", None),
        ("左手だけ振って。右手は振らないで。", "WAVE", "left", None),
        ("手は動かさず、少し待っていてね。", "WAIT", None, None),
        ("personを見てください。", "LOOK_AT", None, "person"),
        ("personへ右手を伸ばして。", "REACH", "right", "person"),
        ("ここにいないtaroを見てください。", "WAIT", None, None),
        ("踊ってください。", "WAIT", None, None),
        ("今は動かず、こんにちはと日本語で挨拶して。", "WAIT", None, None),
    ]
    if args.cases is not None:
        data = json.loads(args.cases.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list) or not 1 <= len(data) <= 100:
            raise ValueError("case file must contain 1..100 cases")
        cases = []
        for item in data:
            if set(item) != {"prompt", "skill", "hand", "target"}:
                raise ValueError("invalid case fields")
            if not isinstance(item["prompt"], str) or not 0 < len(item["prompt"]) <= 2000:
                raise ValueError("invalid case prompt")
            expected = cognition.Goal(skill=item["skill"], hand=item["hand"],
                                      target=item["target"], duration_s=1)
            expected.validate_world(world)
            cases.append((item["prompt"], expected.skill, expected.hand, expected.target))
    fingerprint = hashlib.sha256(Path(cognition.__file__).read_bytes()).hexdigest()
    rows = []
    for prompt, skill, hand, target in cases:
        started = time.perf_counter()
        row = {"prompt": prompt, "expected": {"skill": skill, "hand": hand, "target": target}}
        try:
            reply, decision = decide_conversation(config, prompt, world)
            goal = decision.goal
            row.update(valid=True, reply=reply, goal=goal.model_dump(),
                       correct=goal.skill == skill and goal.hand == hand and goal.target == target)
        except Exception as exc:
            row.update(valid=False, correct=False, error=f"{type(exc).__name__}: {exc}")
        row["latency_s"] = time.perf_counter() - started
        rows.append(row)
        result = {"model": args.model, "cases": rows, "complete": len(rows) == len(cases),
                  "cognition_sha256": fingerprint,
                  "case_set_sha256": hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode()).hexdigest(),
                  "valid_rate": sum(r["valid"] for r in rows) / len(rows),
                  "intent_accuracy": sum(r["correct"] for r in rows) / len(rows),
                  "median_latency_s": statistics.median(r["latency_s"] for r in rows),
                  "reply_semantics_reviewed": False}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))


if __name__ == "__main__":
    main()
