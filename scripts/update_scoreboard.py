#!/usr/bin/env python3
"""Parse agent_arena_eval.json into SCOREBOARD row fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_JSON = ROOT / "experiments" / "outputs" / "team_prod" / "evaluations" / "agent_arena_eval.json"


def difficulty_rates(data: dict) -> tuple[float, float, float]:
    buckets = {1: [], 2: [], 3: []}
    for info in data.get("individual", {}).values():
        d = info.get("difficulty")
        if d in buckets:
            buckets[d].append(bool(info.get("success")))
    def rate(xs: list[bool]) -> float:
        return 100.0 * sum(xs) / len(xs) if xs else 0.0
    return rate(buckets[1]), rate(buckets[2]), rate(buckets[3])


def main() -> None:
    if not EVAL_JSON.is_file():
        print("missing", EVAL_JSON, file=sys.stderr)
        raise SystemExit(1)
    data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    agg = data.get("aggregate", {})
    tgc = agg.get("task_goal_completion", 0)
    sgc = agg.get("scenario_goal_completion", 0)
    d1, d2, d3 = difficulty_rates(data)
    passed = sum(1 for v in data.get("individual", {}).values() if v.get("success"))
    total = len(data.get("individual", {}))
    print(f"TGC={tgc} SGC={sgc} diff1={d1:.0f}% diff2={d2:.0f}% diff3={d3:.0f}% ({passed}/{total})")


if __name__ == "__main__":
    main()
