#!/usr/bin/env python3
"""Reject diffs that hardcode eval/dev task IDs (disqualification risk)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Official eval IDs + common dev ID shape — block literal task_id strings in agent code.
TASK_ID_RE = re.compile(r"\b[a-f0-9]{7}_[123]\b")

ALLOWLIST_FILES = {
    "scripts/check_no_task_ids.py",
    "scripts/analyze_traces.py",
    "scripts/export_task_logs.py",
    "scripts/build_aggregate_review.py",
    "EVAL.md",
    "FAILURES.md",
    "CHANGELOG.md",
    "SCOREBOARD.md",
}

SCAN_FILES = [
    ROOT / "agent.py",
    ROOT / "hydradb_context.py",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def scan_file(path: Path) -> list[str]:
    if rel(path) in ALLOWLIST_FILES:
        return []
    text = path.read_text(encoding="utf-8")
    hits: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        if "task_id" in line.lower() and TASK_ID_RE.search(line):
            # allow reading world.task.id / trace task_id metadata, not comparisons
            if re.search(r"task_id\s*==|task_id\s*!=|['\"]" + TASK_ID_RE.pattern, line):
                hits.append(f"{rel(path)}:{i}: {line.strip()[:120]}")
        elif TASK_ID_RE.search(line) and "task_id" not in line.lower():
            hits.append(f"{rel(path)}:{i}: {line.strip()[:120]}")
    return hits


def main() -> int:
    violations: list[str] = []
    for path in SCAN_FILES:
        if path.is_file():
            violations.extend(scan_file(path))
    if violations:
        print("TASK_ID hardcoding detected (disallowed):", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    print("OK: no task_id hardcoding in agent.py / hydradb_context.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
