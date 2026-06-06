#!/usr/bin/env python3
"""Analyze AppWorld traces and task logs; update FAILURES.md + JSON report."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = ROOT / "traces" / "team_prod"
LOGS_GLOB = ROOT / "experiments" / "outputs" / "team_prod" / "tasks"
REPORT_JSON = ROOT / "failure_report.json"
FAILURES_MD = ROOT / "FAILURES.md"

CATEGORIES = [
    "WRONG_API_NAME",
    "TOKEN_MISUSE",
    "FIELD_KEYERROR",
    "VALIDATION_422",
    "CARD_VALIDITY",
    "STUBBED_CALL",
    "FAKE_RECIPIENT",
    "PAGINATION_MISS",
    "ANSWER_FORMAT",
    "CASING",
    "RAN_OUT",
    "LLM_ERROR",
    "OTHER",
]


def classify_step(step: dict[str, Any], instruction: str = "") -> list[tuple[str, str]]:
    """Return list of (category, snippet) for one trace step."""
    hits: list[tuple[str, str]] = []
    obs = str(step.get("observation") or "")
    code = str(step.get("code") or "")
    reply = str(step.get("reply") or "")
    blob = obs + "\n" + code + "\n" + reply

    if step.get("llm_error"):
        hits.append(("LLM_ERROR", str(step.get("llm_error") or "")[:200]))
        return hits

    if "No API named" in obs or "has no attribute" in obs and "apis." in obs:
        m = re.search(r"No API named '([^']+)'", obs)
        name = m.group(1) if m else "?"
        hits.append(("WRONG_API_NAME", f"No API named {name!r}"))

    if "401" in obs or "not authorized" in obs.lower() or "Invalid credentials" in obs:
        hits.append(("TOKEN_MISUSE", obs.strip()[:200]))

    if "KeyError" in obs:
        m = re.search(r"KeyError: '?([^'\n]+)'?", obs)
        hits.append(("FIELD_KEYERROR", f"KeyError {m.group(1) if m else '?'}"))

    if "422" in obs or "Validation error" in obs:
        hits.append(("VALIDATION_422", obs.strip()[:200]))

    if re.search(r"expired|insufficient balance|add_payment_card", obs + code, re.I):
        hits.append(("CARD_VALIDITY", obs.strip()[:160] or code[:160]))

    if re.search(r"#\s*apis\.|print\(.*placeholder|TODO.*apis\.", code, re.I):
        hits.append(("STUBBED_CALL", code.strip()[:160]))

    if re.search(r"@example\.com|name@example", blob, re.I):
        hits.append(("FAKE_RECIPIENT", blob[:160]))

    if re.search(r"page_index\s*=\s*0|page_limit\s*=\s*5", code) and "page_index=1" not in code:
        if any(k in instruction.lower() for k in ("paginate", "all alarms", "all draft", "cable bill")):
            hits.append(("PAGINATION_MISS", "only first page fetched"))

    if re.search(r"account_name.*==.*['\"]Gmail|SimpleNote|Splitwise|Phone['\"]", code):
        hits.append(("CASING", "Title-case account_name lookup"))

    if re.search(r"complete_task\(answer\s*=\s*['\"][^'\"]{3,}", code):
        hits.append(("ANSWER_FORMAT", code.strip()[:160]))

    if step.get("completed") is False and not hits and not step.get("execution_error"):
        if "StopIteration" in obs and "account_name" in code:
            hits.append(("CASING", "StopIteration on password lookup"))

    return hits or []


def analyze_trace(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    task_id = data.get("task_id", path.stem)
    instruction = str(data.get("instruction") or "")
    steps = data.get("steps") or []
    categories: list[tuple[str, str]] = []
    for step in steps:
        categories.extend(classify_step(step, instruction))
    status = data.get("final_status", "?")
    if status == "max_interactions":
        categories.append(("RAN_OUT", f"hit MAX_INTERACTIONS after {len(steps)} steps"))
    return {
        "task_id": task_id,
        "final_status": status,
        "num_steps": len(steps),
        "categories": categories,
        "instruction_preview": instruction[:120],
    }


def load_eval_failures() -> dict[str, bool]:
    eval_path = ROOT / "experiments" / "outputs" / "team_prod" / "evaluations" / "agent_arena_eval.json"
    if not eval_path.is_file():
        return {}
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    return {tid: info.get("success", False) for tid, info in data.get("individual", {}).items()}


def build_report() -> dict[str, Any]:
    trace_files = sorted(TRACES_DIR.glob("*.json")) if TRACES_DIR.is_dir() else []
    per_task = [analyze_trace(p) for p in trace_files]
    eval_success = load_eval_failures()

    cat_counter: Counter[str] = Counter()
    cat_apps: dict[str, Counter[str]] = defaultdict(Counter)
    cat_snippets: dict[str, list[str]] = defaultdict(list)
    cat_tasks: dict[str, set[str]] = defaultdict(set)

    for task in per_task:
        tid = task["task_id"]
        for cat, snippet in task["categories"]:
            cat_counter[cat] += 1
            cat_tasks[cat].add(tid)
            if len(cat_snippets[cat]) < 3:
                cat_snippets[cat].append(f"{tid}: {snippet[:140]}")

    ranked = sorted(
        cat_counter.keys(),
        key=lambda c: cat_counter[c] * len(cat_tasks[c]),
        reverse=True,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trace_count": len(per_task),
        "eval_results": eval_success,
        "category_counts": dict(cat_counter),
        "category_task_counts": {c: len(cat_tasks[c]) for c in ranked},
        "ranked_categories": ranked,
        "snippets": {c: cat_snippets[c] for c in ranked if cat_snippets[c]},
        "per_task": per_task,
    }


def render_failures_md(report: dict[str, Any]) -> str:
    lines = [
        "# Failure taxonomy (live)",
        "",
        f"_Updated: {report['generated_at']}_",
        "",
        "## Ranked categories (frequency × tasks blocked)",
        "",
        "| rank | category | hits | tasks |",
        "|------|----------|------|-------|",
    ]
    for i, cat in enumerate(report["ranked_categories"][:15], 1):
        lines.append(
            f"| {i} | {cat} | {report['category_counts'].get(cat, 0)} | "
            f"{report['category_task_counts'].get(cat, 0)} |"
        )
    lines.extend(["", "## Representative snippets", ""])
    for cat in report["ranked_categories"][:8]:
        snippets = report["snippets"].get(cat) or []
        if not snippets:
            continue
        lines.append(f"### {cat}")
        for s in snippets:
            lines.append(f"- {s}")
        lines.append("")

    if report.get("eval_results"):
        lines.extend(["## Eval pass/fail", ""])
        for tid, ok in sorted(report["eval_results"].items()):
            lines.append(f"- `{tid}`: {'PASS' if ok else '**FAIL**'}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    report = build_report()
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    FAILURES_MD.write_text(render_failures_md(report), encoding="utf-8")
    print(f"Wrote {REPORT_JSON} and updated {FAILURES_MD}")
    if report["ranked_categories"]:
        top = report["ranked_categories"][0]
        print(f"Top category: {top} ({report['category_counts'].get(top, 0)} hits, "
              f"{report['category_task_counts'].get(top, 0)} tasks)")


if __name__ == "__main__":
    main()
