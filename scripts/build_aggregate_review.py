"""Build a compact aggregate JSON of traces + experiment logs for external review."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "aggregate_run_review.json"
PREVIEW = 280


def preview(text: str, limit: int = PREVIEW) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def load_eval_lookup(experiment: str, dataset: str) -> dict[str, dict]:
    path = ROOT / "experiments" / "outputs" / experiment / "evaluations" / f"{dataset}.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("individual", {})


def summarize_api_calls(path: Path) -> dict:
    if not path.is_file():
        return {"count": 0, "apps": [], "endpoints": []}
    apps: set[str] = set()
    endpoints: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        app = row.get("app_name") or row.get("app") or ""
        api = row.get("api_name") or row.get("api") or ""
        if app:
            apps.add(app)
        if app and api:
            endpoints.append(f"{app}.{api}")
    return {
        "count": len(endpoints),
        "apps": sorted(apps),
        "endpoints": endpoints[-8:],  # last few calls often include complete_task
    }


def summarize_trace(trace_path: Path) -> dict:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    steps = trace.get("steps", [])
    exec_errors: list[str] = []
    timeline: list[dict] = []
    for s in steps:
        obs = str(s.get("observation") or "")
        if "Execution failed" in obs or "Execution raised an exception" in obs:
            exec_errors.append(preview(obs, 400))
        timeline.append({
            "step": s.get("step"),
            "code_preview": preview(s.get("code") or s.get("reply") or "", 200),
            "observation_preview": preview(obs, 200),
            "completed": s.get("completed"),
            "llm_error": bool(s.get("llm_error")),
        })
    last_code = steps[-1].get("code") if steps else None
    return {
        "task_id": trace.get("task_id"),
        "dataset": trace.get("dataset"),
        "experiment": trace.get("experiment"),
        "model": trace.get("model"),
        "max_interactions": trace.get("max_interactions"),
        "supervisor": trace.get("supervisor"),
        "instruction": trace.get("instruction"),
        "final_status": trace.get("final_status"),
        "num_steps": len(steps),
        "execution_error_count": len(exec_errors),
        "execution_errors_sample": exec_errors[:3],
        "last_code_preview": preview(last_code or "", 300),
        "step_timeline": timeline,
        "trace_path": str(trace_path.relative_to(ROOT)).replace("\\", "/"),
    }


def task_record(
    task_id: str,
    experiment: str,
    trace_summary: dict | None,
    eval_info: dict | None,
) -> dict:
    task_dir = ROOT / "experiments" / "outputs" / experiment / "tasks" / task_id
    logs = task_dir / "logs"
    eval_dir = task_dir / "evaluation"
    paths = {
        "experiment_task_dir": str(task_dir.relative_to(ROOT)).replace("\\", "/"),
        "environment_io": str((logs / "environment_io.md").relative_to(ROOT)).replace("\\", "/")
        if (logs / "environment_io.md").is_file()
        else None,
        "api_calls": str((logs / "api_calls.jsonl").relative_to(ROOT)).replace("\\", "/")
        if (logs / "api_calls.jsonl").is_file()
        else None,
        "evaluation_report": str((eval_dir / "report.md").relative_to(ROOT)).replace("\\", "/")
        if (eval_dir / "report.md").is_file()
        else None,
    }
    api_summary = summarize_api_calls(logs / "api_calls.jsonl") if paths["api_calls"] else {}
    eval_summary = None
    if eval_info:
        eval_summary = {
            "passed": eval_info.get("success"),
            "difficulty": eval_info.get("difficulty"),
            "failures": [
                preview(f.get("requirement", "") + ": " + f.get("trace", ""), 350)
                for f in eval_info.get("failures", [])[:2]
            ],
        }
    rec = {
        "task_id": task_id,
        "experiment": experiment,
        "log_paths": paths,
        "api_calls_summary": api_summary,
        "evaluation": eval_summary,
    }
    if trace_summary:
        rec.update({
            k: trace_summary[k]
            for k in (
                "dataset",
                "model",
                "instruction",
                "final_status",
                "num_steps",
                "execution_error_count",
                "execution_errors_sample",
                "last_code_preview",
                "step_timeline",
                "trace_path",
            )
            if k in trace_summary
        })
    return rec


def collect_experiment(experiment: str, trace_dir: Path | None, dataset_for_eval: str | None) -> dict:
    eval_lookup = load_eval_lookup(experiment, dataset_for_eval) if dataset_for_eval else {}
    trace_by_id: dict[str, dict] = {}
    if trace_dir and trace_dir.is_dir():
        for p in sorted(trace_dir.glob("*.json")):
            if p.name.startswith("_"):
                continue
            summary = summarize_trace(p)
            trace_by_id[summary["task_id"]] = summary

    task_root = ROOT / "experiments" / "outputs" / experiment / "tasks"
    task_ids = sorted(trace_by_id.keys())
    if task_root.is_dir():
        for d in task_root.iterdir():
            if d.is_dir() and d.name not in task_ids:
                task_ids.append(d.name)
    task_ids.sort()

    tasks = [
        task_record(tid, experiment, trace_by_id.get(tid), eval_lookup.get(tid))
        for tid in task_ids
    ]
    status_counts: dict[str, int] = {}
    for t in tasks:
        st = t.get("final_status") or "unknown"
        status_counts[st] = status_counts.get(st, 0) + 1
    eval_passed = sum(1 for t in tasks if t.get("evaluation") and t["evaluation"].get("passed"))
    eval_total = sum(1 for t in tasks if t.get("evaluation") is not None)

    return {
        "experiment": experiment,
        "trace_dir": str(trace_dir.relative_to(ROOT)).replace("\\", "/") if trace_dir else None,
        "tasks_dir": str(task_root.relative_to(ROOT)).replace("\\", "/") if task_root.is_dir() else None,
        "task_count": len(tasks),
        "final_status_counts": status_counts,
        "evaluation": {
            "dataset": dataset_for_eval,
            "scored_tasks": eval_total,
            "passed": eval_passed,
            "tgc_percent": round(100 * eval_passed / eval_total, 1) if eval_total else None,
        },
        "tasks": tasks,
    }


def main() -> None:
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Compact aggregate for reviewing agent runs (send to Opus). Full logs linked per task.",
        "note": "observation/code fields are truncated previews; open log_paths for full detail.",
        "runs": [
            collect_experiment(
                "team_prod",
                ROOT / "traces" / "team_prod",
                "agent_arena_eval",
            ),
            collect_experiment(
                "team_demo",
                ROOT / "traces" / "team_demo",
                "dev",
            ),
        ],
    }
    OUT.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
