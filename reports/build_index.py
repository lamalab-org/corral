"""Build runs.csv and trials.csv for one environment's benchmark reports.

Reads every ``reports/<env>/level_<L>/<agent>__<model>/report.json.gz`` (or
``.json.xz``, used where gzip compresses a report far worse) and
writes ``reports/<env>/runs.csv`` (one row per run) and
``reports/<env>/trials.csv`` (one row per trial, with its Langfuse links).
Anything under ``archive/`` or ``provenance/`` is ignored.

Langfuse IDs are derived the same way Corral sets them when tracing:
the session ID is the benchmark run ID and the trace ID is
``sha256("corral:" + execution_id)[:32]``
(``corral.observability.langfuse.deterministic_trace_id``).

Usage:
    python reports/build_index.py reports/<env>
    python reports/build_index.py reports/<env> --verify   # check traces exist in Langfuse
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import lzma
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LANGFUSE_HOST = os.environ.get(
    "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
).rstrip("/")
LANGFUSE_PROJECT_ID = os.environ.get("LANGFUSE_PROJECT_ID", "cmsio90yc092lad0ggc5lyo0x")

AGENTS = {
    "react": "react",
    "claude-code": "claude-code",
    "claude_code": "claude-code",
    "claudecode": "claude-code",
    "codex": "codex",
    "openhands": "openhands",
    "ai-scientist": "ai-scientist",
    "ai_scientist": "ai-scientist",
    "aiscientist": "ai-scientist",
}
MODELS = {
    "claude-opus-5-5": ("claude-opus-5-5", "anthropic/claude-opus-5-5"),
    "gpt-5.6-sol": ("gpt-5.6-sol", "openai/gpt-5.6-sol"),
    "kimi-k3": ("kimi-k3", "openrouter/moonshotai/kimi-k3"),
}

RUN_FIELDS = [
    "environment",
    "level",
    "agent",
    "model",
    "model_id",
    "run_id",
    "tasks",
    "trials_per_task",
    "trials",
    "passes",
    "error_trials",
    "average_score",
    "pass_at_1",
    "pass_at_k",
    "pass_hat_k",
    "input_tokens",
    "output_tokens",
    "generated_at",
    "report_file",
    "report_sha256",
    "langfuse_session_url",
]
TRIAL_FIELDS = [
    "environment",
    "level",
    "agent",
    "model",
    "run_id",
    "task_id",
    "trial",
    "execution_id",
    "score",
    "success",
    "error",
    "duration_s",
    "input_tokens",
    "output_tokens",
    "tool_calls",
    "langfuse_session_id",
    "langfuse_trace_id",
    "langfuse_trace_url",
    "report_file",
]


def trace_id(execution_id: str) -> str:
    return hashlib.sha256(f"corral:{execution_id}".encode()).hexdigest()[:32]


def trace_url(tid: str) -> str:
    return f"{LANGFUSE_HOST}/project/{LANGFUSE_PROJECT_ID}/traces/{tid}"


def session_url(session_id: str) -> str:
    return f"{LANGFUSE_HOST}/project/{LANGFUSE_PROJECT_ID}/sessions/{session_id}"


def canonical(agent: str, model: str) -> tuple[str, str, str]:
    model_key = model.rsplit("/", 1)[-1]
    if agent not in AGENTS or model_key not in MODELS:
        raise ValueError(f"unknown agent/model: {agent!r} {model!r}")
    return AGENTS[agent], *MODELS[model_key]


def runtime(metadata: dict) -> dict:
    return next(iter(metadata["agent"]["by_task"].values()))["runtime"]


def index_run(env_dir: Path, report: Path, env: str) -> tuple[dict, list[dict]]:
    raw = (lzma if report.suffix == ".xz" else gzip).decompress(report.read_bytes())
    data = json.loads(raw)
    meta = data["metadata"]
    level = int(report.parent.parent.name.removeprefix("level_"))
    agent, model, model_id = canonical(runtime(meta)["name"], runtime(meta)["model"])
    if report.parent.name != f"{agent}__{model}":
        raise ValueError(f"{report}: directory does not match {agent}__{model}")
    run_id = meta["benchmark_run_id"]
    k = int(meta["benchmark"]["trials_per_task"])
    metrics = data["metrics"]
    rel = report.relative_to(env_dir).as_posix()

    trials = []
    for task_id, task in sorted(data["task_results"].items()):
        for slot, t in enumerate(task["trials"]):
            usage = t.get("token_usage") or {}
            tid = trace_id(t["trial_id"])
            # A substituted rerun trial keeps its own execution ID, and so its own session.
            session = t["trial_id"].rsplit(":", 2)[0]
            trials.append(
                {
                    "environment": env,
                    "level": level,
                    "agent": agent,
                    "model": model,
                    "run_id": run_id,
                    "task_id": task_id,
                    "trial": slot,
                    "execution_id": t["trial_id"],
                    "score": t.get("score"),
                    "success": t.get("success"),
                    "error": (
                        t.get("error_message") or t.get("evaluation_error") or ""
                    ).strip(),
                    "duration_s": round(t["duration"], 3)
                    if t.get("duration") is not None
                    else "",
                    "input_tokens": usage.get("input_tokens", ""),
                    "output_tokens": usage.get("output_tokens", ""),
                    "tool_calls": usage.get("tool_calls", ""),
                    "langfuse_session_id": session,
                    "langfuse_trace_id": tid,
                    "langfuse_trace_url": trace_url(tid),
                    "report_file": rel,
                }
            )

    usage = metrics.get("Total Token Usage") or {}
    run = {
        "environment": env,
        "level": level,
        "agent": agent,
        "model": model,
        "model_id": model_id,
        "run_id": run_id,
        "tasks": len(data["task_results"]),
        "trials_per_task": k,
        "trials": len(trials),
        "passes": sum(bool(t["success"]) for t in trials),
        "error_trials": sum(bool(t["error"]) for t in trials),
        "average_score": round(float(metrics["Average Score"]), 4),
        "pass_at_1": round(float(metrics.get("Pass@1", "nan")), 4),
        "pass_at_k": round(float(metrics.get(f"Pass@{k}", "nan")), 4),
        "pass_hat_k": round(float(metrics.get(f"Pass^{k}", "nan")), 4),
        "input_tokens": usage.get("input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""),
        "generated_at": data.get("generated_at", ""),
        "report_file": rel,
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "langfuse_session_url": session_url(run_id),
    }
    return run, trials


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify(trials: list[dict]) -> list[str]:
    """Return the trace IDs that are not in their Langfuse session."""
    auth = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()

    def get(url: str) -> dict:
        request = urllib.request.Request(
            url, headers={"Authorization": f"Basic {auth}"}
        )
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code != 429 and error.code < 500:
                    raise
                delay = float(error.headers.get("Retry-After") or 2**attempt)
            except urllib.error.URLError:
                delay = 2**attempt
            time.sleep(min(delay, 60))
        raise RuntimeError(f"Langfuse request kept failing: {url}")

    found = set()
    for session in sorted({t["langfuse_session_id"] for t in trials}):
        page = 1
        while True:
            query = urllib.parse.urlencode(
                {"sessionId": session, "limit": 100, "page": page, "fields": "core"}
            )
            data = get(f"{LANGFUSE_HOST}/api/public/traces?{query}")
            found.update(trace["id"] for trace in data["data"])
            if page >= data["meta"]["totalPages"]:
                break
            page += 1
    return [
        t["langfuse_trace_id"] for t in trials if t["langfuse_trace_id"] not in found
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("env_dir", type=Path)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check every trace exists in its Langfuse session",
    )
    args = parser.parse_args()
    env_dir = args.env_dir.resolve()
    env = env_dir.name

    runs, trials = [], []
    for report in sorted(env_dir.glob("level_*/*/report.json.[gx]z")):
        run, run_trials = index_run(env_dir, report, env)
        runs.append(run)
        trials.extend(run_trials)
    write_csv(env_dir / "runs.csv", RUN_FIELDS, runs)
    write_csv(env_dir / "trials.csv", TRIAL_FIELDS, trials)
    sys.stdout.write(f"{env}: {len(runs)} runs, {len(trials)} trials\n")

    if args.verify:
        missing = verify(trials)
        sys.stdout.write(
            f"Langfuse: {len(trials) - len(missing)}/{len(trials)} traces found\n"
        )
        for tid in missing:
            sys.stdout.write(f"  missing {tid}\n")


if __name__ == "__main__":
    main()
