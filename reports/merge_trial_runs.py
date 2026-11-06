"""Merge single-trial `corral bench` runs into one verbose report per run.

Corral-mini runs each trial as its own 1-trial `corral bench` run, named
`<agent>__<model>__L<level>__<task>__k<k>`, so idle slots can start the next
trial at once. This script groups those runs by agent, model and level and
writes the standard report layout (see README.md):

    <out>/level_<L>/<agent>__<model>/report.json.gz
    <out>/level_<L>/<agent>__<model>/run-metadata.json
    <out>/level_<L>/<agent>__<model>/provenance.json

Reports are rebuilt in verbose form (messages and tool calls) from each trial's
saved state (its commit database), so runs launched without `--verbose` get
the same content as the October reports. Metrics are recomputed by Corral's own
`BenchmarkResult` over the merged trials. Trial and execution IDs are kept, so
the deterministic Langfuse trace IDs still resolve; each trial's Langfuse
session is its own single-trial run ID.

A group is written only when every task has the expected number of completed
trials (`--trials`). Run from the environment's host venv, which has Corral
installed:

    python reports/merge_trial_runs.py /data/.../runs reports/corral_md \\
        --model kimi-k3.1-pro --model gpt-6.1-sol
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from corral.evaluation import EvaluationResult
from corral.persistence.sharded import ShardedCommitStore
from corral.report.projection import _tool_statistics
from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults

# Runs that finished; "completed_with_errors" marks agent or evaluation errors,
# which are trial outcomes, not infrastructure failures.
FINISHED = {"completed", "completed_with_errors"}
NAME = re.compile(
    r"^(?P<agent>[^_]+(?:-[^_]+)*)__(?P<model>.+)__L(?P<level>\d+)__(?P<task>.+)__k(?P<k>\d+)$"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_trials(runs: Path, models: set[str] | None) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        match = NAME.match(run_dir.name)
        if not match or (models and match["model"] not in models):
            continue
        report = run_dir / "report.json"
        manifest = run_dir / "run-metadata.json"
        if not report.exists() or not manifest.exists():
            continue
        if json.loads(manifest.read_text()).get("status") not in FINISHED:
            continue
        key = (match["agent"], match["model"], int(match["level"]))
        groups[key].append(
            {"dir": run_dir, "task": match["task"], "k": int(match["k"])}
        )
    return groups


async def load_trial(
    trial: dict, unscorable_as_zero: bool
) -> tuple[TaskTrialResult, dict, dict]:
    run_dir: Path = trial["dir"]
    report = json.loads((run_dir / "report.json").read_text())
    ((task_id, task),) = report["task_results"].items()
    (saved,) = task["trials"]
    result_path = next(run_dir.glob("task-*/k-*/result.json"))
    result = json.loads(result_path.read_text())

    store = ShardedCommitStore(run_dir, benchmark_run_id=run_dir.name)
    state = await store.for_execution(result["execution_id"]).materialize(
        result["branch_id"], result["commit_hash"]
    )
    messages = [
        {"agent_run_id": run_id, **dict(message)}
        for run_id, conversation in state.conversations.items()
        for message in conversation
    ]
    evaluation = saved.get("evaluation")
    projected = TaskTrialResult(
        task_id=task_id,
        trial_id=saved["trial_id"],
        score=(
            0.0
            if saved["score"] is None
            and unscorable_as_zero
            and saved.get("evaluation_error")
            else saved["score"]
        ),
        state={},
        tool_statistics=_tool_statistics(state),
        output=saved.get("output"),
        evaluation=(
            EvaluationResult.model_validate(evaluation) if evaluation else None
        ),
        evaluation_error=saved.get("evaluation_error"),
        messages=messages,
        duration=saved.get("duration"),
        token_usage=saved.get("token_usage"),
        error_message=saved.get("error_message"),
        surrendered=bool(saved.get("surrendered")),
    )
    if bool(projected.success) != bool(saved.get("success")):
        raise ValueError(f"{run_dir.name}: rebuilt success differs from report")
    return projected, report, json.loads((run_dir / "run-metadata.json").read_text())


async def merge(
    key: tuple, trials: list[dict], out: Path, run_prefix: str, unscorable_as_zero: bool
) -> dict:
    agent, model, level = key
    trials = sorted(trials, key=lambda t: (t["task"], t["k"]))
    loaded = [await load_trial(t, unscorable_as_zero) for t in trials]

    task_results: dict[str, TaskTrialResults] = {}
    for projected, _, _ in loaded:
        task_results.setdefault(
            projected.task_id, TaskTrialResults(task_id=projected.task_id)
        ).trials.append(projected)
    trials_per_task = {len(t.trials) for t in task_results.values()}
    (n_trials,) = trials_per_task

    run_id = f"{run_prefix}__{agent}__{model}__L{level}"
    first_meta = loaded[0][1]["metadata"]
    metadata = json.loads(json.dumps(first_meta))
    benchmark = metadata["benchmark"]
    for key_name in ("dependency_graph", "max_iterations_by_task"):
        merged: dict = {}
        for _, report, _ in loaded:
            merged.update(report["metadata"]["benchmark"].get(key_name) or {})
        benchmark[key_name] = dict(sorted(merged.items()))
    by_task: dict = {}
    for _, report, _ in loaded:
        by_task.update(report["metadata"]["agent"]["by_task"])
    metadata["agent"]["by_task"] = dict(sorted(by_task.items()))
    benchmark["trials_per_task"] = n_trials
    benchmark["k_values"] = list(range(1, n_trials + 1))
    benchmark["task_ids"] = sorted(task_results)
    metadata["benchmark_run_id"] = run_id
    metadata["verbose"] = True
    metadata["merged_from"] = [t["dir"].name for t in trials]

    result = BenchmarkResult(
        task_results=dict(sorted(task_results.items())),
        k=list(range(1, n_trials + 1)),
        total_duration=sum(
            r["metrics"].get("total_benchmark_duration") or 0 for _, r, _ in loaded
        ),
        verbose=True,
        metadata=metadata,
    )
    data = result._prepare_report_data(result.calculate_metrics())

    folder = out / f"level_{level}" / f"{agent}__{model}"
    folder.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data, indent=2, default=str).encode()
    with gzip.GzipFile(folder / "report.json.gz", "wb", mtime=0) as fh:
        fh.write(raw)
    manifests = [m for _, _, m in loaded]
    (folder / "run-metadata.json").write_text(
        json.dumps(
            {
                "benchmark_run_id": run_id,
                "environment": manifests[0]["environment"],
                "agent": manifests[0]["agent"],
                "model": manifests[0]["model"],
                "sandbox": manifests[0]["sandbox"],
                "status": "completed",
                "trials_per_task": n_trials,
                "requested_task_ids": sorted(task_results),
                "created_at": min(m["created_at"] for m in manifests),
                "completed_at": max(m["completed_at"] for m in manifests),
                "note": "merged from single-trial runs; each trial's Langfuse "
                "session is its own run ID (see provenance.json)",
            },
            indent=2,
        )
        + "\n"
    )
    (folder / "provenance.json").write_text(
        json.dumps(
            {
                "source": "merged single-trial corral bench runs, rebuilt verbose "
                "from each trial's commit database by reports/merge_trial_runs.py",
                "image_digest": sorted(
                    {
                        r["metadata"]["benchmark"]
                        .get("sandbox", {})
                        .get("docker", {})
                        .get("image_digest")
                        or ""
                        for _, r, _ in loaded
                    }
                ),
                "unscorable_scored_as_zero": [
                    {"execution_id": p.trial_id, "evaluation_error": p.evaluation_error}
                    for p, r, _ in loaded
                    if p.score == 0.0
                    and next(iter(r["task_results"].values()))["trials"][0]["score"]
                    is None
                ],
                "trials": [
                    {
                        "run_id": t["dir"].name,
                        "execution_id": p.trial_id,
                        "report_sha256": sha256(t["dir"] / "report.json"),
                    }
                    for t, (p, _, _) in zip(trials, loaded, strict=True)
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "folder": folder,
        "tasks": len(task_results),
        "trials": len(loaded),
        "score": data["metrics"].get("Average Score"),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, help="directory of single-trial runs")
    parser.add_argument("out", type=Path, help="environment report folder")
    parser.add_argument("--model", action="append", help="only these models")
    parser.add_argument("--trials", type=int, default=5, help="trials per task")
    parser.add_argument("--tasks", type=int, default=3, help="tasks per level")
    parser.add_argument("--run-prefix", default="corral-mini-v1")
    parser.add_argument(
        "--unscorable-as-zero",
        action="store_true",
        help="score 0 for trials whose evaluation failed (score None), e.g. an "
        "unparseable submission; Corral's metrics are undefined otherwise. The "
        "evaluation error is kept and the trials are listed in provenance.json",
    )
    args = parser.parse_args()

    groups = find_trials(args.runs, set(args.model) if args.model else None)
    status = 0
    for key, trials in sorted(groups.items()):
        per_task: dict[str, set[int]] = defaultdict(set)
        for t in trials:
            per_task[t["task"]].add(t["k"])
        complete = len(per_task) == args.tasks and all(
            ks == set(range(1, args.trials + 1)) for ks in per_task.values()
        )
        label = "{}__{} L{}".format(*key)
        if not complete:
            sys.stdout.write(f"skip {label}: {len(trials)} completed trials\n")
            continue
        try:
            info = await merge(
                key, trials, args.out, args.run_prefix, args.unscorable_as_zero
            )
        except Exception as exc:  # report every group, then fail
            sys.stderr.write(f"FAILED {label}: {exc!r}\n")
            status = 1
            continue
        sys.stdout.write(
            f"wrote {info['folder']}: {info['tasks']} tasks, "
            f"{info['trials']} trials, average score {info['score']}\n"
        )
    return status


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
