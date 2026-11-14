"""Merge single-trial `corral bench` runs into one verbose report per run.

A queue-driven campaign (one `corral bench --trials 1` run per trial) leaves
one report per trial. This groups those runs by (level, agent, model) and
writes one report per group in the layout of reports/README.md:

    <out>/level_<L>/<agent>__<model>/report.json.gz
    <out>/level_<L>/<agent>__<model>/provenance.json

Each trial's messages and tool calls are rebuilt from its final state snapshot
with the same code `corral bench --verbose` uses, so the merged reports match
verbose reports even when the runs were not verbose. Metrics are recomputed by
`BenchmarkResult` over all merged trials. Trials keep their original
execution IDs, so their Langfuse trace IDs and per-trial sessions still
resolve.

Usage (from the repository root, in an environment with Corral installed):

    python scripts/merge_trial_reports.py RUNS_DIR OUT_DIR --run-prefix PREFIX

    RUNS_DIR      directory with one run directory per trial
    OUT_DIR       e.g. reports/psychometrics
    --run-prefix  merged run IDs are <prefix>-l<L>-<agent>-<model>
"""

# ruff: noqa: T201  (progress output of a command-line script)

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from corral.core.state import ExecutionState
from corral.evaluation import EvaluationResult
from corral.report.projection import _tool_statistics, normalise_k_values
from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reports"))
from build_index import AGENTS, MODELS


def load_trial(run_dir: Path) -> tuple[dict, TaskTrialResult, dict]:
    """Return (report metadata, verbose trial, run metadata) for a 1-trial run."""
    report = json.loads((run_dir / "report.json").read_text())
    ((task_id, task),) = report["task_results"].items()
    (trial,) = task["trials"]
    (trial_dir,) = run_dir.glob(f"task-{task_id}/k-*")
    result = json.loads((trial_dir / "result.json").read_text())
    snapshot = json.loads((trial_dir / "state-snapshots" / "final.json").read_text())
    if snapshot["through_commit_hash"] != result["commit_hash"]:
        raise ValueError(f"{run_dir}: final snapshot is not the final commit")
    state = ExecutionState.model_validate(snapshot)
    messages = [
        {"agent_run_id": run_id, **dict(message)}
        for run_id, conversation in state.conversations.items()
        for message in conversation
    ]
    statistics = _tool_statistics(state)
    if sum(statistics["by_tool"].values()) != sum(trial.get("by_tool", {}).values()):
        raise ValueError(f"{run_dir}: rebuilt tool calls do not match the report")
    evaluation = trial.get("evaluation")
    rebuilt = TaskTrialResult(
        task_id=task_id,
        trial_id=trial["trial_id"],
        score=trial["score"],
        state={},
        tool_statistics=statistics,
        output=trial.get("output"),
        evaluation=(
            EvaluationResult.model_validate(evaluation) if evaluation else None
        ),
        evaluation_error=trial.get("evaluation_error"),
        messages=messages,
        duration=trial.get("duration"),
        token_usage=trial.get("token_usage"),
        error_message=trial.get("error_message"),
        surrendered=bool(trial.get("surrendered")),
    )
    if rebuilt.success != trial["success"]:
        raise ValueError(f"{run_dir}: success differs after rebuilding")
    run_metadata = json.loads((run_dir / "run-metadata.json").read_text())
    return report["metadata"], rebuilt, run_metadata


def merged_metadata(run_id: str, parts: list[dict], trials_per_task: int) -> dict:
    base = json.loads(json.dumps(parts[0]))
    task_ids = sorted({t for p in parts for t in p["benchmark"]["task_ids"]})
    benchmark = base["benchmark"]
    for key in ("dependency_graph", "max_iterations_by_task"):
        benchmark[key] = {k: v for p in parts for k, v in p["benchmark"][key].items()}
    benchmark.update(
        task_ids=task_ids,
        trials_per_task=trials_per_task,
        k_values=normalise_k_values(None, trials_per_task),
        verbose=True,
    )
    for section in ("agent", "environment", "model"):
        base[section]["by_task"] = {
            k: v for p in parts for k, v in p[section]["by_task"].items()
        }
    base["benchmark_run_id"] = run_id
    base.pop("storage", None)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--trials-per-task", type=int, default=5)
    args = parser.parse_args()

    groups: dict[tuple[int, str, str], list[Path]] = defaultdict(list)
    for run_dir in sorted(
        p for p in args.runs_dir.iterdir() if (p / "report.json").exists()
    ):
        meta = json.loads((run_dir / "report.json").read_text())["metadata"]
        runtime = next(iter(meta["agent"]["by_task"].values()))["runtime"]
        level = next(iter(meta["environment"]["by_task"].values()))["runtime"][
            "options"
        ]["level"]
        agent = AGENTS[runtime["name"]]
        model = MODELS[runtime["model"].rsplit("/", 1)[-1]][0]
        groups[(level, agent, model)].append(run_dir)

    for (level, agent, model), run_dirs in sorted(groups.items()):
        loaded = [load_trial(d) for d in run_dirs]
        tasks: dict[str, TaskTrialResults] = {}
        for _, trial, _ in sorted(loaded, key=lambda x: x[1].trial_id):
            tasks.setdefault(
                trial.task_id, TaskTrialResults(trial.task_id)
            ).trials.append(trial)
        counts = {t: len(r.trials) for t, r in tasks.items()}
        if set(counts.values()) != {args.trials_per_task}:
            raise ValueError(f"L{level} {agent}/{model}: trials per task {counts}")

        run_id = f"{args.run_prefix}-l{level}-{agent}-{model}"
        started = min(datetime.fromisoformat(r["created_at"]) for _, _, r in loaded)
        finished = max(datetime.fromisoformat(r["completed_at"]) for _, _, r in loaded)
        result = BenchmarkResult(
            task_results=tasks,
            k=normalise_k_values(None, args.trials_per_task),
            total_duration=(finished - started).total_seconds(),
            verbose=True,
            metadata=merged_metadata(
                run_id, [m for m, _, _ in loaded], args.trials_per_task
            ),
        )
        target = args.out_dir / f"level_{level}" / f"{agent}__{model}"
        target.mkdir(parents=True, exist_ok=True)
        raw = target / "report.json"
        result.generate_report(str(raw))
        with (
            raw.open("rb") as src,
            gzip.open(target / "report.json.gz", "wb", 9) as dst,
        ):
            shutil.copyfileobj(src, dst)
        raw.unlink()
        (target / "provenance.json").write_text(
            json.dumps(
                {
                    "merged_run_id": run_id,
                    "method": "single-trial corral bench runs merged by "
                    "scripts/merge_trial_reports.py; messages and tool calls "
                    "rebuilt from each trial's final state snapshot",
                    "started_at": started.isoformat(),
                    "completed_at": finished.isoformat(),
                    "source_runs": sorted(r["benchmark_run_id"] for _, _, r in loaded),
                },
                indent=2,
            )
            + "\n"
        )
        passes = sum(bool(t.success) for r in tasks.values() for t in r.trials)
        print(
            f"L{level} {agent}__{model}: {len(loaded)} trials, {passes} passes -> {target}"
        )


if __name__ == "__main__":
    main()
