"""Prepare reproducible runs and compare blind final answers offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sqlite3
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from stargazer.models import load_task
from stargazer.protocol import (
    ASSISTED_PROTOCOL,
    BLIND_PROTOCOL,
    execution_fingerprint,
    protocol_version,
)
from stargazer.score import LEGACY_CRITERIA, EvaluationCriteria, make_stargazer_scorer


def provenance() -> dict:
    root = Path(__file__).resolve().parents[4]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    paths = sorted((root / "src" / "corral").rglob("*.py")) + sorted(
        (root / "tasks" / "stargazer" / "src").rglob("*.py")
    )
    paths += sorted((root / "tasks" / "stargazer" / "src").rglob("*.md"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {
        "code_revision": revision or "unavailable",
        "source_hash": digest.hexdigest(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pydantic", "rebound", "cloudpickle")
        },
    }


def bank_identity(data_root: str | Path) -> str:
    """Use the same verified bank identity as the execution environment."""
    data_root = Path(data_root)
    from stargazer.env import DATA_ROOT

    if data_root.resolve() == DATA_ROOT.resolve():
        files = sorted(data_root.glob("synthetic/*.json"))
        return hashlib.sha256(
            b"".join(path.name.encode() + path.read_bytes() for path in files)
        ).hexdigest()
    from stargazer.bank import verify_bank

    return verify_bank(data_root, purpose="evaluation")["bank_hash"]


def compare_run(run_root: str | Path, data_root: str | Path) -> dict:
    """Grade only committed main-branch final JSON, never validation history."""
    root = Path(run_root)
    data_root = Path(data_root)
    bank_hash = bank_identity(data_root)
    records = []
    for database in sorted(root.rglob("commits.sqlite3")):
        with sqlite3.connect(
            database.resolve().as_uri() + "?mode=ro", uri=True
        ) as connection:
            # Startup can fail before SQLite creates the commit schema.
            has_commits = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='commits'"
            ).fetchone()
            rows = (
                connection.execute(
                    "SELECT execution_id, event_type,event_json,occurred_at FROM commits WHERE branch_id='main' ORDER BY sequence"
                ).fetchall()
                if has_commits
                else []
            )
        if not rows:
            metadata_path = database.with_name("metadata.json")
            metadata = (
                json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            )
            records.append(
                {
                    "execution_id": metadata.get("execution_id"),
                    "task_id": metadata.get("task_id"),
                    "execution_version": None,
                    "execution_provenance": {},
                    "validation_calls": 0,
                    "usage": {},
                    "model": {},
                    "budget": None,
                    "status": "unfinished",
                    "termination": "no_commits",
                    "error": "Execution has no committed events",
                }
            )
            continue
        if len({row[0] for row in rows}) != 1:
            raise ValueError("Expected one execution per commit database")
        events = [(kind, json.loads(value)) for _, kind, value, _ in rows]
        start = next(event for kind, event in events if kind == "execution.started")
        task_id = start["task"]["id"]
        if start["task"].get("stargazer", {}).get("bank_hash") != bank_hash:
            raise ValueError("Committed bank hash differs from supplied task bank")
        version = start["task"].get("execution_version", "")
        protocol = version.split(":", 1)[0]
        if protocol not in {BLIND_PROTOCOL, ASSISTED_PROTOCOL}:
            raise ValueError(
                "Offline comparison requires fresh blind-protocol executions"
            )
        accepted = [event for kind, event in events if kind == "submission.accepted"]
        if len(accepted) > 1:
            raise ValueError("Multiple final submissions in a main branch")
        usage = {
            key: sum(event.get("usage_delta", {}).get(key, 0) for _, event in events)
            for key in (
                "llm_calls",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
            )
        }
        record = {
            "execution_id": rows[0][0],
            "task_id": task_id,
            "execution_version": version,
            "protocol": protocol,
            "arm": "B" if protocol == ASSISTED_PROTOCOL else "A",
            "execution_provenance": start["task"]["stargazer"],
            "validation_calls": sum(
                kind == "tool.started" and event["tool_name"] == "validate_fit"
                for kind, event in events
            ),
            "tool_calls": sum(kind == "tool.started" for kind, _ in events),
            "wall_seconds": (
                datetime.fromisoformat(rows[-1][3]) - datetime.fromisoformat(rows[0][3])
            ).total_seconds(),
            "usage": usage,
            "model": start.get("model", {}),
            "budget": start.get("scaffold", {}).get("max_iterations"),
            "status": "unfinished",
            "termination": next(
                (
                    event.get("metadata", {}).get("agent_status")
                    or event.get("error_type")
                    or event.get("status")
                    for kind, event in reversed(events)
                    if kind in {"execution.failed", "execution.completed"}
                ),
                "unrecorded",
            ),
            "error": next(
                (
                    event.get("error")
                    for kind, event in reversed(events)
                    if kind == "execution.failed"
                ),
                None,
            ),
        }
        if accepted and not accepted[0].get("surrendered"):
            task = load_task(data_root / "synthetic" / f"{task_id}.json")
            answer = accepted[0]["answer"]
            results = {
                name: make_stargazer_scorer(
                    task, criteria, bank_hash=bank_hash
                ).evaluate_submission(answer)
                for name, criteria in [
                    ("legacy", LEGACY_CRITERIA),
                    ("complete", EvaluationCriteria()),
                ]
            }
            scientific = results["complete"].metadata.get("evaluation", {})
            assignment = (
                scientific.get("metrics", {}).get("matching", {}).get("assignment", {})
            )
            record.update(
                status="submitted",
                scores={name: result.score for name, result in results.items()},
                rms_ms=scientific.get("rms_ms"),
                complete_matching=scientific.get("ok_complete_matching"),
                matched_pairs=len(assignment["pairs"]) if assignment else None,
                false_positives=len(assignment["unmatched_guess"])
                if assignment
                else None,
                missed_planets=len(assignment["unmatched_truth"])
                if assignment
                else None,
                per_planet_errors=scientific.get("comparison", {}).get("matched_pairs"),
                delta_bic_per_point=scientific.get("delta_bic_per_point"),
                match_score=scientific.get("match_score"),
                bic=scientific.get("metrics", {}).get("bic"),
                failure_reasons=results["complete"].metadata["failure_reasons"],
                withheld_prediction_rms_ms=results["complete"].metadata.get(
                    "withheld_prediction_rms_ms"
                ),
            )
        elif accepted:
            record["status"] = "surrendered"
        records.append(record)
    completed = sum(record["status"] == "submitted" for record in records)
    return {
        "provenance": provenance(),
        "bank_hash": bank_hash,
        "denominator_policy": "all recorded trials, including unfinished, cancelled and surrendered",
        "total": len(records),
        "submitted": completed,
        "unfinished_or_surrendered": len(records) - completed,
        "errors": sum(record["error"] is not None for record in records),
        "termination_counts": dict(
            Counter(record["termination"] for record in records)
        ),
        "completion_rate": completed / len(records) if records else None,
        "passes": {
            name: sum(row.get("scores", {}).get(name, 0) for row in records)
            for name in ("legacy", "complete")
        },
        "arms": {
            arm: {
                "total": sum(row.get("arm") == arm for row in records),
                "submitted": sum(
                    row.get("arm") == arm and row["status"] == "submitted"
                    for row in records
                ),
                "complete_passes": sum(
                    row.get("scores", {}).get("complete", 0)
                    for row in records
                    if row.get("arm") == arm
                ),
            }
            for arm in ("A", "B")
        },
        "records": records,
    }


def preflight(data_root: Path, *, concurrency: int = 2) -> dict:
    """Exercise concurrent restricted workers; never invoke a model or grader.

    Run inside a prepared Docker image with the benchmark's capabilities. The
    container cgroup peak includes controller and workers, not just Python heaps.
    """
    from corral.runtime import permissions
    from stargazer.docker import execute_analysis

    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    contexts = [
        load_task(path).public_fit_context()
        for path in sorted((data_root / "synthetic").glob("*.json"))
    ]
    if not contexts:
        raise ValueError("No tasks for preflight")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="stargazer-preflight-") as private:
        if not permissions.enabled():
            permissions.configure(Path(private))

        def run(index):
            context = contexts[index % len(contexts)]
            data = {
                **asdict(context.observations),
                "star_mass_sun": context.star_mass_sun,
                "los_axis": context.los_axis,
                "integrator_preference": context.integrator_preference,
            }
            with tempfile.TemporaryDirectory(
                prefix="preflight-", dir="/workspace"
            ) as workspace:
                first = execute_analysis(
                    code="from scipy.optimize import least_squares\nd = stargazer_diagnostics([])\ncurve = stargazer_predict([])\nprint(d['valid'])",
                    public_data=data,
                    checkpoint=None,
                    workspace=workspace,
                )
                second = execute_analysis(
                    code="print(np.array_equal(curve, stargazer_predict([])))\nprint(d['valid'])",
                    public_data=data,
                    checkpoint=first["checkpoint"],
                    workspace=workspace,
                )
            if (
                first["output"].strip() != "True"
                or second["output"].strip() != "True\nTrue"
            ):
                raise RuntimeError(
                    f"Preflight worker {index} failed: {first['output']} {second['output']}"
                )
            return {"worker": index, "completed": True}

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            workers = list(pool.map(run, range(concurrency)))
    cgroup = Path("/sys/fs/cgroup")
    peak = cgroup / "memory.peak"
    if not peak.is_file():
        raise RuntimeError("Container peak memory unavailable; require cgroup v2")
    return {
        "workers": workers,
        "concurrency": concurrency,
        "wall_seconds": time.monotonic() - started,
        "container_peak_memory_bytes": int(peak.read_text()),
        "container_memory_limit": (cgroup / "memory.max").read_text().strip(),
        "container_cpu_limit": (cgroup / "cpu.max").read_text().strip(),
        "provenance": provenance(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "compare", "preflight"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--image-digest")
    parser.add_argument("--scorer", choices=["legacy", "complete"], default="complete")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--memory", default="7g")
    parser.add_argument("--cpus", type=float, default=2)
    parser.add_argument("--pids-limit", type=int, default=512)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.data_root, concurrency=args.concurrency)
        result["image_digest"] = args.image_digest
    elif args.command == "compare":
        if args.run_root is None:
            parser.error("compare requires --run-root")
        result = compare_run(args.run_root, args.data_root)
    else:
        if not args.model or not args.image_digest:
            parser.error("prepare requires --model and --image-digest")
        files = sorted((args.data_root / "synthetic").glob("*.json"))
        if (
            min(
                args.concurrency,
                args.iterations,
                args.trials,
                args.cpus,
                args.pids_limit,
            )
            <= 0
        ):
            parser.error("resource and trial budgets must be positive")
        code = provenance()
        bank = bank_identity(args.data_root)
        criteria = LEGACY_CRITERIA if args.scorer == "legacy" else EvaluationCriteria()
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "interaction_protocol": ASSISTED_PROTOCOL,
            "bank_hash": bank,
            "primary_scorer": args.scorer,
            "provenance": code,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_iterations": args.iterations,
            "trials_per_task": args.trials,
            "image_digest": args.image_digest,
            "resources": {
                "concurrency": args.concurrency,
                "memory": args.memory,
                "cpus": args.cpus,
                "pids_limit": args.pids_limit,
            },
            "launch_order": [path.stem for path in files],
            "rerun_policy": "retain all original attempts; rerun infrastructure failures separately under identical settings",
            "arms": {
                arm: {
                    "analysis_assistance": assisted,
                    "protocol": protocol_version(assisted),
                    "execution_version": execution_fingerprint(
                        protocol=protocol_version(assisted),
                        bank=bank,
                        criteria=asdict(criteria),
                        development_mode=False,
                        source_hash=code["source_hash"],
                    ),
                }
                for arm, assisted in (("A", False), ("B", True))
            },
            "criteria": {
                "legacy": asdict(LEGACY_CRITERIA),
                "complete": asdict(EvaluationCriteria()),
            },
            "task_hashes": {
                path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files
            },
            "effective_observation_hashes": {
                path.stem: hashlib.sha256(
                    json.dumps(
                        asdict(load_task(path).public_fit_context()),
                        sort_keys=True,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                for path in files
            },
        }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
