"""Rescore one restored policy inside its original benchmark Docker image."""

# ruff: noqa: T201 - command-line progress output

from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from inference_opt import datasets
from inference_opt.runner import _archive, private_root, run_policy
from inference_opt.score import final_result
from inference_opt.tools import _policy_hash

from corral.runtime import permissions


def main() -> None:
    job_root = Path("/rescore")
    job = json.loads((job_root / "job.json").read_text())
    output = {"execution_id": job["execution_id"], "started_at": time.time()}
    try:
        task_dir = (
            Path("/opt/corral/tasks/inference_opt/environments")
            / ("level_" + str(job["level"]))
            / "tasks_json"
        )
        task = next(
            value
            for path in sorted(task_dir.glob("*.json"))
            if (value := json.loads(path.read_text()))["id"] == job["task_id"]
        )
        config = task["initial_input"]
        output["config"] = config
        manifest = datasets.load_manifest()
        assert manifest["content_fingerprint"] == job["content_fingerprint"]
        assert manifest["dataset_version"] == job["dataset_version"]
        assert config["final_max_student_calls"] == job["original_budget"]
        assert config["models"] == job["original_models"]
        work_dir = Path("/workspace/task")
        shutil.copytree(job_root / "workspace", work_dir)
        policy_dir = work_dir / job["policy_dir"]
        assert _policy_hash(policy_dir) == job["policy_hash"]
        output["policy_hash"] = job["policy_hash"]

        targets = datasets.load_targets(config["benchmark"], "examples")
        baselines = datasets.load_example_baselines(config["models"])
        revealed = set(job["revealed_ids"])
        examples = []
        for item in datasets.load_items(config["benchmark"], "examples"):
            if item.item_id not in revealed:
                continue
            replies = {}
            for model in config["models"]:
                row = baselines[model].get(item.item_id)
                replies[model] = (
                    None
                    if row is None
                    else {
                        "correct": bool(row["correct"]),
                        "completion": str(row.get("completion", "")),
                        "reasoning": str(row.get("reasoning", "")),
                        "finish_reason": str(row.get("finish_reason", "")),
                    }
                )
            examples.append(
                {
                    **datasets.public_record(item),
                    "target": targets.get(item.item_id, ""),
                    "baseline": replies,
                }
            )

        checkpoints = Path("/corral-state")
        checkpoints.chmod(0o700)
        permissions.configure(checkpoints)
        permissions.workspace_identity(work_dir)
        print(json.dumps({"event": "scoring_started", **output}), flush=True)
        runs = run_policy(
            config=config,
            work_dir=work_dir,
            run_id="rescore-1",
            policy_dir=policy_dir,
            split="test",
            models=config["models"],
            budget_per_model=int(config["final_max_student_calls"]),
            examples=examples,
            write_copy=False,
        )
        failures = collections.Counter(
            call.get("error") or "provider finish_reason=error"
            for run in runs.values()
            for call in run.calls
            if call.get("error") or call.get("finish_reason") == "error"
        )
        output["student_service_errors"] = dict(failures)
        output["models"] = {
            name: {
                "calls_used": run.calls_used,
                "n_items": run.n_items,
                "n_answered": len(run.answers),
                "n_correct": run.n_correct,
                "load_error": run.load_error,
                "run_error": run.error,
                "infrastructure_error": run.infrastructure_error,
                "grading_error": run.grading_error,
            }
            for name, run in runs.items()
        }
        try:
            output["evaluation"] = final_result(config, runs)
        except Exception as exc:
            output["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        invalid = failures or any(
            run.infrastructure_error or run.grading_error for run in runs.values()
        )
        output["status"] = (
            "infrastructure_error"
            if invalid
            else "evaluation_error"
            if output.get("evaluation_error")
            else "completed"
        )
        archive = _archive(private_root(work_dir) / "rescore-1")
        shutil.copyfile(archive, job_root / "private_run.tar.gz")
        output["private_run_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    except Exception as exc:
        output.update(status="controller_error", error=f"{type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        output["ended_at"] = time.time()
        (job_root / "result.json").write_text(json.dumps(output, indent=2) + "\n")
        for path in job_root.rglob("*"):
            if not path.is_symlink():
                os.chown(path, job["host_uid"], job["host_gid"])
        print(json.dumps({"event": "scoring_finished", **output}), flush=True)
    sys.exit(0 if output["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
