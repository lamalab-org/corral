"""Rescoring must retain the full trial denominator and reject service failures."""

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def test_summary_keeps_missing_trials_and_rejects_infrastructure_errors(tmp_path):
    path = Path(__file__).parents[1] / "scripts/rescore_saved_policies.py"
    spec = importlib.util.spec_from_file_location("rescore_saved_policies", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO = tmp_path
    reports = tmp_path / "reports/inference_opt"
    reports.mkdir(parents=True)
    with (reports / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["execution_id", "langfuse_session_id", "score"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "execution_id": f"trial:{i}",
                "langfuse_session_id": "run",
                "score": 1 if i == 0 else 0,
            }
            for i in range(50)
        )
    jobs = []
    for i, status in enumerate(["completed", "infrastructure_error"]):
        directory = tmp_path / f"job-{i}"
        directory.mkdir()
        (directory / "result.json").write_text(
            json.dumps({"status": status, "evaluation": {"score": 1}})
        )
        jobs.append(
            {
                "execution_id": f"trial:{i}",
                "langfuse_session_id": "run",
                "langfuse_trace_id": "trace",
                "task_id": "task",
                "trial_index": i,
                "original_score": 0,
                "policy_hash": "hash",
                "image_digest": "image",
                "job_dir": str(directory),
            }
        )
    args = SimpleNamespace(output=tmp_path, report_dir=reports)
    module.write_summary(args, jobs)
    scores = list(csv.DictReader((reports / "rescore_run_scores.csv").open()))
    assert len(scores) == 1
    assert int(scores[0]["trials"]) == 50
    assert int(scores[0]["completed_rescores"]) == 1
    assert float(scores[0]["original_average_score"]) == 0
    assert float(scores[0]["average_with_completed_rescores"]) == 1 / 50
    trials = list(csv.DictReader((reports / "rescore_summary.csv").open()))
    assert trials[0]["rescored_score"] == "1"
    assert trials[1]["rescored_score"] == ""
