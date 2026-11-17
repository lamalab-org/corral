"""Rescore retained October Stargazer answers with the calibrated complete scorer.

Run with PYTHONPATH=src:tasks/stargazer/src. By default only write an audit;
--write also updates reports and their indexes. Execution evidence is preserved.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

from stargazer.models import load_task
from stargazer.score import EvaluationCriteria, make_stargazer_scorer

ROOT = Path(__file__).resolve().parents[1]
SCORER_COMMIT = "e0626172e2e516050e04413a579a26954ceee7b7"


def update_metrics(data: dict) -> None:
    """Refresh score reductions while retaining durations, tokens and other metrics."""
    for task in data["task_results"].values():
        trials = task["trials"]
        rate = mean(bool(t["success"]) for t in trials)
        task["Task Average Score"] = mean(t["score"] for t in trials)
        task["Task Success Rate"] = rate
        for k in range(1, 6):
            task[f"Task Pass@{k}"] = 1 - (1 - rate) ** k
            task[f"Task Pass^{k}"] = rate**k
    tasks = list(data["task_results"].values())
    trials = [t for task in tasks for t in task["trials"]]
    data["metrics"]["Average Score"] = mean(t["score"] for t in trials)
    data["metrics"]["Overall Success Rate"] = mean(bool(t["success"]) for t in trials)
    for k in range(1, 6):
        for label in (f"Pass@{k}", f"Pass^{k}"):
            data["metrics"][label] = mean(task[f"Task {label}"] for task in tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    criteria = EvaluationCriteria()
    previous = replace(criteria, minimum_match_score=0.8, minimum_planet_score=None)
    rows, reports = [], []
    for path in sorted((ROOT / "reports/stargazer").glob("level_*/*/report.json.gz")):
        original = gzip.decompress(path.read_bytes())
        data = json.loads(original)
        source_hash = hashlib.sha256(original).hexdigest()
        existing = data["metadata"].get("stargazer_rescore")
        if existing:
            raise ValueError(
                f"{path}: already rescored; use the original input reports"
            )
        for task_id, task_result in data["task_results"].items():
            task = load_task(
                ROOT / f"tasks/stargazer/data/synthetic/{task_id}.json",
                source="synthetic",
            )
            for slot, trial in enumerate(task_result["trials"]):
                answer = (trial.get("output") or {}).get("answer")
                before = trial["score"]
                reasons = []
                if answer:
                    old = make_stargazer_scorer(task, previous).evaluate_submission(
                        answer
                    )
                    if old.score != before:
                        raise ValueError(
                            f"old scorer does not reproduce {trial['trial_id']}"
                        )
                    prior_metadata = (trial.get("evaluation") or {}).get("metadata", {})
                    scorer = make_stargazer_scorer(
                        task,
                        criteria,
                        bank_hash=prior_metadata.get("bank_hash", "development"),
                        protocol_version=prior_metadata.get(
                            "interaction_protocol", "stargazer-assisted-v1"
                        ),
                    )
                    result = scorer.evaluate_submission(answer)
                    reasons = result.metadata["failure_reasons"]
                    trial["score"] = result.score
                    trial["success"] = result.score == 1.0
                    prior = trial.get("evaluation")
                    trial["evaluation"] = {
                        "commit_hash": SCORER_COMMIT,
                        "scorer_version": "stargazer-score-v3:complete-mean0.60-floor0.35",
                        "score": result.score,
                        "metrics": {"score": result.score},
                        "feedback": result.feedback,
                        "metadata": {
                            **result.metadata,
                            "previous_evaluation": prior,
                        },
                    }
                rows.append(
                    {
                        "level": int(path.parent.parent.name.removeprefix("level_")),
                        "agent_model": path.parent.name,
                        "task_id": task_id,
                        "trial": slot,
                        "execution_id": trial["trial_id"],
                        "old_score": before,
                        "new_score": trial["score"],
                        "rescored": bool(answer),
                        "failure_reasons": ";".join(reasons)
                        if answer
                        else "no_saved_answer",
                    }
                )
        update_metrics(data)
        data["metadata"]["stargazer_rescore"] = {
            "scorer_commit": SCORER_COMMIT,
            "criteria": asdict(criteria),
            "previous_criteria": asdict(previous),
            "source_report_sha256": source_hash,
            "method": "saved output.answer; missing answers retain original execution outcome",
        }
        reports.append((path, data))
    if len(rows) != 700:
        raise ValueError(f"expected 700 October trials, got {len(rows)}")
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    with (args.audit_dir / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for level in (1, 2):
        group = [row for row in rows if row["level"] == level]
        sys.stdout.write(
            f"L{level}: {sum(r['old_score'] for r in group):g} -> "
            f"{sum(r['new_score'] for r in group):g}/{len(group)}; "
            f"{sum(r['rescored'] for r in group)} answers rescored\n"
        )
    if args.write:
        for path, data in reports:
            raw = (
                json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            ).encode()
            path.write_bytes(gzip.compress(raw, mtime=0))
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "reports/build_index.py"),
                str(ROOT / "reports/stargazer"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
