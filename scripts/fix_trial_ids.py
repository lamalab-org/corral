"""
Fix attempt_ trial IDs in report JSONs by renumbering all trials in each
affected task sequentially (1, 2, 3, ...) in order of appearance.

Usage:
    python fix_trial_ids.py           # dry run, shows what would change
    python fix_trial_ids.py --apply   # apply changes in-place
"""

import argparse
import json
from pathlib import Path

from loguru import logger


def fix_report(path: Path, apply: bool) -> bool:
    data = json.loads(path.read_text())
    task_results = data.get("task_results", {})
    changed = False

    for task_name, task_data in task_results.items():
        if not isinstance(task_data, dict):
            continue
        trials = task_data.get("trials", [])
        if not any("attempt" in str(t.get("trial_id", "")) for t in trials):
            continue

        old_ids = [t.get("trial_id") for t in trials]
        for i, trial in enumerate(trials, start=1):
            trial["trial_id"] = str(i)
        new_ids = [t.get("trial_id") for t in trials]

        logger.info(f"  {task_name}: {old_ids} → {new_ids}")
        changed = True

    if changed and apply:
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"  [SAVED] {path}")

    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true", help="Apply fixes in-place (default: dry run)"
    )
    args = parser.parse_args()

    roots = [Path("reports"), Path("reports_v2")]
    total_files = 0

    for root in roots:
        for report in sorted(root.rglob("*.json")):
            if "agent_logs" in str(report):
                continue
            try:
                data = json.loads(report.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue

            task_results = data.get("task_results", {})
            has_attempt = any(
                "attempt" in str(t.get("trial_id", ""))
                for td in task_results.values()
                if isinstance(td, dict)
                for t in td.get("trials", [])
                if isinstance(t, dict)
            )
            if not has_attempt:
                continue

            logger.info(f"\n{'DRY RUN: ' if not args.apply else ''}Fixing {report}")
            if fix_report(report, apply=args.apply):
                total_files += 1

    logger.info(f"\n{'Applied' if args.apply else 'Would fix'} {total_files} file(s).")
    if not args.apply:
        logger.info("Run with --apply to apply changes.")


if __name__ == "__main__":
    main()
