"""Compare the flagged IDs in paired Level 1 and Level 2 Corral reports."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import lzma
from collections import Counter
from pathlib import Path


def load_report(path: Path) -> dict:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    elif path.suffix == ".xz":
        raw = lzma.decompress(raw)
    return json.loads(raw)


def findings(path: Path) -> dict[tuple[str, str], list[dict]]:
    """Return findings keyed by source environment and opaque case ID."""
    report = load_report(path)
    result = {}
    for environment, task in report["task_results"].items():
        trials = task["trials"]
        if len(trials) != 1:
            raise ValueError(f"Expected one audit trial for {environment}; got {len(trials)}")
        answer = trials[0].get("output") or {}
        if not isinstance(answer, dict) or "answer" not in answer:
            raise ValueError(f"No submitted audit report for {environment}")
        final = json.loads(answer["answer"])
        for flag in final["flagged_trials"]:
            key = (environment, flag["trial_id"])
            result.setdefault(key, []).append(flag)
    return result


def compare(identified: Path, masked: Path, output: Path) -> dict:
    known = findings(identified)
    blind = findings(masked)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "environment", "trial_id", "identified_issues", "masked_issues",
            "identified_rerun", "masked_rerun",
        ])
        writer.writeheader()
        for environment, case_id in sorted(known.keys() | blind.keys()):
            a = known.get((environment, case_id), [])
            b = blind.get((environment, case_id), [])
            writer.writerow({
                "environment": environment, "trial_id": case_id,
                "identified_issues": ";".join(
                    f.get("issue_type", f.get("issue", "")) for f in a
                ),
                "masked_issues": ";".join(
                    f.get("issue_type", f.get("issue", "")) for f in b
                ),
                "identified_rerun": ";".join(f.get("rerun", "") for f in a),
                "masked_rerun": ";".join(f.get("rerun", "") for f in b),
            })
    summary = {
        "identified_flagged_ids": len(known),
        "masked_flagged_ids": len(blind),
        "identified_findings": sum(map(len, known.values())),
        "masked_findings": sum(map(len, blind.values())),
        "flagged_in_both": len(known.keys() & blind.keys()),
        "identified_only": len(known.keys() - blind.keys()),
        "masked_only": len(blind.keys() - known.keys()),
        "joined_csv": str(output),
    }
    identified_types = Counter(
        f["issue_type"] for flags in known.values() for f in flags
        if "issue_type" in f
    )
    masked_types = Counter(
        f["issue_type"] for flags in blind.values() for f in flags
        if "issue_type" in f
    )
    if identified_types or masked_types:
        summary["identified_issue_type_counts"] = dict(identified_types)
        summary["masked_issue_type_counts"] = dict(masked_types)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identified", type=Path)
    parser.add_argument("masked", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("audit_comparison.csv"))
    args = parser.parse_args()
    print(json.dumps(compare(args.identified, args.masked, args.csv), indent=2))


if __name__ == "__main__":
    main()
