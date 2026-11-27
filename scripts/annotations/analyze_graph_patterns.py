"""Derive inspectable pattern artifacts from completed graphs without model calls."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Run snapshots may place this file directly under scripts/.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "annotations" else SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from corral.epistemic_analysis.graph import validate_support_quotes
from corral.epistemic_analysis.pattern_review import (
    PATTERN_REVIEW_VERSION,
    analyze_patterns,
)
from corral.epistemic_analysis.storage import atomic_save
from corral.epistemic_analysis.viewer.graph_run import GraphRunStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    store = GraphRunStore(args.root)
    reviews = json.loads(args.reviews.read_text()) if args.reviews else {}
    parent_hits, repeated_parents, records = {}, set(), []
    for identity, job in store.rows.items():
        if store.completed(identity) is None:
            continue
        record, graph = store.graph(identity)
        result = analyze_patterns(graph, record["graph_sha256"])
        for review in reviews.get(identity, []):
            if review["graph_sha256"] != record["graph_sha256"]:
                raise ValueError("Review belongs to a different graph revision")
            if (
                not review.get("support")
                or not validate_support_quotes(graph["messages"], review["support"])[0]
            ):
                raise ValueError("Review requires exact original source quotes")
            result["reviews"].append(review)
        atomic_save(args.root / "pattern-analysis" / f"{identity}.json", result)
        parent = job.get("parent_id") or identity
        parent_hits.setdefault(parent, set()).update(
            p["name"] for p in result["patterns"] if p["status"] == "matched"
        )
        if result["repeated_tests"]:
            repeated_parents.add(parent)
        records.append(
            {
                "id": identity,
                "parent_id": parent,
                "graph_sha256": record["graph_sha256"],
                "repeated_groups": len(result["repeated_tests"]),
                "review_count": len(result["reviews"]),
            }
        )
    summary = {
        "version": PATTERN_REVIEW_VERSION,
        "units": len(records),
        "parent_traces": len(parent_hits),
        "parent_counts": dict(
            Counter(name for names in parent_hits.values() for name in names)
        ),
        "repeated_test_parent_traces": len(repeated_parents),
        "records": records,
    }
    atomic_save(args.root / "pattern-analysis" / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}))  # noqa: T201


if __name__ == "__main__":
    main()
