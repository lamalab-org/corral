"""Remove one category from a benchmark's train and test splits.

Each removed question is replaced, in the same split, by the unused bp question
from another category, inside the benchmark's IRT range, whose difficulty is
closest to it, so the halves stay
matched. Updates the labels, ``items.parquet``, the manifest's per-benchmark
statistics and ``content_fingerprint``. Re-run ``select_examples.py`` afterwards
so the examples follow the new train mix, and re-measure the benchmark's baselines.

    uv run python scripts/drop_category.py --bp-root /path/to/bp \\
        --benchmark chembench --category chemical_preference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from frozen_set import content_fingerprint, jsonl_text, read_jsonl, reference_accuracy
from inference_opt import datasets
from select_examples import _candidates, _hf_categories, _irt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bp-root", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    data = datasets.data_root()
    manifest = datasets.load_manifest()
    public_path = data / "public" / f"{args.benchmark}.jsonl"
    public = read_jsonl(public_path)
    dropped = [r for r in public if r["split"] != "examples" and r["category"] == args.category]
    if not dropped:
        raise SystemExit(f"{args.benchmark} has no {args.category!r} questions in train or test")

    candidates = [
        row for row in _candidates(
            args.bp_root, args.benchmark,
            {r["item_id"] for r in public},
            {r["question_hash"] for r in public if r.get("question_hash")},
            _hf_categories(),
        )
        if row["record"]["category"] != args.category
        and row["irt"] >= manifest["per_benchmark"][args.benchmark]["irt_difficulty_cutoff"]
    ]
    difficulty = dict(zip(*(_irt(args.bp_root, args.benchmark)[c] for c in ("item_id", "irt")),
                          strict=True))

    replacements: dict[str, dict] = {}
    for record in dropped:
        target = difficulty[record["item_id"]]
        best = min(candidates, key=lambda row: (abs(row["irt"] - target), row["record"]["item_id"]))
        candidates.remove(best)
        replacements[record["item_id"]] = {**best, "split": record["split"]}
    reference = reference_accuracy(
        args.bp_root, args.benchmark,
        {row["record"]["item_id"] for row in replacements.values()},
    )

    labels_path = data / "private" / "labels.jsonl"
    labels = read_jsonl(labels_path)
    new_public, new_labels = [], []
    for record in public:
        row = replacements.get(record["item_id"])
        new_public.append(record if row is None else {
            **row["record"], "split": row["split"],
            "reference_accuracy": reference.get(row["record"]["item_id"]),
        })
    for label in labels:
        row = replacements.get(label["item_id"])
        new_labels.append(label if row is None else {
            "answer_format": row["record"]["answer_format"],
            "item_id": row["record"]["item_id"],
            "split": row["split"],
            "target": row["answer"],
        })
    public_path.write_text(jsonl_text(new_public), encoding="utf-8")
    labels_path.write_text(jsonl_text(new_labels), encoding="utf-8")

    parquet = data / "items.parquet"
    frame = pd.read_parquet(parquet)
    frame = frame[frame.benchmark != args.benchmark]
    frame = pd.concat([frame, pd.DataFrame(new_public)], ignore_index=True)
    frame.to_parquet(parquet, index=False)

    scored = [r for r in new_public if r["split"] != "examples"]
    entry = manifest["per_benchmark"][args.benchmark]
    for split in ("train", "test"):
        values = [difficulty[r["item_id"]] for r in scored if r["split"] == split]
        entry[f"{split}_mean_irt_difficulty"] = sum(values) / len(values)
    entry["categories"] = len({r["category"] for r in scored})
    entry["irt_difficulty_cutoff"] = min(difficulty[r["item_id"]] for r in scored)
    entry["max_irt_difficulty"] = max(difficulty[r["item_id"]] for r in scored)
    entry["dropped_categories"] = sorted({*entry.get("dropped_categories", []), args.category})
    manifest["content_fingerprint"] = content_fingerprint(data, manifest["benchmarks"])
    (data / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for old, row in replacements.items():
        print(f"{row['split']:5} {old} -> {row['record']['item_id']} "
              f"({row['record']['category']}, irt {row['irt']:.2f} vs {difficulty[old]:.2f})")


if __name__ == "__main__":
    main()
