"""Select the 60 hardest frozen questions per benchmark using IRT estimates.

Reads item difficulty estimates from the sibling ``bp`` checkout, preserves
question text and private labels, and makes a deterministic balanced 30/30
train/test split. Higher 2PL difficulty means harder items.

Usage::

    uv run python scripts/select_hard_questions.py \
        --irt-root /Users/n0w0f/git/n0w0f_2027/bp/results
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "inference_opt" / "data" / "frozen" / "v1"
BENCHMARKS = ("chembench", "bbh", "gpqa_diamond", "gsm8k", "mmlu_pro", "arc_challenge")
SEED = 20260930
N_SELECTED = 60


def _irt_file(irt_root: Path, benchmark: str) -> tuple[Path, str]:
    candidates = [
        (irt_root / "irt" / benchmark / "items.parquet", "difficulty"),
        (irt_root / "research_irt" / benchmark / "items.parquet", "difficulty_map"),
    ]
    for path, column in candidates:
        if path.is_file():
            return path, column
    raise FileNotFoundError(f"No IRT item estimates found for {benchmark} under {irt_root}")


def _split_hash(item_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{item_id}".encode()).hexdigest()


def _choose_split(records: list[dict[str, Any]]) -> dict[str, str]:
    """Assign exactly half to each split, balancing category and difficulty."""
    ordered = sorted(records, key=lambda row: (-row["irt_difficulty"], row["item_id"]))
    counts: dict[tuple[str, int], Counter[str]] = {}
    split_counts: Counter[str] = Counter()
    assignments: dict[str, str] = {}
    for row in ordered:
        category = str(row.get("category") or "__none__")
        # Assign in pairs within each category/difficulty order, while a global
        # count guard ensures a 30/30 split even when category counts are odd.
        key = (category, 0)
        bucket = counts.setdefault(key, Counter())
        train_n, test_n = split_counts["train"], split_counts["test"]
        if train_n >= N_SELECTED // 2:
            split = "test"
        elif test_n >= N_SELECTED // 2:
            split = "train"
        elif bucket["train"] < bucket["test"]:
            split = "train"
        elif bucket["test"] < bucket["train"]:
            split = "test"
        else:
            split = "train" if int(_split_hash(row["item_id"]), 16) % 2 == 0 else "test"
        assignments[row["item_id"]] = split
        bucket[split] += 1
        split_counts[split] += 1
    if split_counts != Counter({"train": 30, "test": 30}):
        raise RuntimeError(f"Expected 30/30 split, got {dict(split_counts)}")
    return assignments


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in records
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--irt-root", type=Path, required=True,
                        help="bp results directory containing irt/ and research_irt/")
    args = parser.parse_args()

    source_dir = DATA / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    for benchmark in BENCHMARKS:
        public_path = DATA / "public" / f"{benchmark}.jsonl"
        pool_path = source_dir / f"{benchmark}.jsonl"
        if not pool_path.exists():
            pool_path.write_text(public_path.read_text(encoding="utf-8"), encoding="utf-8")

    labels_path = DATA / "private" / "labels.jsonl"
    labels = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line]
    labels_by_id = {row["item_id"]: row for row in labels}
    selected_all: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": "v1",
        "seed": SEED,
        "benchmarks": list(BENCHMARKS),
        "items_per_benchmark": N_SELECTED,
        "train_per_benchmark": N_SELECTED // 2,
        "test_per_benchmark": N_SELECTED // 2,
        "selection": {
            "rule": "top 60 items by IRT difficulty",
            "difficulty_direction": "higher difficulty is harder",
            "split": "deterministic 30/30 split balanced by category",
            "stratified_by": "category; difficulty rank within category",
            "source": "bp benchmark IRT item estimates",
        },
        "per_benchmark": {},
    }

    for benchmark in BENCHMARKS:
        public_path = DATA / "public" / f"{benchmark}.jsonl"
        pool_path = DATA / "source" / f"{benchmark}.jsonl"
        source = [json.loads(line) for line in pool_path.read_text(encoding="utf-8").splitlines() if line]
        path, difficulty_column = _irt_file(args.irt_root, benchmark)
        estimates = pd.read_parquet(path, columns=["item_id", difficulty_column])
        difficulty = {
            str(row.item_id): float(getattr(row, difficulty_column))
            for row in estimates.itertuples(index=False)
            if not pd.isna(getattr(row, difficulty_column))
        }
        candidates = [row for row in source if row["item_id"] in difficulty]
        candidates.sort(key=lambda row: (-difficulty[row["item_id"]], row["item_id"]))
        selected = candidates[:N_SELECTED]
        if len(selected) != N_SELECTED:
            raise RuntimeError(f"{benchmark}: only {len(selected)} items have IRT estimates")
        for row in selected:
            row["irt_difficulty"] = difficulty[row["item_id"]]
        split_map = _choose_split(selected)
        selected.sort(key=lambda row: (split_map[row["item_id"]], -row["irt_difficulty"], row["item_id"]))
        for row in selected:
            row["split"] = split_map[row["item_id"]]
            row.pop("irt_difficulty")
        selected_all.extend(selected)

        train_ids = {row["item_id"] for row in selected if row["split"] == "train"}
        selected_difficulties = [difficulty[row["item_id"]] for row in selected]
        cats = {str(row.get("category") or "__none__") for row in selected}
        manifest["per_benchmark"][benchmark] = {
            "irt_source": str(path),
            "difficulty_column": difficulty_column,
            "irt_difficulty_cutoff": min(selected_difficulties),
            "selected": len(selected),
            "train": len(train_ids),
            "test": len(selected) - len(train_ids),
            "categories": len(cats),
            "train_mean_irt_difficulty": sum(difficulty[x] for x in train_ids) / len(train_ids),
            "test_mean_irt_difficulty": sum(difficulty[row["item_id"]] for row in selected if row["split"] == "test") / (len(selected) - len(train_ids)),
            "max_irt_difficulty": max(selected_difficulties),
        }

    # Ensure every selected public item still has exactly one private target.
    for row in selected_all:
        if row["item_id"] not in labels_by_id:
            raise RuntimeError(f"missing private label for {row['item_id']}")
    kept_ids = {row["item_id"] for row in selected_all}
    labels = [row for row in labels if row["item_id"] in kept_ids]

    for benchmark in BENCHMARKS:
        rows = [row for row in selected_all if row["benchmark"] == benchmark]
        (DATA / "public" / f"{benchmark}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in labels),
        encoding="utf-8",
    )
    frame = pd.DataFrame(selected_all)
    frame.to_parquet(DATA / "items.parquet", index=False)
    manifest["content_fingerprint"] = _fingerprint(selected_all)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for benchmark, stats in manifest["per_benchmark"].items():
        print(f"{benchmark}: cutoff={stats['irt_difficulty_cutoff']:.3f}, train/test={stats['train']}/{stats['test']}")


if __name__ == "__main__":
    main()
