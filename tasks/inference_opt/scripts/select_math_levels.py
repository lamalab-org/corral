"""Build the math-only question set (``v2``) from MATH Level 5 questions.

One MATH subject per benchmark:

    math_algebra, math_prealgebra, math_number_theory,
    math_counting_probability, math_geometry

Answers are graded exactly as inspect_evals' MATH eval grades them
(``expression_exact_match_sympy``), so answers may be numbers or expressions.
A question is kept only if that scorer grades its own gold answer as correct:
it cannot match matrices, coordinate pairs or intervals even to themselves, so
on those every reply would be wrong. Every question is labelled
``answer_format: "math"``, so its metadata never says whether the answer is a
number or an expression.

bp does not store MATH's difficulty level, so it is recovered by matching each
question to the original test set (``EleutherAI/hendrycks_math``). Per subject,
80 questions are drawn in a fixed pseudo-random order, sorted by bp's IRT
difficulty and dealt in blocks of eight, forward then backward, to train (30),
test (30) and examples (20), so the three splits are equally hard.

    uv run python scripts/select_math_levels.py --bp-root /path/to/bp

Run ``measure_baselines.py`` afterwards for train and test, and again with
``--splits examples``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_set import (  # noqa: E402
    content_fingerprint,
    jsonl_text,
    rank,
    reference_accuracy,
)

VERSION = "v2"
LEVEL = "Level 5"
ANSWER_FORMAT = "math"
SIZES = {"train": 30, "test": 30, "examples": 20}
#: One block of eight, dealt forward; the next block is dealt backward.
BLOCK = ("train", "test", "examples", "train", "test", "examples", "train", "test")
SEED = 20261005
SOURCE = "EleutherAI/hendrycks_math"

#: Benchmark name -> MATH subject.
SUBSETS: dict[str, str] = {
    "math_algebra": "algebra",
    "math_prealgebra": "prealgebra",
    "math_number_theory": "number theory",
    "math_counting_probability": "counting & probability",
    "math_geometry": "geometry",
}


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _levels() -> dict[str, str]:
    """MATH's own difficulty level for every test question, keyed by its text."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [
        f for f in list_repo_files(SOURCE, repo_type="dataset")
        if "test" in f and f.endswith(".parquet")
    ]
    source = pd.concat(
        pd.read_parquet(hf_hub_download(SOURCE, f, repo_type="dataset")) for f in files
    )
    return dict(zip(source.problem.map(_key), source.level, strict=True))


def _gradable(targets: list[str]) -> list[bool]:
    """Whether inspect_evals' MATH scorer grades each gold answer as matching itself."""
    from inspect_evals.math.utils import is_equiv_sympy, normalize_final_answer

    async def check(target: str) -> bool:
        normalized = await normalize_final_answer(target)
        return bool(await is_equiv_sympy(normalized, normalized))

    async def check_all() -> list[bool]:
        return [await check(target) for target in targets]

    return asyncio.run(check_all())


def _pool(bp: Path) -> pd.DataFrame:
    """Gradable Level 5 MATH questions, with subject and IRT difficulty."""
    items = pd.read_parquet(bp / "datasets" / "math" / "items.parquet")
    items = items[(items.reconciliation_confidence == "high")
                  & (items.target_conflict.astype(str) == "False")]
    irt = pd.read_parquet(bp / "results" / "irt" / "math" / "items.parquet")
    frame = items.merge(irt[["item_id", "difficulty", "category"]], on="item_id")
    frame["level"] = frame.question.map(_key).map(_levels())
    if frame.level.isna().any():
        raise RuntimeError(f"{int(frame.level.isna().sum())} bp questions not found in {SOURCE}")
    frame = frame[frame.level == LEVEL].drop_duplicates("question_hash")
    return frame[_gradable(frame.answer.astype(str).tolist())]


def _deal(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Sort by difficulty and deal blocks of eight, forward then backward."""
    ordered = sorted(rows, key=lambda r: (r["irt"], r["record"]["item_id"]))
    dealt: dict[str, list[dict[str, Any]]] = {split: [] for split in SIZES}
    for start in range(0, len(ordered), len(BLOCK)):
        turn = BLOCK if start // len(BLOCK) % 2 == 0 else BLOCK[::-1]
        for split, row in zip(turn, ordered[start:start + len(BLOCK)], strict=False):
            dealt[split].append(row)
    return dealt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bp-root", type=Path, required=True)
    args = parser.parse_args()

    pool = _pool(args.bp_root)
    data = ROOT / "inference_opt" / "data" / "frozen" / VERSION
    for old in (data / "public").glob("*.jsonl"):
        old.unlink()
    (data / "public").mkdir(parents=True, exist_ok=True)
    (data / "private").mkdir(parents=True, exist_ok=True)

    total = sum(SIZES.values())
    labels: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    per_benchmark: dict[str, Any] = {}
    examples_payload: list[dict[str, Any]] = []
    for benchmark, subject in SUBSETS.items():
        candidates = pool[pool.category == subject]
        if len(candidates) < total:
            raise RuntimeError(f"{subject}: {len(candidates)} questions, need {total}")
        chosen = sorted(candidates.itertuples(index=False),
                        key=lambda r: rank(SEED, r.item_id))[:total]
        rows = []
        for row in chosen:
            sample_id = str(row.inspect_item_id)
            rows.append({
                "irt": float(row.difficulty),
                "answer": str(row.answer),
                "record": {
                    "answer_format": ANSWER_FORMAT, "benchmark": benchmark,
                    "category": subject, "item_id": f"{benchmark}:{sample_id}",
                    "question": str(row.question), "question_hash": row.question_hash,
                    "sample_id": sample_id, "subcategory": LEVEL.lower(),
                    "bp_item_id": row.item_id,
                },
            })
        dealt = _deal(rows)
        assert {split: len(part) for split, part in dealt.items()} == SIZES

        reference = reference_accuracy(
            args.bp_root, "math",
            {r["record"]["bp_item_id"] for part in dealt.values() for r in part})
        public = []
        for split in SIZES:
            for row in sorted(dealt[split], key=lambda r: (-r["irt"], r["record"]["item_id"])):
                record = {**row["record"], "split": split}
                bp_item_id = record.pop("bp_item_id")
                record["reference_accuracy"] = reference.get(bp_item_id)
                public.append(record)
                labels.append({"answer_format": ANSWER_FORMAT, "item_id": record["item_id"],
                               "split": split, "target": row["answer"]})
                if split == "examples":
                    examples_payload.append(record)
        (data / "public" / f"{benchmark}.jsonl").write_text(jsonl_text(public), encoding="utf-8")
        frames.append(pd.DataFrame(public))

        stats = {split: [r["irt"] for r in dealt[split]] for split in SIZES}
        per_benchmark[benchmark] = {
            "categories": 1,
            "difficulty_column": "difficulty",
            "level": LEVEL,
            "subject": subject,
            "pool": len(candidates),
            "selected": SIZES["train"] + SIZES["test"],
            "irt_difficulty_cutoff": min(min(v) for v in stats.values()),
            "max_irt_difficulty": max(max(v) for v in stats.values()),
            **SIZES,
            **{f"{split}_mean_irt_difficulty": sum(v) / len(v) for split, v in stats.items()},
        }
        print(f"{benchmark:26} pool {len(candidates):3}  " + "  ".join(
            f"{split} {len(dealt[split])} (irt {sum(stats[split]) / len(stats[split]):.2f})"
            for split in SIZES))

    (data / "private" / "labels.jsonl").write_text(jsonl_text(labels), encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_parquet(data / "items.parquet", index=False)

    benchmarks = list(SUBSETS)
    examples_json = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in examples_payload
    )
    manifest = {
        "benchmarks": benchmarks,
        "dataset_version": VERSION,
        "schema_version": 1,
        "seed": SEED,
        "items_per_benchmark": SIZES["train"] + SIZES["test"],
        "train_per_benchmark": SIZES["train"],
        "test_per_benchmark": SIZES["test"],
        "examples_per_benchmark": SIZES["examples"],
        "per_benchmark": per_benchmark,
        "selection": {
            "rule": (f"{total} items per benchmark, one MATH subject each, all {LEVEL}; "
                     "kept only if inspect_evals' MATH scorer grades the gold answer as "
                     "matching itself"),
            "answer_format": "every question is 'math'; graded by inspect_evals' "
                             "expression_exact_match_sympy",
            "level_source": f"MATH difficulty level matched from {SOURCE}",
            "split": "sorted by bp IRT difficulty and dealt in blocks of eight "
                     "(train, test, examples), forward then backward",
            "examples": "revealed to the agent, never scored",
            "reference_accuracy": "mean correctness of a 7-8B reference model cohort",
        },
        "content_fingerprint": content_fingerprint(data, benchmarks),
        "examples_fingerprint": hashlib.sha256(examples_json.encode()).hexdigest(),
    }
    (data / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {data}")


if __name__ == "__main__":
    main()
