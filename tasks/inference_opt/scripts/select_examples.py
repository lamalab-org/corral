"""Add an ``examples`` split to the frozen question set.

The agent reveals labelled questions from this split, and policies receive them
as ``ctx.examples``. They are separate from train, so revealing an answer never
shows the gold answer of a question that ``evaluate_candidate`` scores.

For each benchmark, picks 30 bp items that are not already in the set, with the
same category counts as train, from the same IRT difficulty range (at or above
the benchmark's ``irt_difficulty_cutoff``) where it has enough of a category.
Train and test are left untouched, as is ``content_fingerprint``, which covers
the scored questions.

    uv run python scripts/select_examples.py --bp-root /path/to/bp

Run ``measure_baselines.py --splits examples`` afterwards to record what each
student answers zero-shot on the new questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_opt import datasets  # noqa: E402

SPLIT = "examples"
N_EXAMPLES = 30
SEED = 20261005
NUMERIC = re.compile(r"-?\d+(\.\d+)?")
COHORT = re.compile(r"(?<![\d.])[78](?:\.\d+)?b(?![a-z])", re.IGNORECASE)


def _irt(bp: Path, benchmark: str) -> pd.DataFrame:
    """The bp difficulty estimates the set was selected with."""
    path = bp / "results" / "irt" / benchmark / "items.parquet"
    if path.is_file():
        return pd.read_parquet(path).rename(columns={"difficulty": "irt"})
    frame = pd.read_parquet(bp / "results" / "research_irt" / benchmark / "items.parquet")
    return frame[~frame.saturated].rename(columns={"difficulty_map": "irt"})


def _hf_categories() -> dict[str, dict[str, tuple[str | None, str | None]]]:
    """(category, subcategory) for the benchmarks whose bp tables lack them."""
    from huggingface_hub import hf_hub_download

    mmlu = pd.read_parquet(hf_hub_download(
        "TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    gpqa = pd.read_csv(hf_hub_download("Idavidrein/gpqa", "gpqa_diamond.csv", repo_type="dataset"))
    return {
        "mmlu_pro": {str(q): (c, None) for q, c in zip(mmlu.question_id, mmlu.category, strict=True)},
        "gpqa_diamond": {
            r: (d, s) for r, d, s in zip(
                gpqa["Record ID"], gpqa["High-level domain"], gpqa["Subdomain"], strict=True)
        },
    }


def _answer_format(benchmark: str, options: list[str] | None, answer: str) -> str | None:
    if options:
        # ChemBench, like its own harness, asks for every correct option on every
        # multiple-choice question, so whether one or several are right is never shown.
        if benchmark == "chembench" or "," in answer:
            return "mcq_multi"
        return "mcq_single"
    if benchmark == "bbh":
        return "text"
    return "numeric" if NUMERIC.fullmatch(answer) else None


def _candidates(
    bp: Path, benchmark: str, taken: set[str], taken_hashes: set[str],
    categories: dict[str, tuple[str | None, str | None]],
) -> list[dict[str, Any]]:
    """Gradable bp items that the set does not use yet, with their difficulty."""
    items = pd.read_parquet(bp / "datasets" / benchmark / "items.parquet")
    items = items[(items.reconciliation_confidence == "high")
                  & (items.target_conflict.astype(str) == "False")]
    irt = _irt(bp, benchmark)
    columns = ["item_id", "irt"] + (["category"] if "category" in irt else [])
    frame = items.merge(irt[columns].dropna(subset=["irt"]), on="item_id")
    frame = frame[~frame.item_id.isin(taken) & ~frame.question_hash.isin(taken_hashes)]
    rows = []
    for row in frame.itertuples(index=False):
        options = ([str(o) for o in row.options]
                   if row.options is not None and len(row.options) else None)
        answer = str(row.answer)
        answer_format = _answer_format(benchmark, options, answer)
        if answer_format is None:
            continue
        question = str(row.question)
        if benchmark == "bbh":
            question = question.rsplit("\n\nQUESTION: ", 1)[-1]
        sample_id = str(row.inspect_item_id)
        category = getattr(row, "category", None)
        category = None if category is None or pd.isna(category) else str(category)
        subcategory = None
        if benchmark in categories:
            category, subcategory = categories[benchmark].get(sample_id, (category, None))
        record = {
            "answer_format": answer_format, "benchmark": benchmark, "category": category,
            "item_id": row.item_id, "question": question, "question_hash": row.question_hash,
            "sample_id": sample_id, "split": SPLIT, "subcategory": subcategory,
        }
        if options:
            record["options"] = options
        rows.append({"record": record, "answer": answer, "irt": float(row.irt)})
    return rows


def _rank(item_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{item_id}".encode()).hexdigest()


def _choose(
    rows: list[dict[str, Any]], mix: Counter[str], cutoff: float
) -> tuple[list[dict[str, Any]], int]:
    """Pick rows with train's category counts; return them and how many fall below ``cutoff``.

    Within a category, items at or above the cutoff come first, in a fixed
    pseudo-random order. A category with too few of those is filled with the
    hardest items below the cutoff (gpqa_diamond's unused hard items are all
    chemistry). Whatever is still missing is taken round-robin from the rest.
    """
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["irt"] < cutoff,
                                           -r["irt"] if r["irt"] < cutoff else 0,
                                           _rank(r["record"]["item_id"]))):
        by_category[str(row["record"]["category"])].append(row)
    chosen: list[dict[str, Any]] = []
    for category, count in sorted(mix.items()):
        chosen += by_category[category][:count]
        del by_category[category][:count]
    queues = [by_category[c] for c in sorted(by_category)]
    while len(chosen) < sum(mix.values()) and any(queues):
        for queue in queues:
            if queue and len(chosen) < sum(mix.values()):
                chosen.append(queue.pop(0))
    return chosen, sum(row["irt"] < cutoff for row in chosen)


def _reference_accuracy(bp: Path, benchmark: str, ids: set[str]) -> dict[str, float]:
    """Mean correctness of 7-8B models in bp, or of every respondent where too few."""
    responses = pd.read_parquet(
        bp / "datasets" / benchmark / "responses.parquet",
        columns=["item_id", "source_model_name", "correct"],
    )
    responses = responses[responses.item_id.isin(ids)]
    cohort = responses[responses.source_model_name.fillna("").str.contains(COHORT)]
    overall = responses.groupby("item_id").correct.mean()
    grouped = cohort.groupby("item_id").correct.agg(["mean", "size"])
    result = {}
    for item_id in ids:
        if item_id in grouped.index and grouped.loc[item_id, "size"] >= 5:
            result[item_id] = round(float(grouped.loc[item_id, "mean"]), 4)
        elif item_id in overall.index:
            result[item_id] = round(float(overall.loc[item_id]), 4)
    return result


_LINE_BREAKS = {c: f"\\u{ord(c):04x}" for c in "\x0b\x0c\x1c\x1d\x1e\x85  "}


def _jsonl(rows: list[dict[str, Any]]) -> str:
    """JSONL whose lines survive ``str.splitlines``, which also splits on U+2028."""
    lines = []
    for row in rows:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        lines.append("".join(_LINE_BREAKS.get(c, c) for c in line) + "\n")
    return "".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bp-root", type=Path, required=True,
                        help="bp checkout containing datasets/ and results/")
    args = parser.parse_args()

    data = datasets.data_root()
    manifest = datasets.load_manifest()
    labels_path = data / "private" / "labels.jsonl"
    labels = [row for row in _read_jsonl(labels_path) if row["split"] != SPLIT]
    categories = _hf_categories()
    added: list[dict[str, Any]] = []

    for benchmark in manifest["benchmarks"]:
        public_path = data / "public" / f"{benchmark}.jsonl"
        public = [row for row in _read_jsonl(public_path) if row["split"] != SPLIT]
        entry = manifest["per_benchmark"][benchmark]
        dropped = set(entry.get("dropped_categories", []))
        candidates = [
            row for row in _candidates(
                args.bp_root, benchmark,
                {row["item_id"] for row in public},
                {row["question_hash"] for row in public if row.get("question_hash")},
                categories,
            )
            if row["record"]["category"] not in dropped
        ]
        mix = Counter(str(row["category"]) for row in public if row["split"] == "train")
        chosen, below = _choose(candidates, mix, entry["irt_difficulty_cutoff"])
        if len(chosen) != N_EXAMPLES:
            raise RuntimeError(f"{benchmark}: only {len(chosen)} unused items")
        reference = _reference_accuracy(
            args.bp_root, benchmark, {row["record"]["item_id"] for row in chosen})
        chosen.sort(key=lambda r: (-r["irt"], r["record"]["item_id"]))
        records = [
            {**row["record"], "reference_accuracy": reference.get(row["record"]["item_id"])}
            for row in chosen
        ]
        public_path.write_text(_jsonl(public + records), encoding="utf-8")
        labels += [
            {"answer_format": row["record"]["answer_format"], "item_id": row["record"]["item_id"],
             "split": SPLIT, "target": row["answer"]}
            for row in chosen
        ]
        added += records
        entry[SPLIT] = N_EXAMPLES
        entry[f"{SPLIT}_below_cutoff"] = below
        entry[f"{SPLIT}_mean_irt_difficulty"] = sum(r["irt"] for r in chosen) / N_EXAMPLES
        print(f"{benchmark}: {N_EXAMPLES} examples from {len(candidates)} unused, "
              f"{len({r['category'] for r in records})} categories, {below} below the cutoff, "
              f"mean irt {entry[f'{SPLIT}_mean_irt_difficulty']:.2f} "
              f"(train {entry['train_mean_irt_difficulty']:.2f})")

    labels_path.write_text(_jsonl(labels), encoding="utf-8")
    parquet = data / "items.parquet"
    frame = pd.read_parquet(parquet)
    frame = pd.concat([frame[frame.split != SPLIT], pd.DataFrame(added)], ignore_index=True)
    frame.to_parquet(parquet, index=False)

    manifest[f"{SPLIT}_per_benchmark"] = N_EXAMPLES
    manifest["selection"][SPLIT] = (
        f"{N_EXAMPLES} more bp items per benchmark, not in train or test, with train's "
        "category counts, at or above the IRT cutoff where a category has enough "
        "(else the hardest below it); revealed to the agent, never scored"
    )
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in added
    )
    manifest[f"{SPLIT}_fingerprint"] = hashlib.sha256(payload.encode()).hexdigest()
    (data / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
