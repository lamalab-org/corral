"""Add GSM-Symbolic, GSM-P1 and GSM-P2 to the frozen set (``v2``).

GSM-Symbolic (Apple, https://github.com/apple-aiml-research/ml-gsm-symbolic)
turns GSM8K test questions into templates whose names and numbers vary: each
template has 50 instances. GSM-P1 adds one clause to every template, GSM-P2
two, so each needs more reasoning steps. GSM-Symbolic and GSM-P1 share 100
GSM8K questions; GSM-P2 uses 50 of them.

Splits are by GSM8K question, the same in all three variants, so no template a
policy is tested on was ever seen in train or in the examples:

    the 50 questions in GSM-P2:   20 train, 20 test, 10 examples
    the other 50:                 10 train, 10 test, 10 examples (not in GSM-P2)

Each template contributes 5 of its 50 instances, chosen in a fixed
pseudo-random order, so a split holds 150 questions (100 for GSM-P2).
``category`` is the template, which the template-consistency score groups by.

    uv run python scripts/select_gsm_symbolic.py --source-dir <dir with GSM_symbolic.jsonl,
        GSM_p1.jsonl and GSM_p2.jsonl from the repository's generated_data/>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_set import rank, read_jsonl, replace_benchmarks  # noqa: E402

VERSION = "v2"
SEED = 20261008
INSTANCES = 5
SPLITS = ("train", "test", "examples")
#: GSM8K questions per split: those that GSM-P2 also uses, and those it does not.
IN_P2 = {"train": 20, "test": 20, "examples": 10}
NOT_IN_P2 = {"train": 10, "test": 10, "examples": 10}
VARIANTS = {
    "gsm_symbolic": "GSM_symbolic.jsonl",
    "gsm_p1": "GSM_p1.jsonl",
    "gsm_p2": "GSM_p2.jsonl",
}


def _target(answer: str) -> str:
    """The final answer of a GSM8K-style solution, after ``####``."""
    return answer.rsplit("####", 1)[-1].strip().replace(",", "")


def _split_by_question(rows: dict[str, list[dict[str, Any]]]) -> dict[int, str]:
    """Assign every GSM8K question to one split, the same for all variants."""
    in_p2 = sorted({r["original_id"] for r in rows["gsm_p2"]}, key=lambda q: rank(SEED, str(q)))
    others = sorted(
        {r["original_id"] for r in rows["gsm_symbolic"]} - set(in_p2),
        key=lambda q: rank(SEED, str(q)),
    )
    split_of: dict[int, str] = {}
    for pool, sizes in ((in_p2, IN_P2), (others, NOT_IN_P2)):
        start = 0
        for split in SPLITS:
            for question in pool[start:start + sizes[split]]:
                split_of[question] = split
            start += sizes[split]
    return split_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = {name: read_jsonl(args.source_dir / file) for name, file in VARIANTS.items()}
    split_of = _split_by_question(rows)

    publics: dict[str, list[dict[str, Any]]] = {}
    labels: list[dict[str, Any]] = []
    per_benchmark: dict[str, Any] = {}
    for benchmark, source in rows.items():
        by_template: dict[int, list[dict[str, Any]]] = {}
        for row in source:
            if row["original_id"] in split_of:
                by_template.setdefault(row["id"], []).append(row)
        public = []
        for template, instances in sorted(by_template.items()):
            chosen = sorted(instances, key=lambda r: rank(SEED, f"{benchmark}:{template}:{r['instance']}"))
            for row in sorted(chosen[:INSTANCES], key=lambda r: r["instance"]):
                sample_id = f"{template}-{row['instance']}"
                split = split_of[row["original_id"]]
                public.append({
                    "answer_format": "numeric",
                    "benchmark": benchmark,
                    "category": f"template {template}",
                    "item_id": f"{benchmark}:{sample_id}",
                    "question": row["question"],
                    "question_hash": hashlib.sha256(row["question"].encode()).hexdigest(),
                    "reference_accuracy": None,
                    "sample_id": sample_id,
                    "split": split,
                    "subcategory": f"gsm8k {row['original_id']}",
                })
                labels.append({
                    "answer_format": "numeric",
                    "item_id": f"{benchmark}:{sample_id}",
                    "split": split,
                    "target": _target(row["answer"]),
                })
        public.sort(key=lambda r: (SPLITS.index(r["split"]), r["item_id"]))
        publics[benchmark] = public
        counts = {s: sum(r["split"] == s for r in public) for s in SPLITS}
        templates = {s: len({r["category"] for r in public if r["split"] == s}) for s in SPLITS}
        per_benchmark[benchmark] = {
            "instances_per_template": INSTANCES,
            "templates": templates,
            "selected": counts["train"] + counts["test"],
            **counts,
        }
        print(f"{benchmark:13} " + "  ".join(
            f"{s} {templates[s]} templates / {counts[s]} questions" for s in SPLITS))

    data = ROOT / "inference_opt" / "data" / "frozen" / VERSION
    replace_benchmarks(data, VERSION, "gsm", publics, labels, per_benchmark, {
        "source": "https://github.com/apple-aiml-research/ml-gsm-symbolic, generated_data/",
        "rule": (f"{INSTANCES} instances per template; splits by GSM8K question, the "
                 "same in every variant"),
        "answer_format": "numeric; graded by the repository's rule: the last number "
                         "in the reply, compared as a float",
        "examples": "revealed to the agent, never scored",
        "seed": SEED,
    })
    print(f"wrote {data}")


if __name__ == "__main__":
    main()
