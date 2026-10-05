"""Mark every ChemBench multiple-choice question as ``mcq_multi``.

ChemBench's own harness asks for every correct option on every multiple-choice
question, whether one or several are right. Storing the true count as
``mcq_single`` or ``mcq_multi`` would tell the agent, through the public files
and revealed examples, how many options to pick. Its scorer already compares the
full set of letters, so grading does not change. Updates ``content_fingerprint``.

    uv run python scripts/mark_chembench_multi.py
"""

from __future__ import annotations

import json

import pandas as pd
from drop_category import _fingerprint
from inference_opt import datasets
from select_examples import _jsonl, _read_jsonl


def main() -> None:
    data = datasets.data_root()
    manifest = datasets.load_manifest()
    public_path = data / "public" / "chembench.jsonl"
    public = _read_jsonl(public_path)
    multiple_choice = {row["item_id"] for row in public if row.get("options")}
    for row in public:
        if row["item_id"] in multiple_choice:
            row["answer_format"] = "mcq_multi"
    public_path.write_text(_jsonl(public), encoding="utf-8")

    labels_path = data / "private" / "labels.jsonl"
    labels = _read_jsonl(labels_path)
    for row in labels:
        if row["item_id"] in multiple_choice:
            row["answer_format"] = "mcq_multi"
    labels_path.write_text(_jsonl(labels), encoding="utf-8")

    parquet = data / "items.parquet"
    frame = pd.read_parquet(parquet)
    frame.loc[frame.item_id.isin(multiple_choice), "answer_format"] = "mcq_multi"
    frame.to_parquet(parquet, index=False)

    manifest["per_benchmark"]["chembench"]["answer_formats"] = (
        "every multiple-choice question is mcq_multi: answer with all correct options"
    )
    manifest["content_fingerprint"] = _fingerprint(data, manifest["benchmarks"])
    (data / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"chembench: {len(multiple_choice)} multiple-choice questions marked mcq_multi")


if __name__ == "__main__":
    main()
