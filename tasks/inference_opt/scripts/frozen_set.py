"""Helpers shared by the scripts that build and edit a frozen question set.

A frozen set lives in ``inference_opt/data/frozen/<version>/``: one public JSONL
file per benchmark, the private labels, ``items.parquet`` and ``manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

#: Model names of the 7-8B reference cohort in bp's response tables.
COHORT = re.compile(r"(?<![\d.])[78](?:\.\d+)?b(?![a-z])", re.IGNORECASE)

#: Characters ``str.splitlines`` treats as line breaks that ``json.dumps`` keeps raw.
_LINE_BREAKS = {c: f"\\u{ord(c):04x}" for c in "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"}


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    """Rows as JSONL whose lines survive ``str.splitlines``."""
    lines = []
    for row in rows:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        lines.append("".join(_LINE_BREAKS.get(c, c) for c in line) + "\n")
    return "".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rank(seed: int, item_id: str) -> str:
    """A fixed pseudo-random sort key for one item."""
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def reference_accuracy(bp: Path, benchmark: str, ids: set[str]) -> dict[str, float]:
    """Mean correctness of bp's 7-8B models per item, or of every model where fewer than five."""
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


def content_fingerprint(data: Path, benchmarks: list[str]) -> str:
    """sha256 of every train and test record, in file order."""
    rows = [
        row
        for benchmark in benchmarks
        for row in read_jsonl(data / "public" / f"{benchmark}.jsonl")
        if row["split"] != "examples"
    ]
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return hashlib.sha256(payload.encode()).hexdigest()
