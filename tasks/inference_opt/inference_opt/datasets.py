"""Read the frozen public questions and private targets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "BENCHMARKS",
    "DATASET_VERSION",
    "DatasetError",
    "FrozenItem",
    "Split",
    "data_root",
    "example_baselines_path",
    "load_example_baselines",
    "load_items",
    "load_manifest",
    "load_targets",
    "public_record",
    "write_jsonl",
]

DATASET_VERSION = "v2"

#: One benchmark per MATH subject, Level 5 only (``scripts/select_math_levels.py``),
#: and the three GSM-Symbolic variants (``scripts/select_gsm_symbolic.py``).
#: ``data/frozen/v1`` is the earlier mixed set (MMLU-Pro, BBH, GPQA, MATH,
#: ChemBench, ARC); it is kept on disk but not used.
BENCHMARKS: tuple[str, ...] = (
    "math_algebra",
    "math_prealgebra",
    "math_number_theory",
    "math_counting_probability",
    "math_geometry",
    "gsm_symbolic",
    "gsm_p1",
    "gsm_p2",
)

#: ``train`` is scored by experiments, ``test`` by the submission, and ``examples``
#: is what the agent reveals and policies get as ``ctx.examples``; it is never scored.
Split = Literal["train", "test", "examples"]


class DatasetError(RuntimeError):
    """Raised when the frozen dataset is missing or inconsistent."""


def data_root(version: str = DATASET_VERSION) -> Path:
    """Locate the frozen dataset directory.

    ``CORRAL_INFERENCE_DATA_DIR`` overrides the packaged location, which is how the
    benchmark runner points at a dataset mounted outside the image.
    """
    override = os.environ.get("CORRAL_INFERENCE_DATA_DIR")
    if override:
        root = Path(override).expanduser()
        return root / version if (root / version).is_dir() else root
    return Path(__file__).resolve().parent / "data" / "frozen" / version


def labels_path(version: str = DATASET_VERSION) -> Path:
    """Locate ``private/labels.jsonl``, which only trusted code may read."""
    override = os.environ.get("CORRAL_INFERENCE_LABELS_PATH")
    if override:
        return Path(override).expanduser()
    return data_root(version) / "private" / "labels.jsonl"


@dataclass(frozen=True, slots=True)
class FrozenItem:
    """One question in the frozen set, without its answer."""

    item_id: str
    benchmark: str
    sample_id: str
    split: str
    question: str
    answer_format: str
    options: tuple[str, ...] | None = None
    category: str | None = None
    subcategory: str | None = None
    #: Accuracy of a 7-8B reference model cohort on this item; lower means harder.
    reference_accuracy: float | None = None
    question_hash: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> FrozenItem:
        options = record.get("options")
        return cls(
            item_id=str(record["item_id"]),
            benchmark=str(record["benchmark"]),
            sample_id=str(record.get("sample_id", record["item_id"].split(":", 1)[-1])),
            split=str(record["split"]),
            question=str(record["question"]),
            answer_format=str(record["answer_format"]),
            options=tuple(str(option) for option in options) if options else None,
            category=record.get("category"),
            subcategory=record.get("subcategory"),
            reference_accuracy=record.get("reference_accuracy"),
            question_hash=record.get("question_hash"),
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetError(
            f"frozen dataset file not found: {path}; set CORRAL_INFERENCE_DATA_DIR "
            "to a mounted dataset"
        )
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{number}: malformed JSON: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL, creating parent directories. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
            written += 1
    return written


@lru_cache(maxsize=16)
def _cached_items(benchmark: str, version: str) -> tuple[dict[str, Any], ...]:
    path = data_root(version) / "public" / f"{benchmark}.jsonl"
    return tuple(_read_jsonl(path))


def load_items(
    benchmark: str,
    split: Split | None = None,
    *,
    version: str = DATASET_VERSION,
) -> list[FrozenItem]:
    """Load the frozen items for one benchmark, optionally filtered by split."""
    if benchmark not in BENCHMARKS:
        raise DatasetError(
            f"unknown benchmark {benchmark!r}; expected one of {', '.join(BENCHMARKS)}"
        )
    records = _cached_items(benchmark, version)
    items = [FrozenItem.from_record(dict(record)) for record in records]
    if split is not None:
        items = [item for item in items if item.split == split]
    if not items:
        raise DatasetError(
            f"no items for benchmark={benchmark!r} split={split!r} in {data_root(version)}"
        )
    return items


def load_targets(
    benchmark: str | None = None,
    split: Split | None = None,
    *,
    version: str = DATASET_VERSION,
) -> dict[str, str]:
    """Load gold answers keyed by ``item_id``.

    Only ever called from trusted evaluator-side code. Policy execution does not
    receive the path this reads.
    """
    path = labels_path(version)
    if not path.is_file():
        raise DatasetError(
            f"private labels not found: {path}. "
            "Set CORRAL_INFERENCE_LABELS_PATH to an evaluator-only labels file."
        )
    records = _read_jsonl(path)
    targets: dict[str, str] = {}
    for record in records:
        if benchmark and not str(record["item_id"]).startswith(f"{benchmark}:"):
            continue
        if split and record.get("split") != split:
            continue
        targets[str(record["item_id"])] = str(record["target"])
    return targets


def load_manifest(version: str = DATASET_VERSION) -> dict[str, Any]:
    """Read the provenance manifest, including ``content_fingerprint``."""
    path = data_root(version) / "manifest.json"
    if not path.is_file():
        raise DatasetError(f"frozen dataset manifest not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"malformed manifest at {path}: {exc}") from exc


def example_baselines_path(version: str = DATASET_VERSION) -> Path:
    """Where ``measure_baselines.py --splits examples`` writes the students' answers."""
    return Path(__file__).resolve().parent / "data" / "baselines" / f"examples_{version}.jsonl"


def load_example_baselines(
    models: Iterable[str], *, version: str = DATASET_VERSION
) -> dict[str, dict[str, dict[str, Any]]]:
    """Each student's zero-shot answer to each example, as ``{model: {item_id: row}}``.

    A row holds ``correct``, ``completion`` and ``finish_reason``. A model that
    was never measured maps to an empty dict.
    """
    wanted = set(models)
    result: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in wanted}
    path = example_baselines_path(version)
    if not path.is_file():
        return result
    for record in _read_jsonl(path):
        if record["model"] in wanted:
            result[record["model"]][str(record["item_id"])] = record
    return result


def public_record(item: FrozenItem) -> dict[str, Any]:
    """The subset of an item policy execution is allowed to receive."""
    record = {
        "item_id": item.item_id,
        "benchmark": item.benchmark,
        "sample_id": item.sample_id,
        "split": item.split,
        "question": item.question,
        "answer_format": item.answer_format,
        "category": item.category,
        "subcategory": item.subcategory,
    }
    if item.options:
        record["options"] = list(item.options)
    return record
