"""Read the frozen public questions and private targets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from inference_opt.api import Question

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

__all__ = [
    "BENCHMARKS",
    "DATASET_VERSION",
    "DatasetError",
    "FrozenItem",
    "Split",
    "data_root",
    "iter_questions",
    "load_items",
    "load_manifest",
    "load_targets",
    "public_record",
    "to_question",
    "write_jsonl",
]

DATASET_VERSION = "v1"

#: The six benchmarks in the frozen set. ``aime2024`` was dropped (only 30 items
#: exist in total, which cannot support a 30/30 split); ``arc_challenge`` replaced it.
BENCHMARKS: tuple[str, ...] = (
    "chembench",
    "bbh",
    "gpqa_diamond",
    "gsm8k",
    "mmlu_pro",
    "arc_challenge",
)

Split = Literal["train", "test"]

#: Fields exposed to policy execution. The target is not included.
PUBLIC_FIELDS: tuple[str, ...] = (
    "item_id",
    "benchmark",
    "sample_id",
    "split",
    "question",
    "options",
    "answer_format",
    "category",
    "subcategory",
)


class DatasetError(RuntimeError):
    """Raised when the frozen dataset is missing or internally inconsistent.

    This is a harness fault, never an agent fault, and must propagate rather than
    being folded into a score of zero.
    """


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
    #: Accuracy of the reference model cohort on this item when the set was
    #: assembled. Between 0.2 and 0.8 by construction; lower means harder.
    reference_accuracy: float | None = None
    question_hash: str | None = None

    @property
    def answer_type(self) -> str:
        """Map the storage-level format onto the policy-facing answer type."""
        if self.answer_format in ("mcq_single", "mcq_multi"):
            return "mcq"
        if self.answer_format == "numeric":
            return "numeric"
        return "text"

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
            f"frozen dataset file not found: {path}. Build it with "
            "scripts/build_pool.py then scripts/freeze_dataset.py."
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
    records = _read_jsonl(labels_path(version))
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


def to_question(item: FrozenItem, index: int = 0, total: int = 1) -> Question:
    """Convert a frozen item into the policy-facing :class:`~inference_opt.api.Question`."""
    labels = (
        tuple(chr(ord("A") + position) for position in range(len(item.options)))
        if item.options
        else None
    )
    return Question(
        id=item.item_id,
        text=item.question,
        benchmark=item.benchmark,
        answer_type=item.answer_type,  # type: ignore[arg-type]
        choices=item.options,
        choice_labels=labels,
        topic=item.category,
        index=index,
        total=total,
    )


def iter_questions(items: Sequence[FrozenItem]) -> Iterator[Question]:
    """Yield questions with ``index``/``total`` populated for pacing."""
    total = len(items)
    for index, item in enumerate(items):
        yield to_question(item, index=index, total=total)
