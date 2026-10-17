"""Serializable inputs and results for grading one run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["GradeSpec", "GradeSummary"]


@dataclass(frozen=True, slots=True)
class GradeSpec:
    """One run's answers to grade. Targets travel separately, on stdin."""

    run_id: str
    #: JSONL of the public question records the policy saw.
    questions_path: str
    #: JSON object mapping item id to the policy's completion text.
    answers_path: str
    #: Where Inspect writes its log, outside every workspace.
    out_dir: str
    benchmark: str = ""
    split: str = "train"

    @property
    def log_dir(self) -> str:
        return str(Path(self.out_dir) / "log")

    @property
    def summary_path(self) -> str:
        return str(Path(self.out_dir) / "grade.json")

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), "utf-8")
        return path

    @classmethod
    def read(cls, path: Path | str) -> GradeSpec:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{key: raw[key] for key in raw if key in cls.__dataclass_fields__})


@dataclass
class GradeSummary:
    """What grading reports. Written beside the log."""

    run_id: str
    ok: bool = False
    error: str = ""
    n_questions: int = 0

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), "utf-8")
        return path

    @classmethod
    def read(cls, path: Path | str) -> GradeSummary:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{key: raw[key] for key in raw if key in cls.__dataclass_fields__})
