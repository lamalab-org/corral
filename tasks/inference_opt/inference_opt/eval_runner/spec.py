"""Serializable inputs and results for one policy evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RunSpec", "RunSummary"]


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Everything the evaluator needs to run one policy over one question set."""

    run_id: str
    policy_dir: str
    #: JSONL of records written for this evaluation.
    questions_path: str
    out_dir: str

    #: e.g. ``"vllm/Qwen/Qwen2.5-7B-Instruct"`` or ``"mockllm/model"``.
    model_spec: str = "mockllm/model"
    #: Must end in ``/v1``. When this is unset for a ``vllm/`` model, inspect's
    #: provider starts a *new* local vLLM server rather than using ours.
    base_url: str | None = None
    api_key: str | None = None

    total_calls: int = 250
    max_calls_per_question: int = 8
    max_tokens_per_call: int = 2048
    setup_calls: int = 0

    #: Labeled train examples for ``Policy.setup`` when this is a train run.
    revealed_path: str | None = None
    #: Overrides the policy manifest, for the dry-run path.
    time_limit_s: int = 1800
    epochs: int = 1
    seed: int = 0
    benchmark: str = ""
    split: str = "train"
    policy_api: str = "primitive"

    @property
    def log_dir(self) -> str:
        return str(Path(self.out_dir) / "log")

    @property
    def predictions_path(self) -> str:
        return str(Path(self.out_dir) / "predictions.jsonl")

    @property
    def summary_path(self) -> str:
        return str(Path(self.out_dir) / "summary.json")

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), "utf-8")
        return path

    @classmethod
    def read(cls, path: Path | str) -> RunSpec:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {key: raw[key] for key in raw if key in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class RunSummary:
    """What the evaluator reports. Written to ``summary.json``."""

    run_id: str
    ok: bool = False
    error: str = ""
    n_questions: int = 0
    n_answered: int = 0
    n_crashed: int = 0
    n_unparseable: int = 0
    calls_used: int = 0
    output_tokens: int = 0
    setup_calls_used: int = 0
    seconds: float = 0.0
    execution: str = "sequential"
    manifest: dict[str, Any] = field(default_factory=dict)
    budget_exhausted_at: dict[str, Any] | None = None
    questions_after_exhaustion: int = 0
    per_component: dict[str, Any] = field(default_factory=dict)
    first_tracebacks: list[str] = field(default_factory=list)
    setup_log: list[str] = field(default_factory=list)

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), "utf-8")
        return path

    @classmethod
    def read(cls, path: Path | str) -> RunSummary:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {key: raw[key] for key in raw if key in cls.__dataclass_fields__}
        return cls(**known)
