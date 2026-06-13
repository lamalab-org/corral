from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from corral.backend.schema import ToolCall, ToolCallStatus

if TYPE_CHECKING:
    from corral.backend.task import TaskDefinition

TrialStatus = Literal[
    "created",
    "running",
    "submitted",
    "surrendered",
    "scored",
    "finalized",
]


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def get_path(data: Any, path: str) -> Any:
    """Resolve a dotted path (e.g. ``"answer.smiles"``) into nested data."""
    current = data
    for part in path.split("."):
        current = current[part]
    return current


@dataclass
class TaskRunState:
    """Runtime outcome of one task: its output, score, and feedback."""

    task_id: str
    output: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    feedback: str | None = None


@dataclass
class CorralState:
    """Single source of truth for one Corral runtime session.

    This is the only mutable runtime state object. Environments, routers,
    tools, and scoring code should read/write through this class.
    Task definitions remain immutable configuration; this class stores runtime data.
    """

    # Identity
    task_id: str
    task_prompt: str | list[dict[str, Any]]
    trial_id: str = "0"
    trial_counter: int = -1
    run_id: str | None = None

    # Optional task-group identity
    task_group_id: str | None = None

    # Trial lifecycle
    status: TrialStatus = "created"
    started_at: datetime = field(default_factory=utcnow)
    ended_at: datetime | None = None

    # Agent/environment interaction trace
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    # Submission/scoring
    submitted_answer: str | None = None
    score: float | None = None
    feedback: str | None = None
    surrendered: bool = False
    is_attempted: bool = False

    # Per-task runtime outcomes; the dict object is shared between linked
    # environments so chained tasks see their dependencies' outputs
    task_runs: dict[str, TaskRunState] = field(default_factory=dict)

    # Runtime resources
    workspace: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    hidden_args: dict[str, Any] = field(default_factory=dict)

    # Archived trial snapshots
    trials: dict[str, dict[str, Any]] = field(default_factory=dict)

    def start_new_trial(
        self,
        task_prompt: str | list[dict[str, Any]],
        workspace: str | None = None,
    ) -> str:
        """Archive current completed trial, then start a clean trial."""
        if self.status in {"submitted", "surrendered", "scored", "finalized"}:
            self.finalize()
            self.trials[self.trial_id] = self.snapshot(include_trials=False)

        self.trial_counter += 1
        self.trial_id = str(self.trial_counter)
        self.task_prompt = task_prompt
        self.workspace = workspace

        self.status = "running"
        self.started_at = utcnow()
        self.ended_at = None

        self.messages.clear()
        self.tool_calls.clear()
        self.submitted_answer = None
        self.score = None
        self.feedback = None
        self.surrendered = False
        self.is_attempted = False

        return self.trial_id

    def record_message(self, role: str, content: str, **metadata: Any) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": utcnow().isoformat(),
                **metadata,
            }
        )

    def record_tool_call(self, call: ToolCall) -> None:
        self.tool_calls.append(call)

    def submit(self, answer: str) -> None:
        self.submitted_answer = answer
        self.is_attempted = True
        self.status = "submitted"
        self.ended_at = utcnow()

    def surrender(self) -> float:
        self.surrendered = True
        self.is_attempted = True
        self.status = "surrendered"
        self.ended_at = utcnow()
        return 0.0 if self.score is None else self.score

    def set_score(self, score: float, feedback: str | None = None) -> None:
        self.score = score
        self.feedback = feedback
        self.status = "scored"

    def store_task_output(
        self,
        task_id: str,
        answer: Any,
        score: float | None = None,
        feedback: str | None = None,
    ) -> None:
        """Record a task's output so dependent tasks can consume it."""
        self.task_runs[task_id] = TaskRunState(
            task_id=task_id,
            output={"answer": answer},
            score=score,
            feedback=feedback,
        )

    def get_output(self, task_id: str, key: str = "answer") -> Any:
        """Read one field of a task's output; `key` may be a dotted path."""
        if task_id not in self.task_runs:
            raise KeyError(f"Task {task_id!r} has no stored output")
        try:
            return get_path(self.task_runs[task_id].output, key)
        except (KeyError, TypeError) as e:
            raise KeyError(f"Task {task_id!r} has no output key {key!r}") from e

    def is_completed(self, task_id: str) -> bool:
        return task_id in self.task_runs

    def dependencies_satisfied(self, task: TaskDefinition) -> bool:
        return all(self.is_completed(dep_id) for dep_id in task.dependencies())

    def resolve_inputs(self, task: TaskDefinition) -> dict[str, Any]:
        """Combine a task's initial input with its resolved dependency outputs."""
        resolved = dict(task.initial_input)

        for input_name, ref in task.input_map.items():
            if not self.is_completed(ref.task_id):
                raise RuntimeError(
                    f"Task {task.name!r} is not ready: dependency "
                    f"{ref.task_id!r} has not completed."
                )
            resolved[input_name] = self.get_output(ref.task_id, ref.key)

        return resolved

    def finalize(self) -> None:
        if self.ended_at is None:
            self.ended_at = utcnow()
        self.status = "finalized"

    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    def tool_statistics(self) -> dict[str, Any]:
        return {
            "total_calls": len(self.tool_calls),
            "successful_calls": len(
                [t for t in self.tool_calls if t.status == ToolCallStatus.SUCCESS]
            ),
            "failed_calls": len(
                [t for t in self.tool_calls if t.status != ToolCallStatus.SUCCESS]
            ),
            "tools_used": sorted({t.tool_name for t in self.tool_calls}),
            "error_types": {
                status.value: len([t for t in self.tool_calls if t.status == status])
                for status in ToolCallStatus
                if status != ToolCallStatus.SUCCESS
            },
            "tool_calls": [
                {
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "result": call.result,
                    "status": call.status.value,
                    "error_message": call.error_message,
                    "duration": call.duration,
                    "timestamp": (
                        call.timestamp.isoformat() if call.timestamp else None
                    ),
                }
                for call in self.tool_calls
            ],
        }

    def snapshot(self, include_trials: bool = True) -> dict[str, Any]:
        data = asdict(self)
        # Hidden args can hold scoring secrets and arbitrary objects; never expose them.
        data.pop("hidden_args", None)
        data["started_at"] = self.started_at.isoformat()
        data["ended_at"] = self.ended_at.isoformat() if self.ended_at else None
        data["duration"] = self.duration_seconds()
        data["tool_statistics"] = self.tool_statistics()
        # ToolCall records are already serialized inside tool_statistics
        data.pop("tool_calls", None)
        if not include_trials:
            data.pop("trials", None)
        return data
