from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel


class ToolCallStatus(Enum):
    SUCCESS = "success"
    INVALID_TOOL = "invalid_tool"
    INVALID_ARGS = "invalid_args"
    EXECUTION_ERROR = "execution_error"


@dataclass
class ToolArgument:
    """Specification for a tool argument"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    choices: list[Any] | None = None  # from transformers


@dataclass
class ToolCall:
    """Record of a tool being called"""

    tool_name: str
    arguments: dict[str, Any]
    result: str | None
    status: ToolCallStatus
    error_message: str | None
    duration: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


class SurrenderRequest(BaseModel):
    """Request to surrender from a task without submitting an answer"""

    # No parameters needed for surrender


class TrialCompletionResponse(BaseModel):
    """Response when a trial is completed (via submission or surrender)"""

    score: float
    state: dict[str, Any]
    trial_id: str
    surrendered: bool = False


class TrialCreateRequest(BaseModel):
    """Request to open a fresh, isolated runtime for one trial of a task.

    The identifiers are optional provenance carried from the benchmark run;
    the server mints the authoritative `trial_runtime_id` in the response.

    `tool_jobs_per_trial` lets the caller size this runtime's background-job
    pool (`ConcurrencyConfig.tool_jobs_per_trial`); when omitted the server
    keeps its own default (`DEFAULT_JOB_CONCURRENCY`).
    """

    benchmark_run_id: str | None = None
    episode_id: str | None = None
    trial_index: int | None = None
    tool_jobs_per_trial: int | None = None


class TrialCreatedResponse(BaseModel):
    """Response describing a newly created trial runtime.

    `mcp_url` is the per-trial MCP mount an MCP client (Claude Code, Codex, ...)
    should connect to so its tool calls hit *this* runtime's isolated
    environment rather than the shared task template.
    """

    trial_runtime_id: str
    task_id: str
    workspace: str | None = None
    mcp_url: str


class ToLatexRequest(BaseModel):
    """Request to generate LaTeX documentation for a task"""

    output_dir: str
    level: int | str
    env_name: str | None = None
    task_name: str | None = None
    subtask_index: int | None = None
    cache_dir: str | None = None
    verbosity: str | None = None


class ClearLatexCacheRequest(BaseModel):
    """Request to clear LaTeX cache files"""

    env_name: str
    level: int | str
    cache_dir: str | None = None
