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
    workspace_id: str | None = None


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
