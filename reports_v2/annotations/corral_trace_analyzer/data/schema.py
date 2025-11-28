"""
Data schema definitions for trace analysis
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallData:
    """Represents a single tool call"""

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    status: str
    error_message: str | None = None
    duration: float
    timestamp: str


@dataclass
class StepData:
    """Represents a single step in a trace"""

    trace_id: str
    step_id: str
    step_index: int
    node_type: str  # system, user, assistant, tool
    annotatable: bool
    message: str
    markers: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class TraceData:
    """Represents a complete trace with metadata"""

    trace_id: str
    file_id: str
    annotator: str
    model: str
    environment: str
    agent_type: str
    level: int
    task_id: str
    timestamp: str

    # Trial information
    trial_id: str
    score: float
    success: bool
    submitted_answer: str

    # Execution metrics
    tool_execution_duration: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Tool call statistics
    total_calls: int
    successful_calls: int
    failed_calls: int
    tools_used: list[str]

    # Error statistics
    invalid_tool_errors: int
    invalid_args_errors: int
    execution_errors: int

    # Marker counts
    positive_marker_count: int
    negative_marker_count: int
    neutral_marker_count: int
    validation_attempt_count: int
    backtrack_trigger_count: int
    planning_statement_count: int
    reasoning_statement_count: int
    correct_submission_count: int
    neutral_count: int
    iteration_limit_count: int
    missing_validation_count: int
    unnecessary_tool_use_count: int
    non_sense_count: int
    loop_instance_count: int
    hallucination_count: int
    wrong_planning_count: int
    wrong_reasoning_count: int
    syntax_error_count: int
    early_final_answer_count: int
    give_up_count: int
    inefficient_tool_call_count: int
    misunderstood_tool_count: int

    # QA score
    qa_score: float

    # Steps and tool calls
    steps: list[StepData] = field(default_factory=list)
    tool_calls: list[ToolCallData] = field(default_factory=list)
