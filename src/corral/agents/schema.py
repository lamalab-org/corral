from dataclasses import dataclass, field
from typing import Any

#: Sentinel returned verbatim by an agent's run method when it gives up on a
#: task. It is passed through :meth:`BaseAgent.run_agent` without running the
#: answer extractor, and :class:`CorralRunner` checks for exactly this string
#: before calling surrender_task(). Every agent must emit this same value so
#: surrender handling stays homogeneous across agents.
SURRENDER_SENTINEL = "SURRENDER"


@dataclass
class Action:
    """Represents an action to be taken"""

    tool_name: str
    arguments: dict[str, Any]


@dataclass
class Thought:
    """Represents agent's reasoning step"""

    content: str


@dataclass
class AgentRunResult:
    """Outcome of running an agent on a single task.

    Returned by :meth:`BaseAgent.run_agent`. Using named fields instead of a
    bare tuple makes the return value self-documenting and prevents accidental
    misuse (e.g. unpacking the elements in the wrong order).

    Attributes:
        answer: The final answer produced by the agent. May be the sentinel
            `"SURRENDER"` when the agent gave up, or a string starting with
            `"Error"` when the run failed.
        messages: The full list of messages exchanged during the task.
        token_usage: Aggregate token usage for the run, with
            `prompt_tokens`, `completion_tokens` and `total_tokens` keys.
        status: The terminal status of the run. `"success"` and `"surrender"`
            are the only statuses whose `answer` should be submitted to the task
            scorer; any other value (`"timeout"`, `"sdk_failure"`,
            `"tool_failure"`, `"max_iterations"`/`"max_turns"`, `"stuck"`,
            `"agent_error"`, ...) marks an infrastructure failure whose `answer`
            is an error string, not a real model answer. Black-box harness
            agents populate this from their `HarnessRunResult`; agents without a
            structured result leave it at the `"success"` default.
        error_message: Human-readable failure detail when `status` is not a
            submit-worthy status; `None` otherwise.
        metadata: Free-form run provenance (harness/SDK versions, tool schema
            hashes, tool-call counts, ...) for the benchmark record.
    """

    answer: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    status: str = "success"
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
