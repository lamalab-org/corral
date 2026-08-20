"""Shared foundation for agents that own a Corral session loop."""

from __future__ import annotations

import importlib.resources
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from promptstore import PromptStore

from corral.agents.hooks import AgentHooks
from corral.agents.prompt_utils import ensure_jinja_compatible, get_prompt
from corral.agents.schema import AgentOutcome, AgentUsage
from corral.agents.usage import usage_from_mapping
from corral.agents.utils import llm_call as _default_llm_call

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from corral.agents.session import AgentSession

# A module-level seam lets tests and integrations replace the provider call
# without mutating an agent instance.
llm_call = _default_llm_call


class BaseAgent(ABC):
    """Common configuration for agents implementing `run_session`.

    `run_session` is the sole agent execution contract.
    """

    def __init__(
        self,
        *,
        model: str,
        api_endpoint: str | None,
        system_prompt: str | None,
        user_prompt: str | Any,
        surrender_prompt: str | Any | None,
        temperature: float,
        hooks: AgentHooks | None = None,
        **kwargs: Any,
    ) -> None:
        removed_limits = {"max_iterations", "max_turns"} & kwargs.keys()
        if removed_limits:
            names = ", ".join(sorted(removed_limits))
            raise TypeError(
                f"agent-level {names} is no longer supported; configure the "
                "per-task max_iterations budget on BenchmarkTaskMetadata"
            )
        self.model = model
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.hooks = hooks or AgentHooks()
        self.kwargs = dict(kwargs)

        with importlib.resources.path("corral.agents", "") as style_path:
            store = PromptStore(f"{style_path}/prompts")
        self.system_prompt = get_prompt(
            store,
            system_prompt,
            "system_prompt/system_prompt",
        ).fill({})
        try:
            self.user_prompt = store.get(user_prompt)
        except Exception:
            self.user_prompt = ensure_jinja_compatible(user_prompt)
        self.surrender_prompt = (
            ensure_jinja_compatible(surrender_prompt)
            if surrender_prompt is not None
            else None
        )

    def _call_kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs)

    def _usage(
        self,
        raw_usage: Mapping[str, Any] | None,
        *,
        llm_calls: int = 0,
    ) -> AgentUsage:
        """Convert provider usage fields into Corral's canonical schema."""
        return usage_from_mapping(raw_usage, llm_calls=llm_calls)

    @abstractmethod
    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Run the agent-owned loop against one task-bound session."""


@dataclass(slots=True)
class _UsageAccumulator:
    """Accumulate physical provider usage for one session run."""

    convert: Callable[..., AgentUsage] = field(
        default=usage_from_mapping,
        repr=False,
    )
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    llm_calls: int = 0

    def add(self, usage: Mapping[str, Any] | None) -> None:
        normalized = self.convert(usage, llm_calls=1)
        self.input_tokens += normalized.input_tokens
        self.output_tokens += normalized.output_tokens
        self.reasoning_tokens += normalized.reasoning_tokens
        self.llm_calls += normalized.llm_calls

    def outcome(self) -> AgentUsage:
        return AgentUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            llm_calls=self.llm_calls,
        )


async def call_model(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    temperature: float,
    api_endpoint: str | None,
    kwargs: Mapping[str, Any],
) -> Any:
    """Call LiteLLM without blocking the async runtime."""
    call_kwargs = dict(kwargs)
    call_kwargs.pop("return_usage", None)
    return await llm_call(
        model=model,
        messages=[dict(message) for message in messages],
        tools=[dict(tool) for tool in tools] if tools is not None else None,
        temperature=temperature,
        api_endpoint=api_endpoint,
        return_usage=True,
        **call_kwargs,
    )


def provider_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return provider-safe copies of canonical session messages."""
    provider_fields = {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "function_call",
    }
    return [
        {key: value for key, value in message.items() if key in provider_fields}
        for message in messages
    ]


def prompt_with_state_history(
    prompt: str,
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Prepend canonical agent-context messages to an opaque harness prompt."""
    history = provider_messages(messages)
    if not history:
        return prompt
    rendered = json.dumps(
        history,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return (
        "Continue from this canonical Corral agent conversation. "
        "Treat it as prior context and do not repeat completed tool calls:\n"
        f"{rendered}\n\nCurrent task input follows.\n\n{prompt}"
    )


__all__ = ["BaseAgent", "prompt_with_state_history", "provider_messages"]
