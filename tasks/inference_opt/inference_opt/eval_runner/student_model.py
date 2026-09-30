"""Call student models on their vLLM servers through litellm, as an Inspect model.

Inspect's own ``vllm/`` provider needs openai>=3.1, which litellm (a Corral
dependency) does not allow. litellm's ``hosted_vllm/`` route talks to the same
OpenAI-compatible endpoint and separates reasoning from the answer for us.
"""

from __future__ import annotations

import os
from typing import Any

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

__all__ = ["STUDENT_PROVIDERS", "StudentAPI", "student_model"]

#: Model-spec prefixes served by this provider, e.g. ``vllm/Qwen/Qwen3-8B``.
STUDENT_PROVIDERS = ("vllm", "hosted_vllm")

# Keep litellm offline and quiet: it otherwise downloads a model price list on
# import and logs every request at DEBUG.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_LOG", "WARNING")

_STOP_REASONS = {
    "stop": "stop",
    "length": "max_tokens",
    "content_filter": "content_filter",
    "tool_calls": "tool_calls",
}


# Inspect tracks providers through its registry, so instances must come from
# the decorated class.
@modelapi(name="student")
class StudentAPI(ModelAPI):
    """One student model behind an OpenAI-compatible vLLM server."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key)
        self.timeout = timeout

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        import litellm

        options = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "stop": config.stop_seqs,
            "seed": config.seed,
        }
        response = await litellm.acompletion(
            model=f"hosted_vllm/{self.model_name}",
            api_base=self.base_url,
            api_key=self.api_key or "none",
            messages=[{"role": m.role, "content": m.text} for m in input],
            # One draw per request: some servers reject n > 1, and sample() already
            # sends its draws as separate concurrent requests.
            n=1,
            timeout=self.timeout,
            num_retries=0,  # Inspect's retry loop owns retries
            **{k: v for k, v in options.items() if v is not None},
        )
        return _to_output(self.model_name, response)

    def connection_key(self) -> str:
        return f"{self.base_url}|{self.model_name}"

    def should_retry(self, ex: Exception) -> bool:
        import litellm

        transient = (
            litellm.APIConnectionError,
            litellm.Timeout,
            litellm.RateLimitError,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
        )
        return isinstance(ex, transient)


def _to_output(model_name: str, response: Any) -> ModelOutput:
    choice = response.choices[0]
    text = choice.message.content or ""
    reasoning = getattr(choice.message, "reasoning_content", None) or ""
    content: list[Any] = [ContentReasoning(reasoning=reasoning)] if reasoning else []
    content.append(ContentText(text=text))
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return ModelOutput(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=content, model=model_name, source="generate"
                ),
                stop_reason=_STOP_REASONS.get(choice.finish_reason, "unknown"),
            )
        ],
        completion=text,
        usage=ModelUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
        ),
    )


def student_model(
    model_spec: str, base_url: str, api_key: str | None, timeout: float | None
) -> Model:
    """Build an Inspect model for ``<provider>/<served name>`` on ``base_url``."""
    served_name = model_spec.partition("/")[2]
    api = StudentAPI(served_name, base_url=base_url, api_key=api_key, timeout=timeout)
    return Model(api=api, config=GenerateConfig())
