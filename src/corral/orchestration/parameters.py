"""Resolve sampling parameter provenance without changing model requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from corral.orchestration.models import AgentRuntimeDefinition


def model_parameter_metadata(
    agent: Any,
    *,
    model: str | None,
    definition: AgentRuntimeDefinition | None = None,
) -> dict[str, Any]:
    """Record explicit settings, agent/SDK defaults, then known catalog defaults.

    Catalog values describe the provider default, not an observed response.
    Keep them out of AgentRuntimeDefinition so logging never changes the
    request, including when a task is reconstructed or resumed.
    """
    parameters = ("temperature", "reasoning_effort")
    options = definition.options if definition is not None else {}
    kwargs = getattr(agent, "kwargs", {})
    unsupported = getattr(agent, "unsupported_model_parameters", ())
    default_factory = getattr(agent, "model_parameter_defaults", None)
    sdk_defaults = default_factory() if callable(default_factory) else {}
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    for name in parameters:
        if name in unsupported:
            values[name], sources[name] = None, "unsupported"
            continue
        explicit = getattr(definition, name, None)
        if explicit is None:
            explicit = options.get(name)
        configured = getattr(agent, name, None)
        if configured is None and isinstance(kwargs, Mapping):
            configured = kwargs.get(name)
        if explicit is not None:
            values[name], sources[name] = explicit, "explicit"
        elif configured is not None:
            values[name], sources[name] = configured, "agent"
        elif sdk_defaults.get(name) is not None:
            values[name], sources[name] = sdk_defaults[name], "sdk_default"
        else:
            values[name], sources[name] = None, "unknown"

    # LiteLLM exposes default_reasoning_effort for some catalog entries. It
    # does not expose a general default temperature. Unknown models, older
    # LiteLLM versions, and unavailable metadata must not prevent execution.
    if model and sources["reasoning_effort"] == "unknown":
        try:
            from litellm import get_model_info

            info = get_model_info(model)
        except Exception:
            info = {}
        effort = info.get("default_reasoning_effort")
        if isinstance(effort, str):
            values["reasoning_effort"] = effort
            sources["reasoning_effort"] = "litellm_default"

    return {**values, "parameter_sources": sources}
