"""Runtime execution and environment-loading APIs, exported lazily."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ENVIRONMENT_NAMES": (
        "corral.runtime.environment_loader",
        "ENVIRONMENT_NAMES",
    ),
    "ENVIRONMENT_PRESETS": (
        "corral.runtime.environment_loader",
        "ENVIRONMENT_PRESETS",
    ),
    "EnvironmentName": (
        "corral.runtime.environment_loader",
        "EnvironmentName",
    ),
    "EnvironmentPreset": (
        "corral.runtime.environment_loader",
        "EnvironmentPreset",
    ),
    "TaskRuntime": ("corral.runtime.task_runner", "TaskRuntime"),
    "load_environment_group": (
        "corral.runtime.environment_loader",
        "load_environment_group",
    ),
    "normalise_environment_name": (
        "corral.runtime.environment_loader",
        "normalise_environment_name",
    ),
}

__all__ = [
    "ENVIRONMENT_NAMES",
    "ENVIRONMENT_PRESETS",
    "EnvironmentName",
    "EnvironmentPreset",
    "TaskRuntime",
    "load_environment_group",
    "normalise_environment_name",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
