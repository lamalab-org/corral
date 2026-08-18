"""Load registered environment factories with conventional arguments."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from corral.core.environment import Environment


@dataclass(frozen=True, slots=True)
class EnvironmentPreset:
    """Import and task-config conventions for one repository environment."""

    factory: str
    source_dir: str
    config_parameter: str | None = None
    default_config_file: str | None = None


EnvironmentName = Literal[
    "afm",
    "catalyst",
    "corral_md",
    "ml",
    "resistor_network",
    "retrosynthesis",
    "samplemath",
    "spectra_elucidation",
    "wetlab",
]

ENVIRONMENT_NAMES: tuple[EnvironmentName, ...] = (
    "afm",
    "catalyst",
    "corral_md",
    "ml",
    "resistor_network",
    "retrosynthesis",
    "samplemath",
    "spectra_elucidation",
    "wetlab",
)

ENVIRONMENT_PRESETS: dict[EnvironmentName, EnvironmentPreset] = {
    "afm": EnvironmentPreset(
        "env:create_environments", "tasks/afm/src", "task_json_path"
    ),
    "catalyst": EnvironmentPreset(
        "catalyst.env:create_environments",
        "tasks/catalyst/src",
        "local_dir",
    ),
    "corral_md": EnvironmentPreset(
        "corral_md.env:create_environments", "tasks/corral_md/src"
    ),
    "ml": EnvironmentPreset(
        "ml.env:create_environments", "tasks/ml/src", "task_json_path"
    ),
    "resistor_network": EnvironmentPreset(
        "resistor_network.env:create_environments",
        "tasks/resistor_network/src",
        "task_json_path",
    ),
    "retrosynthesis": EnvironmentPreset(
        "retrosynthesis.env:create_rethrosynthesis_environments",
        "tasks/retrosynthesis",
    ),
    "samplemath": EnvironmentPreset(
        "samplemath.env_subtask:create_environments",
        "tasks/samplemath",
        "task_json_path",
        "task_1.json",
    ),
    "spectra_elucidation": EnvironmentPreset(
        "spectra_elucidation.env:create_spectra_elu_environments",
        "tasks/spectra_elucidation",
    ),
    "wetlab": EnvironmentPreset(
        "wetlab.env:create_qualysis_environments", "tasks/wetlab"
    ),
}

_ENVIRONMENT_ALIASES = {
    "corral-md": "corral_md",
    "resistor-network": "resistor_network",
    "spectra-elucidation": "spectra_elucidation",
}


def normalise_environment_name(name: str) -> str:
    """Return the canonical spelling for a built-in environment name."""
    return _ENVIRONMENT_ALIASES.get(name, name)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_factory(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("an environment factory must use the form 'module:function'")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"environment factory {specification!r} is not callable")
    return factory


def _default_task_config(
    repository_root: Path,
    environment_name: str,
    *,
    level: int,
    subtasks: bool,
) -> Path:
    task_kind = "subtasks_json" if subtasks else "tasks_json"
    return (
        repository_root
        / "tasks"
        / environment_name
        / "environments"
        / f"level_{level}"
        / task_kind
    )


def _factory_kwargs(
    factory: Callable[..., Any],
    *,
    initial: Mapping[str, Any],
    work_dir: Path,
    level: int,
    subtasks: bool,
    config_parameter: str | None,
    task_config: Path | None,
) -> dict[str, Any]:
    """Fill only common parameters explicitly declared by the factory."""
    parameters = inspect.signature(factory).parameters
    kwargs = dict(initial)
    common = {
        "work_dir": str(work_dir),
        "level": level,
        "subtask": subtasks,
        "subtask_level": subtasks,
    }
    for name, value in common.items():
        if name in parameters and name not in kwargs:
            kwargs[name] = value

    if task_config is not None:
        parameter = config_parameter
        if parameter is None:
            candidates = [
                name for name in ("task_json_path", "local_dir") if name in parameters
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "task_config requires a factory with exactly one of "
                    "'task_json_path' or 'local_dir', or a built-in preset"
                )
            parameter = candidates[0]
        kwargs.setdefault(parameter, str(task_config))
    return kwargs


def load_environment_group(
    environment: EnvironmentName | str,
    *,
    env_kwargs: Mapping[str, Any] | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Environment]:
    """Load a task-keyed group of Corral environments.

    `environment` selects one of the registered factories. `env_kwargs`
    configures that environment; the common `level`, `subtasks`,
    `task_config`, and `work_dir` keys are normalized across factories,
    while any remaining keys are forwarded directly to the selected factory.
    """
    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else _repository_root()
    )
    environment_name = normalise_environment_name(environment)
    preset = ENVIRONMENT_PRESETS.get(environment_name)
    if preset is None:
        known = ", ".join(ENVIRONMENT_NAMES)
        raise ValueError(f"unknown environment {environment!r}; choose one of {known}")

    source_dir = root / preset.source_dir
    if source_dir.is_dir():
        sys.path.insert(0, str(source_dir))

    options = dict(env_kwargs or {})
    level = options.pop("level", 1)
    subtasks = options.pop("subtasks", False)
    task_config = options.pop("task_config", None)
    work_dir = options.pop("work_dir", None)

    workspace = (
        Path(
            work_dir
            or os.environ.get("CORRAL_WORK_DIR", root / ".corral" / "workspaces")
        )
        .expanduser()
        .resolve()
    )
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["CORRAL_WORK_DIR"] = str(workspace)

    environment_factory = _load_factory(preset.factory)
    config_path: Path | None = None
    config_parameter = preset.config_parameter
    if task_config is not None:
        config_path = Path(task_config).expanduser().resolve()
    elif preset.config_parameter is not None:
        config_path = _default_task_config(
            root,
            environment_name,
            level=level,
            subtasks=subtasks,
        )
        if preset.default_config_file is not None:
            config_path /= preset.default_config_file
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"task configuration not found: {config_path}")

    kwargs = _factory_kwargs(
        environment_factory,
        initial=options,
        work_dir=workspace,
        level=level,
        subtasks=subtasks,
        config_parameter=config_parameter,
        task_config=config_path,
    )
    loaded = environment_factory(**kwargs)
    if isinstance(loaded, Environment):
        environments = {loaded.task_id: loaded}
    elif isinstance(loaded, Mapping):
        environments = dict(loaded)
    else:
        raise TypeError(
            f"environment factory {preset.factory!r} returned "
            f"{type(loaded).__name__}, expected Environment or a mapping"
        )

    if not environments:
        raise ValueError("environment factory returned no tasks")
    invalid = {
        task_id: type(task_environment).__name__
        for task_id, task_environment in environments.items()
        if not isinstance(task_id, str) or not isinstance(task_environment, Environment)
    }
    if invalid:
        raise TypeError(f"environment factory returned invalid task entries: {invalid}")
    return environments


__all__ = [
    "ENVIRONMENT_NAMES",
    "ENVIRONMENT_PRESETS",
    "EnvironmentName",
    "EnvironmentPreset",
    "load_environment_group",
    "normalise_environment_name",
]
