"""Hydra-powered entry point for Corral benchmarks.

Usage examples
--------------
# Default run (local mode, tool_calling agent, default runner):
    corral-hydra

# Choose a different agent preset:
    corral-hydra agent=react

# Override individual values:
    corral-hydra agent.model=anthropic/claude-sonnet-4-5-20250929 runner.trials_per_task=10

# Multi-run sweep across agents and models:
    corral-hydra --multirun agent=react,tool_calling agent.model=openai/gpt-4o,anthropic/claude-sonnet-4-5-20250929

# Docker mode:
    corral-hydra mode=docker docker.image=ghcr.io/lamalab-org/corral-materials:latest

# Quick smoke test:
    corral-hydra runner=quick
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

console = Console()


def _show_resolved_config(cfg: DictConfig) -> None:
    """Pretty-print the resolved Hydra config."""
    table = Table(title="Hydra Benchmark Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Mode", cfg.mode)
    table.add_row("Agent", cfg.agent._target_)
    table.add_row("Model", cfg.agent.model)
    table.add_row("Max Iterations", str(cfg.agent.max_iterations))
    table.add_row("Temperature", str(cfg.agent.temperature))
    table.add_row("Trials per Task", str(cfg.runner.trials_per_task))
    table.add_row("k values", str(list(cfg.runner.k_values)))
    table.add_row("Enable Surrender", str(cfg.runner.enable_surrender))
    table.add_row("Verbose", str(cfg.runner.verbose))

    if cfg.tasks.ids:
        table.add_row("Tasks", ", ".join(cfg.tasks.ids))
    else:
        table.add_row("Tasks", "(all)")

    if cfg.runner.metrics_file:
        table.add_row("Metrics File", cfg.runner.metrics_file)

    if cfg.mode == "docker":
        table.add_row("Docker Image", cfg.docker.image)
        if cfg.docker.agent_image:
            table.add_row("Agent Image", cfg.docker.agent_image)
        if cfg.docker.output_dir:
            table.add_row("Output Dir", cfg.docker.output_dir)

    if cfg.wandb.enabled:
        table.add_row("W&B Project", cfg.wandb.project)

    console.print(table)
    console.print()


def _instantiate_agent(agent_cfg: DictConfig):
    """Instantiate the agent from its ``_target_`` and parameters.

    We do manual instantiation instead of ``hydra.utils.instantiate``
    because ``BaseAgent.__init__`` uses ``**kwargs`` and we want to forward
    ``extra_kwargs`` cleanly.
    """
    target = agent_cfg._target_
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    agent_cls = getattr(module, class_name)

    # Build the constructor kwargs
    init_kwargs: dict[str, Any] = {}
    for key in (
        "model",
        "max_iterations",
        "temperature",
        "api_endpoint",
        "system_prompt",
        "user_prompt",
        "extractor_prompt",
        "surrender_prompt",
    ):
        val = OmegaConf.select(agent_cfg, key, default=None)
        if val is not None:
            init_kwargs[key] = val

    # Merge extra_kwargs
    extra = (
        OmegaConf.to_container(agent_cfg.get("extra_kwargs", {}), resolve=True) or {}
    )
    init_kwargs.update(extra)

    return agent_cls(**init_kwargs)


def _load_custom_metrics(metrics_file: str | None):
    """Optionally load metrics from a user-provided Python file."""
    if not metrics_file:
        return None

    path = Path(metrics_file)
    if not path.exists():
        logger.warning(f"Metrics file not found: {metrics_file}")
        return None

    from corral.report.metrics.loader import load_metrics_from_file

    return load_metrics_from_file(str(path))


def _run_local(cfg: DictConfig) -> float | None:
    """Run benchmark locally (no Docker)."""
    from corral.report import CorralWandbLogger
    from corral.router.routes import CorralRouter
    from corral.run import CorralRunner

    # Build components
    interface = CorralRouter(base_url=cfg.runner.base_url)
    agent = _instantiate_agent(cfg.agent)
    metrics = _load_custom_metrics(cfg.runner.metrics_file)

    # Optional W&B logger
    wandb_logger = None
    if cfg.wandb.enabled:
        wandb_logger = CorralWandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            tags=list(cfg.wandb.get("tags", [])),
        )

    runner = CorralRunner(
        interface=interface,
        agent=agent,
        checkpoint_dir=cfg.runner.checkpoint_dir,
        logger=wandb_logger,
        enable_surrender=cfg.runner.enable_surrender,
        metrics=metrics,
    )

    task_ids = list(cfg.tasks.ids) if cfg.tasks.ids else None
    k_values = list(cfg.runner.k_values)

    result = runner.bench(
        task_ids=task_ids,
        trials_per_task=cfg.runner.trials_per_task,
        k_values=k_values,
        verbose=cfg.runner.verbose,
        tool_verbosity=cfg.runner.tool_verbosity,
    )

    # Return average score for Hydra sweeps (optimise target)
    return result.summary.get("average_score") if hasattr(result, "summary") else None


def _run_docker(cfg: DictConfig) -> None:
    """Run benchmark in Docker containers."""
    from corral.cli.docker_runner import DockerBenchmarkRunner

    # Map agent _target_ back to class name for dockerised execution
    agent_class_name = cfg.agent._target_.rsplit(".", 1)[1]

    # Collect agent extra kwargs
    extra_agent_kwargs: dict[str, Any] = {}
    for key in ("system_prompt", "user_prompt", "extractor_prompt", "surrender_prompt"):
        val = OmegaConf.select(cfg.agent, key, default=None)
        if val is not None:
            extra_agent_kwargs[key] = val
    extra = (
        OmegaConf.to_container(cfg.agent.get("extra_kwargs", {}), resolve=True) or {}
    )
    extra_agent_kwargs.update(extra)

    # Resolve task list
    task_ids_str = ",".join(cfg.tasks.ids) if cfg.tasks.ids else None

    # Resolve env_args
    env_args = (
        OmegaConf.to_container(cfg.docker.get("env_args", {}), resolve=True) or None
    )

    runner = DockerBenchmarkRunner()
    runner.run(
        env_image=cfg.docker.image,
        agent_class=agent_class_name,
        model=cfg.agent.model,
        trials_per_task=cfg.runner.trials_per_task,
        task_ids=task_ids_str,
        max_iterations=cfg.agent.max_iterations,
        temperature=cfg.agent.temperature,
        output_file=cfg.runner.output,
        output_dir=cfg.docker.output_dir,
        detach=cfg.docker.detach,
        verbose=cfg.runner.verbose,
        agent_kwargs=extra_agent_kwargs if extra_agent_kwargs else None,
        agent_image=cfg.docker.agent_image,
        env_args=env_args,
        metrics_file=cfg.runner.metrics_file,
    )


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

# We use ``version_base=None`` (Hydra 1.3+ behaviour) and register the
# structured configs via the ConfigStore so that Hydra can validate the YAML
# files against the dataclass schemas.


def _register_configs() -> None:
    """Register structured configs with Hydra's ConfigStore."""
    from hydra.core.config_store import ConfigStore

    from corral.conf.config import (
        AgentConfig,
        CorralConfig,
        DockerConfig,
        RunnerConfig,
    )

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=CorralConfig)
    cs.store(group="agent", name="base_agent", node=AgentConfig)
    cs.store(group="runner", name="base_runner", node=RunnerConfig)
    cs.store(group="docker", name="base_docker", node=DockerConfig)


_register_configs()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Corral Hydra launcher.

    Run benchmarks with composable, sweepable configuration.
    Hydra automatically creates a timestamped output directory for each run
    containing the full resolved config and any saved outputs.
    """
    logger.info(f"Hydra output directory: {hydra.utils.get_original_cwd()}")
    _show_resolved_config(cfg)

    # Save resolved config as YAML for reproducibility
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    logger.info(f"Resolved config:\n{resolved_yaml}")

    if cfg.mode == "docker":
        _run_docker(cfg)
        return None
    elif cfg.mode == "local":
        return _run_local(cfg)
    else:
        console.print(f"[red]Unknown mode:[/red] {cfg.mode}. Use 'local' or 'docker'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
