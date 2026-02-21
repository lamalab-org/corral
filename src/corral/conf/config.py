"""Structured configs for Hydra-based launching.

These dataclasses define the full configuration schema for Corral benchmarks.
They are used by Hydra for type-safe, composable configuration and can also
be used programmatically outside of Hydra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for the agent.

    Attributes:
        _target_: Fully-qualified class path for ``hydra.utils.instantiate``.
        model: LLM model identifier (e.g. ``openai/gpt-4o``).
        max_iterations: Maximum LLM calls per trial.
        temperature: Sampling temperature.
        api_endpoint: Optional custom API endpoint URL.
        system_prompt: Optional override for the system prompt.
        user_prompt: Optional override for the user prompt.
        extractor_prompt: Optional override for the extractor prompt.
        surrender_prompt: Optional override for the surrender prompt.
        extra_kwargs: Arbitrary extra keyword arguments forwarded to the agent constructor.
    """

    _target_: str = "corral.agents.tool_calling.ToolCallingAgent"
    model: str = "openai/gpt-4o"
    max_iterations: int = 20
    temperature: float = 1.0
    api_endpoint: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    extractor_prompt: str | None = None
    surrender_prompt: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    """Configuration for the benchmark runner (CorralRunner).

    Attributes:
        base_url: URL of the running environment server.
        trials_per_task: Number of independent trials per task.
        k_values: List of k values for pass@k metrics.
        checkpoint_dir: Directory for trial checkpoints.
        enable_surrender: Allow agents to surrender on unsolvable tasks.
        verbose: Enable verbose logging during trials.
        tool_verbosity: Verbosity level for tool output (``brief`` or ``full``).
        metrics_file: Optional path to a Python file with custom ``Metric`` definitions.
        output: Optional path to write the final results JSON.
    """

    base_url: str = "http://localhost:8000"
    trials_per_task: int = 5
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    checkpoint_dir: str = "./benchmark_checkpoints"
    enable_surrender: bool = False
    verbose: bool = True
    tool_verbosity: str = "brief"
    metrics_file: str | None = None
    output: str | None = None


@dataclass
class DockerConfig:
    """Configuration for Docker-based execution.

    Attributes:
        image: Docker image for the benchmark environment.
        agent_image: Docker image for the agent runner (None = GHCR default).
        port: Preferred port for the environment server (default 8000).
        auto_find_port: Automatically find a free port if ``port`` is busy.
        host: Host address the environment binds to inside the container.
        network: Docker network name.
        env_container_name: Name for the environment Docker container.
        agent_container_name: Name for the agent Docker container.
        detach: Run containers in the background.
        output_dir: Host directory to mount for results.
        env_args: Extra arguments forwarded to the environment container.
    """

    image: str = "ghcr.io/lamalab-org/corral-materials:latest"
    agent_image: str | None = None
    port: int = 8000
    auto_find_port: bool = True
    host: str = "0.0.0.0"
    network: str = "corral-network"
    env_container_name: str = "corral-env"
    agent_container_name: str = "corral-agent"
    detach: bool = False
    output_dir: str | None = None
    env_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class TasksConfig:
    """Configuration for task selection.

    Attributes:
        ids: List of task IDs to run. Empty list means all tasks.
    """

    ids: list[str] = field(default_factory=list)


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging.

    Attributes:
        enabled: Whether to enable W&B logging.
        project: W&B project name.
        entity: W&B entity (team or user).
        tags: Tags to attach to the run.
    """

    enabled: bool = False
    project: str = "corral-benchmark"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class CorralConfig:
    """Top-level configuration for a Corral benchmark run.

    Composes all sub-configs. This is the root config node used by Hydra.
    """

    agent: AgentConfig = field(default_factory=AgentConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Execution mode: "local" runs agent directly, "docker" uses containers
    mode: str = "local"
