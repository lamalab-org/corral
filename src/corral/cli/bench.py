"""Benchmark command for running agents against environments."""

from __future__ import annotations

import inspect
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

bench_app = typer.Typer(no_args_is_help=True)
console = Console()


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    content = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(content) or {}
    elif path.suffix == ".json":
        return json.loads(content)
    else:
        # Try YAML first, then JSON
        try:
            return yaml.safe_load(content) or {}
        except Exception:
            return json.loads(content)


def parse_json_or_file(value: str | None) -> dict[str, Any] | None:
    """Parse a JSON string or @file reference into a dict."""
    if not value:
        return None
    value = value.strip()
    if value.startswith("@"):
        path = Path(value[1:])
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(content)
        return json.loads(content)
    return json.loads(value)


def merge_settings(
    cli_args: dict[str, Any],
    config: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Merge settings with clear precedence: CLI > Config > Defaults."""
    result = defaults.copy()

    # Apply config file values (overrides defaults)
    for key, value in config.items():
        if value is not None:
            result[key] = value

    # Apply CLI arguments (overrides everything)
    for key, value in cli_args.items():
        if value is not None:
            result[key] = value

    return result


DEFAULTS = {
    "image": "ghcr.io/lamalab-org/corral-materials:latest",
    "agent_image": None,  # Uses GHCR default from docker_runner
    "agent": "ReActAgent",
    "model": "claude-sonnet-4-5-20250929",
    "trials": 5,
    "max_iterations": 20,
    "temperature": 1.0,
    "tasks": None,
    "output": None,
    "detach": False,
    "verbose": True,
}


@bench_app.command("run")
def run_benchmark(
    # Config file
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML/JSON config file. CLI args override config values.",
    ),
    # Docker environment image
    image: str | None = typer.Option(
        None,
        "--image",
        "-i",
        help="Docker image for the environment",
    ),
    # Docker agent runner image
    agent_image: str | None = typer.Option(
        None,
        "--agent-image",
        help="Docker image for the agent runner. Defaults to GHCR image. "
        "Use 'local' for locally built image, or specify a custom image.",
    ),
    # Agent and benchmark settings
    agent: str | None = typer.Option(
        None,
        "--agent",
        "-a",
        help="Agent class to use for benchmarking",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model to use",
    ),
    trials: int | None = typer.Option(
        None,
        "--trials",
        "-t",
        help="Number of trials per task",
    ),
    tasks: str | None = typer.Option(
        None,
        "--tasks",
        help="Comma-separated task IDs to run (default: all)",
    ),
    max_iterations: int | None = typer.Option(
        None,
        "--max-iterations",
        help="Maximum iterations per trial",
    ),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        help="LLM temperature",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results",
    ),
    detach: bool = typer.Option(
        False,
        "--detach",
        "-d",
        help="Run in detached mode (background)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    # Flexible kwargs for advanced users
    agent_kwargs: str | None = typer.Option(
        None,
        "--agent-kwargs",
        help="JSON string or @file.yaml with extra agent parameters",
    ),
    runner_kwargs: str | None = typer.Option(
        None,
        "--runner-kwargs",
        help="JSON string or @file.yaml with extra runner parameters",
    ),
):
    """Run agent benchmarks against a Corral environment."""
    # This must be imported here to avoid circular imports
    from corral.cli.docker_runner import DockerBenchmarkRunner

    # Load config file if provided
    file_config = load_config(config) if config else {}

    # Collect CLI arguments (only non-None values will override)
    cli_args = {
        "image": image,
        "agent_image": agent_image,
        "agent": agent,
        "model": model,
        "trials": trials,
        "tasks": tasks,
        "max_iterations": max_iterations,
        "temperature": temperature,
        "output": output,
        "detach": detach,
        "verbose": verbose,
    }

    # Merge with precedence: CLI > Config > Defaults
    settings = merge_settings(cli_args, file_config, DEFAULTS)

    # Parse extra kwargs
    extra_agent_kwargs = parse_json_or_file(agent_kwargs) or {}
    extra_runner_kwargs = parse_json_or_file(runner_kwargs) or {}

    # Merge agent_kwargs from config file too
    if "agent_kwargs" in file_config:
        extra_agent_kwargs = {**file_config["agent_kwargs"], **extra_agent_kwargs}
    if "runner_kwargs" in file_config:
        extra_runner_kwargs = {**file_config["runner_kwargs"], **extra_runner_kwargs}

    # Display final configuration
    _show_config(settings, extra_agent_kwargs, extra_runner_kwargs)

    # Run benchmark
    runner = DockerBenchmarkRunner()

    try:
        runner.run(
            env_image=settings["image"],
            agent_class=settings["agent"],
            model=settings["model"],
            trials_per_task=settings["trials"],
            task_ids=settings["tasks"],
            max_iterations=settings["max_iterations"],
            temperature=settings["temperature"],
            output_file=settings["output"],
            detach=settings["detach"],
            verbose=settings["verbose"],
            agent_kwargs=extra_agent_kwargs,
            runner_kwargs=extra_runner_kwargs,
            agent_image=settings["agent_image"],
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@bench_app.command("stop")
def stop_benchmark():
    """Stop running benchmark containers."""
    # Import here to avoid circular imports
    from corral.cli.docker_runner import DockerBenchmarkRunner

    runner = DockerBenchmarkRunner()
    runner.stop()
    console.print("[green]Benchmark containers stopped.[/green]")


@bench_app.command("env")
def run_environment_only(
    image: str = typer.Option(
        "ghcr.io/lamalab-org/corral-materials:latest",
        "--image",
        "-i",
        help="Docker image for the environment",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to expose the environment on",
    ),
    detach: bool = typer.Option(
        False,
        "--detach",
        "-d",
        help="Run in detached mode (background)",
    ),
):
    """Run ONLY the environment container (no agent)."""
    # Avoid circular imports
    from corral.cli.docker_runner import DockerBenchmarkRunner

    console.print(f"[bold]Starting environment only:[/bold] {image}")
    console.print(f"[dim]Exposed on port {port}[/dim]\n")

    runner = DockerBenchmarkRunner()

    try:
        runner.run_environment_only(
            env_image=image,
            port=port,
            detach=detach,
        )

        if detach:
            console.print(
                f"\n[green]Environment running at:[/green] http://localhost:{port}"
            )
            console.print("[dim]Use 'corral bench stop' to stop the environment[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@bench_app.command("agent")
def run_agent_only(
    base_url: str = typer.Option(
        "http://localhost:8000",
        "--base-url",
        "-u",
        help="URL of the running environment server",
    ),
    agent: str = typer.Option(
        "ReActAgent",
        "--agent",
        "-a",
        help="Agent class to use",
    ),
    model: str = typer.Option(
        "claude-3-5-sonnet-20241022",
        "--model",
        "-m",
        help="LLM model to use",
    ),
    trials: int = typer.Option(
        5,
        "--trials",
        "-t",
        help="Number of trials per task",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
    agent_kwargs: str | None = typer.Option(
        None,
        "--agent-kwargs",
        help="JSON string or @file with extra agent parameters",
    ),
):
    """Run ONLY the agent against an already-running environment."""
    # Avoid circular imports
    from corral.cli.docker_runner import DockerBenchmarkRunner

    console.print(f"[bold]Running agent against:[/bold] {base_url}")
    console.print(f"[dim]Agent: {agent}, Model: {model}[/dim]\n")

    # Load config if provided
    file_config = load_config(config) if config else {}
    extra_kwargs = parse_json_or_file(agent_kwargs) or {}

    # Merge config file agent_kwargs
    if "agent_kwargs" in file_config:
        extra_kwargs = {**file_config["agent_kwargs"], **extra_kwargs}

    runner = DockerBenchmarkRunner()

    try:
        runner.run_agent_only(
            base_url=base_url,
            agent_class=agent,
            model=model,
            trials_per_task=trials,
            agent_kwargs=extra_kwargs,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@bench_app.command("debug")
def debug_mode(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    service: str = typer.Option(
        "agent", "--service", "-s", help="Service to show logs for (agent/environment)"
    ),
):
    """Debug mode: show logs from benchmark containers."""
    # Avoid circular imports
    from corral.cli.docker_runner import DockerBenchmarkRunner

    runner = DockerBenchmarkRunner()
    runner.debug(service=service, follow=follow)


@bench_app.command("list-agents")
def list_agents():
    """List available agent classes."""
    table = Table(title="Available Agent Classes")
    table.add_column("Agent", style="cyan")
    table.add_column("Description", style="white")

    try:
        agent_mod = import_module("corral.agents")
        for attr_name in dir(agent_mod):
            attr = getattr(agent_mod, attr_name)
            if inspect.isclass(attr):
                mro_names = [c.__name__ for c in getattr(attr, "__mro__", [])]
                # Exclude BaseAgent itself (it's abstract) but include its subclasses
                if (
                    "BaseAgent" in mro_names
                    and attr.__name__ != "BaseAgent"
                    and attr.__module__.startswith("corral.agents")
                ):
                    doc = attr.__doc__ or "No description"
                    table.add_row(attr.__name__, doc.split("\n")[0])
    except Exception:
        pass

    console.print(table)


def _show_config(settings: dict, agent_kwargs: dict, runner_kwargs: dict):
    """Display benchmark configuration."""
    table = Table(title="Benchmark Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Environment Image", settings["image"])
    table.add_row("Agent Class", settings["agent"])
    table.add_row("Model", settings["model"])
    table.add_row("Trials per Task", str(settings["trials"]))
    table.add_row("Max Iterations", str(settings["max_iterations"]))
    table.add_row("Temperature", str(settings["temperature"]))
    if settings.get("tasks"):
        table.add_row("Task IDs", settings["tasks"])
    if agent_kwargs:
        table.add_row("Agent Kwargs", str(agent_kwargs))
    if runner_kwargs:
        table.add_row("Runner Kwargs", str(runner_kwargs))

    console.print(table)
    console.print()
