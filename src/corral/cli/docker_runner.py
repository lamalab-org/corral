"""Docker-based benchmark runner."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

import docker
import requests
from docker.errors import NotFound
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from docker.models.containers import Container

console = Console()


# Default GHCR images
DEFAULT_AGENT_IMAGE = "ghcr.io/lamalab-org/corral-agent-runner:latest"
DEFAULT_ENV_BASE_IMAGE = "ghcr.io/lamalab-org/corral-envs-base:latest"


class DockerBenchmarkRunner:
    """Runs benchmarks using Docker containers."""

    NETWORK_NAME = "corral-network"
    ENV_CONTAINER_NAME = "corral-env"
    AGENT_CONTAINER_NAME = "corral-agent"

    def __init__(self):
        self.client = docker.from_env()

    def run(
        self,
        env_image: str,
        agent_class: str = "ReActAgent",
        model: str = "claude-3-5-sonnet-20241022",
        trials_per_task: int = 5,
        task_ids: str | None = None,
        max_iterations: int = 10,
        temperature: float = 0.0,
        output_file: str | None = None,
        detach: bool = False,
        verbose: bool = False,
        agent_kwargs: dict[str, Any] | None = None,
        runner_kwargs: dict[str, Any] | None = None,
        agent_image: str | None = None,
    ):
        """Run benchmark with two-container architecture.

        Args:
            env_image: Docker image for the environment.
            agent_class: Agent class to use for benchmarking.
            model: LLM model to use.
            trials_per_task: Number of trials per task.
            task_ids: Comma-separated task IDs to run.
            max_iterations: Maximum iterations per trial.
            temperature: LLM temperature.
            output_file: Output file for results.
            detach: Run in detached mode.
            verbose: Enable verbose output.
            agent_kwargs: Extra agent parameters.
            runner_kwargs: Extra runner parameters.
            agent_image: Docker image for the agent runner. Defaults to GHCR image.
                        Use 'local' to use a locally built image named 'corral-agent-runner:latest'.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # 1. Create network
            task = progress.add_task("Creating Docker network...", total=None)
            self._ensure_network()
            progress.update(task, description="[green]✓[/green] Network ready")

            # 2. Start environment container
            task = progress.add_task("Starting environment container...", total=None)
            env_container = self._start_environment(env_image)
            progress.update(task, description="[green]✓[/green] Environment started")

            # 3. Wait for environment to be healthy
            task = progress.add_task(
                "Waiting for environment to be ready...", total=None
            )
            self._wait_for_healthy(env_container)
            progress.update(task, description="[green]✓[/green] Environment healthy")

            # 4. Resolve agent image
            resolved_agent_image = self._resolve_agent_image(agent_image)

            # 5. Start agent container
            task = progress.add_task(
                f"Starting agent runner ({resolved_agent_image})...", total=None
            )
            agent_container = self._start_agent(
                agent_class=agent_class,
                model=model,
                trials_per_task=trials_per_task,
                task_ids=task_ids,
                max_iterations=max_iterations,
                temperature=temperature,
                output_file=output_file,
                verbose=verbose,
                agent_kwargs=agent_kwargs,
                runner_kwargs=runner_kwargs,
                agent_image=resolved_agent_image,
            )
            progress.update(task, description="[green]✓[/green] Agent started")

        console.print("\n[bold green]Benchmark running![/bold green]\n")

        if detach:
            console.print(
                "Running in detached mode. Use [cyan]corral bench debug[/cyan] to view output."
            )
            console.print("Use [cyan]corral bench stop[/cyan] to stop the benchmark.")
        else:
            # Stream logs
            console.print("[dim]─" * 60 + "[/dim]")
            for log in agent_container.logs(stream=True, follow=True):
                console.print(log.decode("utf-8"), end="")

            # Wait for completion
            result = agent_container.wait()
            exit_code = result.get("StatusCode", 1)

            if exit_code == 0:
                console.print(
                    "\n[bold green]Benchmark completed successfully![/bold green]"
                )
            else:
                console.print(
                    f"\n[bold red]Benchmark failed with exit code {exit_code}[/bold red]"
                )

            # Cleanup
            self.stop()

    def _resolve_agent_image(self, agent_image: str | None) -> str:
        """Resolve the agent image to use.

        Args:
            agent_image: User-specified image. Can be:
                - None: Use default GHCR image
                - 'local': Use locally built 'corral-agent-runner:latest'
                - Any other string: Use as-is (custom image)

        Returns:
            The resolved Docker image name.
        """
        if agent_image is None:
            return DEFAULT_AGENT_IMAGE
        if agent_image.lower() == "local":
            return "corral-agent-runner:latest"
        return agent_image

    def _ensure_network(self):
        """Create Docker network if it doesn't exist."""
        try:
            self.client.networks.get(self.NETWORK_NAME)
        except NotFound:
            self.client.networks.create(self.NETWORK_NAME, driver="bridge")

    def _start_environment(self, image: str, port: int = 8000) -> Container:
        """Start the environment container."""
        # Stop existing container if running
        try:
            old = self.client.containers.get(self.ENV_CONTAINER_NAME)
            old.stop()
            old.remove()
        except NotFound:
            pass

        return self.client.containers.run(
            image,
            name=self.ENV_CONTAINER_NAME,
            network=self.NETWORK_NAME,
            environment={
                "CORRAL_HOST": "0.0.0.0",
                "CORRAL_PORT": str(port),
            },
            ports={f"{port}/tcp": port},
            detach=True,
        )

    def _start_agent(
        self,
        agent_class: str,
        model: str,
        trials_per_task: int,
        task_ids: str | None,
        max_iterations: int,
        temperature: float,
        output_file: str | None,
        verbose: bool,
        agent_kwargs: dict[str, Any] | None = None,
        runner_kwargs: dict[str, Any] | None = None,
        agent_image: str = DEFAULT_AGENT_IMAGE,
    ) -> Container:
        """Start the agent runner container.

        Args:
            agent_image: Docker image to use for the agent runner.
        """
        # Stop existing container if running
        try:
            old = self.client.containers.get(self.AGENT_CONTAINER_NAME)
            old.stop()
            old.remove()
        except NotFound:
            pass

        env = {
            "BASE_URL": f"http://{self.ENV_CONTAINER_NAME}:8000",
            "AGENT_CLASS": agent_class,
            "MODEL": model,
            "TRIALS_PER_TASK": str(trials_per_task),
            "MAX_ITERATIONS": str(max_iterations),
            "TEMPERATURE": str(temperature),
            "VERBOSE": str(verbose).lower(),
        }

        # Pass through any LiteLLM-supported API keys from environment
        for key in os.environ:
            if key.endswith("_API_KEY"):
                env[key] = os.environ[key]

        if task_ids:
            env["TASK_IDS"] = task_ids
        if output_file:
            env["RUN_NAME"] = output_file.replace(".json", "")
        if agent_kwargs:
            env["AGENT_KWARGS"] = json.dumps(agent_kwargs)
        if runner_kwargs:
            env["RUNNER_KWARGS"] = json.dumps(runner_kwargs)

        return self.client.containers.run(
            agent_image,
            name=self.AGENT_CONTAINER_NAME,
            network=self.NETWORK_NAME,
            environment=env,
            detach=True,
        )

    def run_environment_only(
        self,
        env_image: str,
        port: int = 8000,
        detach: bool = False,
    ):
        """Run only the environment container (no agent).

        Useful for development - run env in Docker, agent locally.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Creating Docker network...", total=None)
            self._ensure_network()
            progress.update(task, description="[green]✓[/green] Network ready")

            task = progress.add_task("Starting environment container...", total=None)
            env_container = self._start_environment(env_image, port=port)
            progress.update(task, description="[green]✓[/green] Environment started")

            task = progress.add_task(
                "Waiting for environment to be ready...", total=None
            )
            self._wait_for_healthy(env_container, port=port)
            progress.update(task, description="[green]✓[/green] Environment healthy")

        console.print(
            f"\n[bold green]Environment running at http://localhost:{port}[/bold green]\n"
        )

        if not detach:
            console.print("[dim]Press Ctrl+C to stop[/dim]")
            console.print("[dim]─" * 60 + "[/dim]")
            try:
                for log in env_container.logs(stream=True, follow=True):
                    console.print(log.decode("utf-8"), end="")
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping environment...[/yellow]")
                self.stop()

    def run_agent_only(
        self,
        base_url: str,
        agent_class: str = "ReActAgent",
        model: str = "claude-3-5-sonnet-20241022",
        trials_per_task: int = 5,
        agent_kwargs: dict[str, Any] | None = None,
    ):
        """Run agent locally against an already-running environment.

        No Docker for the agent - runs in current Python environment.
        """
        from corral import CorralRouter, CorralRunner
        from corral.agents import get_agent_class

        console.print(f"Connecting to environment at {base_url}...")

        router = CorralRouter(base_url)

        # Get agent class and instantiate
        agent_cls = get_agent_class(agent_class)
        agent_init_kwargs = agent_kwargs or {}
        agent_init_kwargs.setdefault("model", model)
        agent = agent_cls(**agent_init_kwargs)

        # Run benchmark
        runner = CorralRunner(router, agent)
        console.print(f"Running {agent_class} with {model}...\n")

        result = runner.bench(trials_per_task=trials_per_task)
        result.generate_report("corral_run.json")

        console.print("\n[bold green]Benchmark completed![/bold green]")

    def _wait_for_healthy(
        self, container: Container, port: int = 8000, timeout: int = 60
    ):
        """Wait for container to be healthy."""
        _ = container  # Keep for API consistency, health check uses localhost
        start = time.time()
        while time.time() - start < timeout:
            try:
                # Try to reach the health endpoint
                resp = requests.get(f"http://localhost:{port}/tasks", timeout=2)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2)

        raise TimeoutError("Environment container failed to become healthy")

    def stop(self):
        """Stop all benchmark containers."""
        for name in [self.AGENT_CONTAINER_NAME, self.ENV_CONTAINER_NAME]:
            try:
                container = self.client.containers.get(name)
                container.stop()
                container.remove()
            except NotFound:
                pass

    def debug(self, service: str = "agent", follow: bool = False):
        """Debug mode: show logs from a container."""
        name = (
            self.AGENT_CONTAINER_NAME if service == "agent" else self.ENV_CONTAINER_NAME
        )

        try:
            container = self.client.containers.get(name)
            if follow:
                for log in container.logs(stream=True, follow=True):
                    console.print(log.decode("utf-8"), end="")
            else:
                console.print(container.logs().decode("utf-8"))
        except NotFound:
            console.print(f"[red]Container '{name}' not found[/red]")
