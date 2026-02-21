"""Docker-based benchmark runner."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import docker
import requests
from docker.errors import NotFound
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from docker.models.containers import Container

console = Console()


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.

    Args:
        start_port: The preferred port to start searching from.
        max_attempts: Maximum number of ports to try.

    Returns:
        An available port number.

    Raises:
        RuntimeError: If no available port is found within max_attempts.
    """
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"Could not find an available port in range {start_port}-{start_port + max_attempts}"
    )


# Default GHCR images
DEFAULT_AGENT_IMAGE = "ghcr.io/lamalab-org/corral-agent-runner:latest"
DEFAULT_ENV_BASE_IMAGE = "ghcr.io/lamalab-org/corral-envs-base:latest"


class DockerBenchmarkRunner:
    """Runs benchmarks using Docker containers."""

    DEFAULT_RESULTS_DIR = "/opt/corral-workspace/results"

    def __init__(
        self,
        network_name: str = "corral-network",
        env_container_name: str = "corral-env",
        agent_container_name: str = "corral-agent",
    ):
        self.client = docker.from_env()
        self.NETWORK_NAME = network_name
        self.ENV_CONTAINER_NAME = env_container_name
        self.AGENT_CONTAINER_NAME = agent_container_name

    def run(
        self,
        env_image: str,
        agent_class: str = "ReActAgent",
        model: str = "claude-sonnet-4-5-20250929",
        trials_per_task: int = 5,
        task_ids: str | None = None,
        max_iterations: int = 10,
        temperature: float = 0.0,
        output_file: str | None = None,
        output_dir: str | None = None,
        detach: bool = False,
        verbose: bool = False,
        agent_kwargs: dict[str, Any] | None = None,
        runner_kwargs: dict[str, Any] | None = None,
        agent_image: str | None = None,
        env_args: dict[str, Any] | None = None,
        metrics_file: str | None = None,
        port: int = 8000,
        auto_find_port: bool = True,
        host: str = "0.0.0.0",
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
            output_dir: Directory to save results. Defaults to current working directory.
                       This directory will be mounted into the container.
            detach: Run in detached mode.
            verbose: Enable verbose output.
            agent_kwargs: Extra agent parameters.
            runner_kwargs: Extra runner parameters.
            agent_image: Docker image for the agent runner. Defaults to GHCR image.
                        Use 'local' to use a locally built image named 'corral-agent-runner:latest'.
            env_args: Environment-specific arguments passed as JSON to the container.
                     These are converted to CLI arguments by the entrypoint script.
            metrics_file: Path to Python file containing custom metrics.
                         Will be mounted into the container.
            port: Preferred port for the environment server (default 8000).
            auto_find_port: Automatically find a free port if ``port`` is busy.
            host: Host address the environment binds to inside the container.
        """
        # Resolve output directory (default to cwd — should already be absolute from CLI)
        results_host_path = Path(output_dir or Path.cwd()) / "corral-results"
        results_host_path.mkdir(parents=True, exist_ok=True)

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
            env_container, env_port = self._start_environment(
                env_image,
                port=port,
                auto_find_port=auto_find_port,
                env_args=env_args,
                host=host,
            )
            progress.update(
                task,
                description=f"[green]✓[/green] Environment started on port {env_port}",
            )

            # 3. Wait for environment to be healthy
            task = progress.add_task(
                "Waiting for environment to be ready...", total=None
            )
            self._wait_for_healthy(env_container, port=env_port)
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
                results_host_path=results_host_path,
                metrics_file=metrics_file,
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
                # List result files written to the host directory
                saved = sorted(results_host_path.glob("*.json"))
                if saved:
                    console.print(
                        f"\n[cyan]Results saved to:[/cyan] {results_host_path}"
                    )
                    for f in saved:
                        console.print(f"  [dim]→[/dim] {f.name}")
                else:
                    console.print(
                        f"\n[yellow]No result files found in {results_host_path}[/yellow]"
                    )
            else:
                console.print(
                    f"\n[bold red]Benchmark failed with exit code {exit_code}[/bold red]"
                )

            # Cleanup
            self.stop()

    def build_agent_runner(
        self,
        tag: str = "corral-agent-runner:latest",
        source_dir: str | None = None,
    ) -> None:
        """Build the agent runner Docker image from local source.

        This is the recommended approach for development so that the image
        reflects the locally installed version of corral rather than the
        published GitHub release.

        Args:
            tag: Image tag to assign to the built image.
            source_dir: Path to the corral source tree (the directory that
                contains ``pyproject.toml``).  Defaults to the directory two
                levels above this file (i.e. the repo root).
        """
        import subprocess

        # Locate the Dockerfile and the build context (repo root).
        # The Dockerfile uses BuildKit heredoc syntax (# syntax=docker/dockerfile:1),
        # so we invoke the docker CLI directly (which uses BuildKit by default)
        # rather than the Python SDK's legacy builder.
        dockerfile_path = (
            Path(__file__).parent.parent.parent.parent
            / ".docker"
            / "corral-agent-runner"
            / "Dockerfile"
        )
        if not dockerfile_path.exists():
            raise FileNotFoundError(
                f"Agent runner Dockerfile not found at {dockerfile_path}. "
                "Make sure you are running from the corral repository."
            )

        build_context = (
            Path(source_dir) if source_dir else dockerfile_path.parent.parent.parent
        )

        console.print(f"[cyan]Building agent runner image:[/cyan] {tag}")
        console.print(f"[dim]Context: {build_context}[/dim]")
        console.print("[dim]CORRAL_SOURCE=local (installing from local source)[/dim]\n")

        try:
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--build-arg",
                    "CORRAL_SOURCE=local",
                    "--file",
                    str(dockerfile_path),
                    "--tag",
                    tag,
                    str(build_context),
                ],
                check=True,
            )
            console.print(f"\n[bold green]Image built successfully:[/bold green] {tag}")
        except subprocess.CalledProcessError as exc:
            console.print(
                f"[bold red]Build failed with exit code {exc.returncode}[/bold red]"
            )
            raise

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

    def _start_environment(
        self,
        image: str,
        port: int = 8000,
        auto_find_port: bool = True,
        env_args: dict[str, Any] | None = None,
        host: str = "0.0.0.0",
    ) -> tuple[Container, int]:
        """Start the environment container.

        Args:
            image: Docker image for the environment.
            port: Preferred port to use.
            auto_find_port: If True, automatically find an available port if the
                preferred port is busy.

        Returns:
            A tuple of (container, actual_port) where actual_port is the port
            the environment is running on.
        """
        # Stop existing container if running
        try:
            old = self.client.containers.get(self.ENV_CONTAINER_NAME)
            old.stop()
            old.remove()
        except NotFound:
            pass

        # Find available port if requested
        actual_port = port
        if auto_find_port:
            actual_port = find_available_port(port)
            if actual_port != port:
                console.print(
                    f"[yellow]Port {port} is busy, using port {actual_port} instead[/yellow]"
                )

        # Store the port for agent connection
        self._env_port = actual_port

        # Build environment variables
        environment = {
            "CORRAL_HOST": host,
            "CORRAL_PORT": str(actual_port),
        }

        # Add environment-specific args as JSON
        if env_args:
            environment["CORRAL_ENV_ARGS"] = json.dumps(env_args)

        # Pass through any LiteLLM-supported API keys from environment
        for key in os.environ:
            if key.endswith("_API_KEY"):
                environment[key] = os.environ[key]

        container = self.client.containers.run(
            image,
            name=self.ENV_CONTAINER_NAME,
            network=self.NETWORK_NAME,
            environment=environment,
            ports={f"{actual_port}/tcp": actual_port},
            detach=True,
        )
        return container, actual_port

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
        results_host_path: Path | None = None,
        metrics_file: str | None = None,
    ) -> Container:
        """Start the agent runner container.

        Args:
            agent_image: Docker image to use for the agent runner.
            results_host_path: Host path to mount for results persistence.
            metrics_file: Path to Python file containing custom metrics.
        """
        # Stop existing container if running
        try:
            old = self.client.containers.get(self.AGENT_CONTAINER_NAME)
            old.stop()
            old.remove()
        except NotFound:
            pass

        env = {
            "BASE_URL": f"http://{self.ENV_CONTAINER_NAME}:{getattr(self, '_env_port', 8000)}",
            "AGENT_CLASS": agent_class,
            "MODEL": model,
            "TRIALS_PER_TASK": str(trials_per_task),
            "MAX_ITERATIONS": str(max_iterations),
            "TEMPERATURE": str(temperature),
            "VERBOSE": str(verbose).lower(),
            "RESULTS_DIR": self.DEFAULT_RESULTS_DIR,
            # Automatically clip k_values to the number of trials so pass@k
            # metrics don't request more trials than were run.
            "K_VALUES": ",".join(str(k) for k in range(1, trials_per_task + 1)),
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

        # Configure volumes for results persistence
        volumes = {}
        if results_host_path:
            volumes[str(results_host_path.absolute())] = {
                "bind": self.DEFAULT_RESULTS_DIR,
                "mode": "rw",
            }

        # Mount custom metrics file if provided
        container_metrics_path = "/opt/corral-workspace/custom_metrics.py"
        if metrics_file:
            metrics_path = Path(metrics_file).resolve()
            if not metrics_path.exists():
                raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
            volumes[str(metrics_path)] = {
                "bind": container_metrics_path,
                "mode": "ro",
            }
            env["METRICS_FILE"] = container_metrics_path
            console.print(f"[cyan]Custom metrics file:[/cyan] {metrics_path}")

        return self.client.containers.run(
            agent_image,
            name=self.AGENT_CONTAINER_NAME,
            network=self.NETWORK_NAME,
            environment=env,
            volumes=volumes if volumes else None,
            detach=True,
        )

    def run_environment_only(
        self,
        env_image: str,
        port: int = 8000,
        detach: bool = False,
        env_args: dict[str, Any] | None = None,
    ):
        """Run only the environment container (no agent).

        Useful for development - run env in Docker, agent locally.

        Args:
            env_image: Docker image for the environment.
            port: Port to expose the environment on.
            detach: Run in detached mode.
            env_args: Environment-specific arguments passed as JSON to the container.
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
            env_container, actual_port = self._start_environment(
                env_image, port=port, env_args=env_args
            )
            progress.update(
                task,
                description=f"[green]✓[/green] Environment started on port {actual_port}",
            )

            task = progress.add_task(
                "Waiting for environment to be ready...", total=None
            )
            self._wait_for_healthy(env_container, port=actual_port)
            progress.update(task, description="[green]✓[/green] Environment healthy")

        console.print(
            f"\n[bold green]Environment running at http://localhost:{actual_port}[/bold green]\n"
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
        model: str = "claude-sonnet-4-5-20250929",
        trials_per_task: int = 5,
        task_ids: str | None = None,
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

        # Parse task IDs if provided
        parsed_task_ids = [t.strip() for t in task_ids.split(",")] if task_ids else None

        # Run benchmark
        runner = CorralRunner(router, agent)
        console.print(f"Running {agent_class} with {model}...\n")

        result = runner.bench(trials_per_task=trials_per_task, task_ids=parsed_task_ids)
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
