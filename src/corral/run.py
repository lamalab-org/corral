import json
import pickle
import re
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

from corral.agents import BaseAgent
from corral.agents.hooks import AgentHooks
from corral.agents.utils import LiteLLMMessage
from corral.report import (
    BenchmarkResult,
    CorralWandbLogger,
    TaskTrialResult,
    TaskTrialResults,
)
from corral.report.metrics import Metric, get_default_metrics
from corral.report.metrics.base import TaskMetric
from corral.report.metrics.registry import MetricRegistry
from corral.router import CorralRouter
from corral.types import BudgetExhaustedError


def create_session_id() -> str:
    """Create a unique session ID"""
    return f"session_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"


def sanitize_model_name(model_name: str, max_length: int = 50) -> str:
    """Sanitize model name for use in filenames.

    Replaces invalid filesystem characters with underscores and limits length.

    Args:
        model_name: The raw model name to sanitize
        max_length: Maximum length for the sanitized name (default: 50)

    Returns:
        Sanitized model name safe for use in filenames
    """
    # Replace invalid filesystem characters with underscores
    invalid_chars = r"[/\\:*?\"<>|\s\-]+"
    sanitized = re.sub(invalid_chars, "_", model_name)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("_")

    # Ensure we have a valid name
    if not sanitized:
        sanitized = "unknown_model"

    return sanitized


def validate_k_values(
    k_values: int | list[int] | None, trials_per_task: int
) -> list[int]:
    """Validate and normalize k_values."""
    if k_values is None:
        # Default to range 1 to min(5, trials_per_task)
        max_k = min(5, trials_per_task)
        return list(range(1, max_k + 1))
    elif isinstance(k_values, int):
        # Single int means max k, generate range 1 to k
        if k_values > trials_per_task:
            raise ValueError(
                f"k value ({k_values}) is greater than the number of trials ({trials_per_task})"
            )
        return list(range(1, k_values + 1))
    else:
        # List of k values - validate all are within bounds
        max_k = max(k_values)
        if max_k > trials_per_task:
            raise ValueError(
                f"k value ({max_k}) is greater than the number of trials ({trials_per_task})"
            )
        return sorted(k_values)


def initialize_task_results(task_ids: list[str]) -> dict[str, TaskTrialResults]:
    """Initialize empty task results"""
    return {task_id: TaskTrialResults(task_id=task_id) for task_id in task_ids}


def filter_incomplete_tasks(
    task_results: dict[str, TaskTrialResults], trials_per_task: int
) -> list[str]:
    """Get list of tasks that still need trials"""
    return [
        task_id
        for task_id, results in task_results.items()
        if len(results.trials) < trials_per_task
    ]


def get_score_from_state(interface: CorralRouter, task_id: str) -> float:
    """Retrieve current score from task state, defaulting to 0.0 if unavailable"""
    try:
        status = interface.get_task_status(task_id)
        return status.get("score", 0.0) or 0.0  # Handle None case
    except Exception as e:
        logger.warning(f"Failed to retrieve score from state: {e}")
        return 0.0


def exception_trial_result(
    task_id: str,
    trial_index: int,
    interface: CorralRouter,
    error: Exception,
    error_type: str,
    token_usage: dict[str, Any],
    surrendered: bool = False,
    messages: list[dict[str, Any]] | None = None,
) -> TaskTrialResult:
    """Create a TaskTrialResult when there is an exception during trial execution. Benchmark continues.

    Args:
        task_id: The task identifier
        trial_index: The trial index
        interface: Router interface to retrieve task state
        error: The exception that occurred
        error_type: Type of error (e.g., "Surrender Error", "Submission Error", "Agent Error")
        token_usage: Token usage statistics
        surrendered: Whether this was a surrender operation
        messages: Optional agent messages

    Returns:
        TaskTrialResult with score retrieved from state or 0.0 if unavailable
    """
    score = get_score_from_state(interface, task_id)
    return TaskTrialResult(
        task_id=task_id,
        trial_id=f"attempt_{trial_index + 1}",
        score=score,
        state={"error": str(error), "attempt": trial_index + 1},
        tool_statistics={"error": str(error)},
        messages=messages,
        duration=None,
        token_usage=token_usage,
        error_message=f"{error_type}: {error}",
        surrendered=surrendered,
    )


def execute_single_trial(
    task_id: str,
    trial_index: int,
    interface: CorralRouter,
    agent: BaseAgent,
    verbose: bool = False,
    tool_verbosity: str | None = None,
    configure_timeout: float | None = None,
    enable_surrender: bool = False,
) -> TaskTrialResult:
    """Execute a single trial - pure function."""
    trial_start_time = datetime.now(tz=timezone.utc)

    try:
        status = interface.configure_additional_apps(task_id, timeout=configure_timeout)
        logger.info(f"Task {task_id} additional apps/services configured: {status}")

        # Agent always returns (answer, messages, token_usage)
        answer, messages, token_usage = agent.run_agent(
            interface,
            task_id,
            verbose=verbose,
            tool_verbosity=tool_verbosity or "brief",
            enable_surrender=enable_surrender,
        )

        # Check if agent decided to surrender
        if answer == "SURRENDER":
            try:
                result = interface.surrender_task(task_id)
                result.token_usage = token_usage
                result.messages = messages
                trial_end_time = datetime.now(tz=timezone.utc)
                result.duration = (trial_end_time - trial_start_time).total_seconds()
                return result
            except Exception as surrender_error:
                trial_end_time = datetime.now(tz=timezone.utc)
                result = exception_trial_result(
                    task_id=task_id,
                    trial_index=trial_index,
                    interface=interface,
                    error=surrender_error,
                    error_type="Surrender Error",
                    token_usage=token_usage,
                    surrendered=True,
                    messages=messages,
                )
                result.duration = (trial_end_time - trial_start_time).total_seconds()
                return result

        # Submit answer
        try:
            result = interface.submit_answer(task_id, answer)
            result.token_usage = token_usage
            result.messages = messages
            trial_end_time = datetime.now(tz=timezone.utc)
            result.duration = (trial_end_time - trial_start_time).total_seconds()
            return result
        except Exception as submit_error:
            trial_end_time = datetime.now(tz=timezone.utc)
            result = exception_trial_result(
                task_id=task_id,
                trial_index=trial_index,
                interface=interface,
                error=submit_error,
                error_type="Submission Error",
                token_usage=token_usage,
                messages=messages,
            )
            result.duration = (trial_end_time - trial_start_time).total_seconds()
            return result
    except BudgetExhaustedError:
        # Re-raise to stop the benchmark immediately
        # When an error with the llm running out of credits occurs
        raise
    except Exception as agent_error:
        trial_end_time = datetime.now(tz=timezone.utc)
        result = exception_trial_result(
            task_id=task_id,
            trial_index=trial_index,
            interface=interface,
            error=agent_error,
            error_type="Agent Error",
            token_usage=agent.get_total_token_usage(),
        )
        result.duration = (trial_end_time - trial_start_time).total_seconds()
        return result


def run_independent_trials(
    task_ids: list[str],
    trials_per_task: int,
    task_results: dict[str, TaskTrialResults],
    trial_executor: Callable[[str, int], TaskTrialResult],
    checkpoint_saver: Callable[[dict[str, TaskTrialResults]], None],
) -> None:
    """Run trials independently for each task"""
    logger.info(
        f"Running independent trials for {len(task_ids)} tasks with {trials_per_task} trials each"
    )
    remaining_tasks = filter_incomplete_tasks(task_results, trials_per_task)

    for task_id in remaining_tasks:
        while len(task_results[task_id].trials) < trials_per_task:
            trial_index = len(task_results[task_id].trials)
            logger.info(f"Starting trial {trial_index + 1} for task {task_id}")

            result = trial_executor(task_id, trial_index)
            task_results[task_id].trials.append(result)

            logger.info(f"Trial {result.trial_id} completed. Score: {result.score:.3f}")

            # Save checkpoint after each trial
            checkpoint_saver(task_results)
            if not result.success:
                logger.error(f"Trial failed for task {task_id}")


def run_chained_trials(
    task_ids: list[str],
    trials_per_task: int,
    task_results: dict[str, TaskTrialResults],
    trial_executor: Callable[[str, int], TaskTrialResult],
    checkpoint_saver: Callable[[dict[str, TaskTrialResults], int], None],
    completed_rounds: int = 0,
) -> None:
    """Run trials in lockstep across all tasks"""
    for trial_round in range(completed_rounds, trials_per_task):
        logger.info(f"Starting trial round {trial_round + 1}/{trials_per_task}")

        success = True
        for task_id in task_ids:
            if len(task_results[task_id].trials) > trial_round:
                continue  # Already completed

            logger.info(f"Running trial {trial_round + 1} for task {task_id}")
            result = trial_executor(task_id, trial_round)
            task_results[task_id].trials.append(result)

            if not result.success:
                logger.error(
                    f"Trial failed for task {task_id}: {getattr(result, 'error_message', 'No error message')}"
                )
                logger.error(f"Full result: {result}")
                success = False
        # Save checkpoint after each round
        checkpoint_saver(task_results, trial_round + 1 if success else trial_round)


def create_wandb_config(agent: BaseAgent, session_id: str, **kwargs) -> dict[str, Any]:
    """Create wandb configuration"""
    return {
        "agent_type": agent.__class__.__name__,
        "model": getattr(agent, "model", "unknown_model"),
        "session_id": session_id,
        "agent_max_iterations": getattr(agent, "max_iterations", None),
        "agent_temperature": getattr(agent, "temperature", None),
        **kwargs,
    }


class CorralRunner:
    """Simplified benchmark runner with functional approach."""

    def __init__(
        self,
        interface: CorralRouter,
        agent: BaseAgent,
        checkpoint_dir: str = "./benchmark_checkpoints",
        checkpoint_name: str | None = None,
        logger: CorralWandbLogger | None = None,
        enable_surrender: bool = False,
        metrics: list[Metric] | None = None,
    ):
        self.interface = interface
        self.agent = agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = (
            checkpoint_name or f"checkpoint_{agent.__class__.__name__}"
        )
        self.logger = logger
        self.enable_surrender = enable_surrender

        # Initialize metric registry
        self._metric_registry = MetricRegistry()
        self._default_k_values: list[int] = [5]  # Match BenchmarkResult default

        # Register initial metrics
        if metrics is not None:
            for metric in metrics:
                self._metric_registry.register(metric)
        # If no metrics provided, registry stays empty until bench() is called
        # This allows lazy initialization with proper k_values

    @property
    def metric_registry(self) -> MetricRegistry:
        """Access the internal metric registry.

        This allows advanced users to directly interact with the registry
        for operations like parallel calculation.

        Returns:
            The MetricRegistry instance used by this runner.
        """
        return self._metric_registry

    # Metric Introspection Methods
    def list_metrics(self, k_values: list[int] | None = None) -> list[dict[str, str]]:
        """List all metrics that will be used for benchmarking.

        Args:
            k_values: k values for pass@k metrics. Used only if no metrics
                      have been explicitly configured. If None, uses [1].

        Returns:
            List of dicts with 'name', 'display_name', 'description', and 'type'
        """
        # If registry is empty and no explicit metrics, show defaults
        if len(self._metric_registry.list_all()) == 0:
            metrics = get_default_metrics(k_values or [1])
        else:
            metrics = self._metric_registry.list_all()

        return [
            {
                "name": m.metadata.name,
                "display_name": m.metadata.display_name,
                "description": m.metadata.description,
                "type": "task" if isinstance(m, TaskMetric) else "overall",
            }
            for m in metrics
        ]

    def print_metrics(self, k_values: list[int] | None = None) -> None:
        """Print a formatted table of metrics that will be used.

        Args:
            k_values: k values for pass@k metrics. Used only if no metrics
                      have been explicitly configured. If None, uses [1].
        """
        metrics = self.list_metrics(k_values)
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Configured Metrics ({len(metrics)} total)")
        logger.info(f"{'=' * 70}")
        for m in sorted(metrics, key=lambda x: (x["type"], x["name"])):
            logger.info(f"  [{m['type']:7}] {m['name']}: {m['description']}")
        logger.info(f"{'=' * 70}\n")

    # Metric Registration Methods
    def register_metric(self, metric: Metric) -> None:
        """Register a new metric.

        If no metrics have been explicitly registered yet, this will first
        populate the registry with default metrics before adding the new one.

        Args:
            metric: The metric instance to register.

        Raises:
            ValueError: If a metric with the same name already exists.
        """
        # Initialize with defaults if registry is empty
        if len(self._metric_registry.list_all()) == 0:
            for default_metric in get_default_metrics(self._default_k_values):
                self._metric_registry.register(default_metric)

        self._metric_registry.register(metric)

    def unregister_metric(self, metric_name: str) -> Metric | None:
        """Unregister a metric by name.

        If no metrics have been explicitly registered yet, this will first
        populate the registry with default metrics before removing.

        Args:
            metric_name: The name of the metric to remove.

        Returns:
            The removed metric instance, or None if not found.
        """
        # Initialize with defaults if registry is empty
        if len(self._metric_registry.list_all()) == 0:
            for default_metric in get_default_metrics(self._default_k_values):
                self._metric_registry.register(default_metric)

        try:
            metric = self._metric_registry.get(metric_name)
            self._metric_registry.unregister(metric_name)
            return metric
        except KeyError:
            return None

    def clear_metrics(self) -> None:
        """Remove all registered metrics."""
        # Create a fresh empty registry
        self._metric_registry = MetricRegistry()

    def reset_metrics(self, k_values: list[int] | None = None) -> None:
        """Reset metrics to the default set.

        Args:
            k_values: k values for pass@k metrics. If None, uses [1].
        """
        k_vals = k_values or [1]
        self._default_k_values = k_vals
        self._metric_registry = MetricRegistry()
        for metric in get_default_metrics(k_vals):
            self._metric_registry.register(metric)

    def generate_latex_docs(
        self,
        task_ids: list[str],
        output_dir: str | None = None,
        level: int | str = 1,
        env_name: str | None = None,
        verbosity: str | None = None,
    ) -> None:
        """Generate LaTeX documentation for a list of tasks.

        The colorbox generation workflow:
        1. First task (main task, no input_from_tasks): saves to cache only
        2. Subsequent tasks (subtasks, have input_from_tasks): add to cache and generate .tex
        3. After all tasks: clear the cache

        Args:
            task_ids: List of task IDs to generate LaTeX for.
            output_dir: Directory for output .tex files. Defaults to
                        "tex_files" in the current working directory.
            level: Task level identifier (e.g., 1, 2,...). Default: 1.
            env_name: Environment name (e.g., "afm", "catalyst").
            verbosity: Tool verbosity level used to filter tool descriptions and
                       return sections in the generated LaTeX. Accepts a
                       `ToolVerbosity` value string (e.g. "brief",
                       "detailed"). Defaults to "detailed" when not
                       provided.
        """
        output_dir = output_dir or str(Path.cwd() / "tex_files")
        logger.info(f"Generating LaTeX documentation for {len(task_ids)} tasks...")

        for task_id in task_ids:
            try:
                result = self.interface.generate_latex(
                    task_id=task_id,
                    output_dir=output_dir,
                    level=level,
                    env_name=env_name,
                    verbosity=verbosity,
                )
                logger.debug(
                    f"Generated LaTeX for task {task_id}: "
                    f"task={result.get('output_path')}, "
                    f"tools={result.get('tools_output_path')}, "
                    f"scoring={result.get('scoring_output_path')}"
                )
            except Exception as e:
                logger.warning(f"Failed to generate LaTeX for task {task_id}: {e}")

        logger.info("LaTeX documentation generation complete.")

    def _get_metrics_for_benchmark(self) -> list[Metric] | None:
        """Get the metrics list for creating a BenchmarkResult.

        If the registry is empty (no explicit configuration), returns None
        to let BenchmarkResult use its default behavior with k_values.
        Otherwise, returns the explicitly configured metrics.

        Returns:
            List of metrics if explicitly configured, or None for defaults.
        """
        registered = self._metric_registry.list_all()
        if len(registered) == 0:
            return None  # Let BenchmarkResult use defaults with k_values
        return registered

    # Run Benchmark Method
    def bench(
        self,
        task_ids: list[str] | None = None,
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        tool_verbosity: str | None = None,
        configure_timeout: float | None = None,
        hooks: AgentHooks | None = None,
        run_name: str | None = None,
    ) -> BenchmarkResult:
        """Run benchmark with functional approach

        Args:
            task_ids: List of task IDs to benchmark. If None, uses all available tasks.
            trials_per_task: Number of trials to run per task.
            k_values: k values for pass@k metrics.
            verbose: Whether to enable verbose logging.
            session_id: Session identifier. Auto-generated if None.
            tool_verbosity: Verbosity level for tools.
            configure_timeout: Timeout for configuring additional apps.
            hooks: Agent hooks to inject into the agent before running.
            run_name: Name for the benchmark run (used in report filename).
                     If None, defaults to "unknown_env".

        Returns:
            BenchmarkResult containing all trial results and metrics.
        """
        task_ids = task_ids or self.interface.get_available_tasks()

        # Set agent hooks if provided
        if hooks:
            self.agent.hooks = hooks

        def trial_executor(task_id: str, trial_index: int) -> TaskTrialResult:
            return execute_single_trial(
                task_id=task_id,
                trial_index=trial_index,
                interface=self.interface,
                agent=self.agent,
                verbose=verbose,
                tool_verbosity=tool_verbosity,
                configure_timeout=configure_timeout,
                enable_surrender=self.enable_surrender,
            )

        return self._run_benchmark(
            task_ids=task_ids,
            trial_executor=trial_executor,
            trials_per_task=trials_per_task,
            k_values=k_values,
            verbose=verbose,
            session_id=session_id,
            tool_verbosity=tool_verbosity,
            hooks=hooks,
            run_name=run_name,
        )

    def bench_from_traces(
        self,
        traces: dict[str, list[LiteLLMMessage]],
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        tool_verbosity: str | None = None,
        configure_timeout: float | None = None,
        hooks: AgentHooks | None = None,
        run_name: str | None = None,
    ) -> BenchmarkResult:
        """Run benchmark from previously saved conversation traces.

        Instead of building prompts from scratch, each task is initialised
        from the trace provided in ``traces``.  A deep-copy of the prototype
        agent (``self.agent``) is created per task with its
        ``_initial_messages`` set to the corresponding trace so the agent
        continues from that conversation state.

        Args:
            traces: Mapping of task_id to the conversation trace (list of
                ``LiteLLMMessage``) to replay from.
            trials_per_task: Number of trials to run per task.
            k_values: k values for pass@k metrics.
            verbose: Whether to enable verbose logging.
            session_id: Session identifier. Auto-generated if None.
            tool_verbosity: Verbosity level for tools.
            configure_timeout: Timeout for configuring additional apps.
            hooks: Agent hooks to inject into every agent clone.
            run_name: Name for the benchmark run (used in report filename).
                     If None, auto-generates one.

        Returns:
            BenchmarkResult containing all trial results and metrics.
        """
        task_ids = list(traces.keys())

        def trial_executor(task_id: str, trial_index: int) -> TaskTrialResult:
            trace = traces[task_id]
            self.agent._initial_messages = list(trace)
            if hooks:
                self.agent.hooks = hooks
            try:
                return execute_single_trial(
                    task_id=task_id,
                    trial_index=trial_index,
                    interface=self.interface,
                    agent=self.agent,
                    verbose=verbose,
                    tool_verbosity=tool_verbosity,
                    configure_timeout=configure_timeout,
                    enable_surrender=self.enable_surrender,
                )
            finally:
                self.agent._initial_messages = None

        return self._run_benchmark(
            task_ids=task_ids,
            trial_executor=trial_executor,
            trials_per_task=trials_per_task,
            k_values=k_values,
            verbose=verbose,
            session_id=session_id,
            tool_verbosity=tool_verbosity,
            hooks=hooks,
            run_name=run_name,
            extra_wandb_config={"from_traces": True},
        )

    def _run_benchmark(
        self,
        task_ids: list[str],
        trial_executor: Callable[[str, int], TaskTrialResult],
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        tool_verbosity: str | None = None,
        hooks: AgentHooks | None = None,
        run_name: str | None = None,
        extra_wandb_config: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Shared benchmark execution logic used by ``bench`` and ``bench_from_traces``.

        This method handles setup, logging, trial orchestration (independent or
        chained), result aggregation, checkpointing, and report generation.

        Args:
            task_ids: List of task IDs to benchmark.
            trial_executor: Callable that runs a single trial given
                ``(task_id, trial_index)`` and returns a ``TaskTrialResult``.
            trials_per_task: Number of trials to run per task.
            k_values: k values for pass@k metrics.
            verbose: Whether to enable verbose logging.
            session_id: Session identifier. Auto-generated if None.
            tool_verbosity: Verbosity level for tools.
            hooks: Agent hooks (used only for wandb config flag).
            run_name: Name for the benchmark run.
            extra_wandb_config: Additional key-value pairs merged into the
                wandb configuration dict.

        Returns:
            BenchmarkResult containing all trial results and metrics.
        """
        # Setup
        if tool_verbosity:
            self.interface.set_verbosity(tool_verbosity)

        session_id = session_id or create_session_id()
        k_values = validate_k_values(k_values, trials_per_task)

        if trials_per_task == 0:
            raise ValueError("Number of trials per task must be greater than 0")

        # Initialize or load results
        task_results = self._load_or_initialize_results(task_ids, session_id)

        checkpoint_saver = partial(self._save_checkpoint, session_id)

        # Setup logging
        if self.logger:
            config = create_wandb_config(
                self.agent,
                session_id,
                trials_per_task=trials_per_task,
                k_values=k_values,
                tool_verbosity=self.interface.current_verbosity,
                task_ids=task_ids,
                dependency_chain=self.interface.supports_dependency_chain(),
                enable_surrender=self.enable_surrender,
                hooks_enabled=hooks is not None,
                **(extra_wandb_config or {}),
            )
            self.logger.start_logging(config)

        # Execute benchmark
        start_time = datetime.now(tz=timezone.utc)

        try:
            if self.interface.supports_dependency_chain():
                checkpoint = self._load_checkpoint(session_id)
                completed_rounds = (
                    checkpoint.get("completed_trials", 0) if checkpoint else 0
                )

                run_chained_trials(
                    task_ids,
                    trials_per_task,
                    task_results,
                    self._make_logging_trial_executor(trial_executor),
                    self._make_chained_checkpoint_saver(checkpoint_saver),
                    completed_rounds,
                )
            else:
                run_independent_trials(
                    task_ids,
                    trials_per_task,
                    task_results,
                    self._make_logging_trial_executor(trial_executor),
                    checkpoint_saver,
                )

            # Create final result
            end_time = datetime.now(tz=timezone.utc)
            total_duration = (end_time - start_time).total_seconds()

            result = BenchmarkResult(
                task_results=task_results,
                k=k_values,
                verbosity=self.interface.current_verbosity,
                verbose=verbose,
                total_duration=total_duration,
                metrics=self._get_metrics_for_benchmark(),
            )

            # Log final results
            if self.logger:
                self.logger.log_final_results(result)

            # Save final checkpoint with finished suffix and remove original
            self._save_finished_checkpoint(session_id, task_results)
            self._remove_original_checkpoint(session_id)

            # Generate and save report
            self._save_benchmark_report(result, run_name, tool_verbosity)

            return result

        finally:
            if self.logger:
                self.logger.finish()

    @staticmethod
    def _load_traces_from_agent_logs(
        directory: Path, trial_index: int
    ) -> dict[str, list[LiteLLMMessage]]:
        """Extract traces from a directory of agent log JSON files."""
        json_files = sorted(directory.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {directory}")

        # Group files by task_id
        task_files: dict[str, list[Path]] = {}
        for file_path in json_files:
            try:
                with file_path.open() as f:
                    data = json.load(f)
                task_id = data.get("task_id")
                if task_id is None:
                    logger.warning(f"Skipping {file_path.name}: no 'task_id' field.")
                    continue
                task_files.setdefault(task_id, []).append(file_path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping {file_path.name}: {e}")

        if not task_files:
            raise ValueError(
                f"No valid agent log files found in directory: {directory}"
            )

        traces: dict[str, list[LiteLLMMessage]] = {}

        for task_id, files in task_files.items():
            if trial_index >= len(files):
                logger.warning(
                    f"Task '{task_id}' has only {len(files)} log file(s), "
                    f"skipping (requested trial_index={trial_index})."
                )
                continue

            selected_file = files[trial_index]
            with selected_file.open() as f:
                data = json.load(f)

            messages = data.get("messages")
            if not messages:
                logger.warning(
                    f"Task '{task_id}' log {selected_file.name} has no messages."
                )
                continue

            traces[task_id] = [LiteLLMMessage(**msg) for msg in messages]

        if not traces:
            raise ValueError(
                f"No traces could be extracted from agent logs in: {directory}"
            )

        return traces

    def _load_or_initialize_results(
        self, task_ids: list[str], session_id: str
    ) -> dict[str, TaskTrialResults]:
        """Load existing results or initialize new ones"""
        checkpoint = self._load_checkpoint(session_id)

        if checkpoint and "task_results" in checkpoint:
            task_results = checkpoint["task_results"]
            logger.info(f"Loaded existing results for {len(task_results)} tasks")

            # Ensure all requested tasks have entries
            for task_id in task_ids:
                if task_id not in task_results:
                    task_results[task_id] = TaskTrialResults(task_id=task_id)
        else:
            task_results = initialize_task_results(task_ids)

        return task_results

    def _make_logging_trial_executor(self, trial_executor: Callable) -> Callable:
        """Wrap trial executor with logging"""

        def wrapped_executor(task_id: str, trial_index: int) -> TaskTrialResult:
            result = trial_executor(task_id, trial_index)
            if self.logger:
                self.logger.log_trial(result)
            return result

        return wrapped_executor

    def _make_chained_checkpoint_saver(self, checkpoint_saver: Callable) -> Callable:
        """Create checkpoint saver for chained execution"""

        def wrapped_saver(
            task_results: dict[str, TaskTrialResults], completed_trials: int
        ) -> None:
            checkpoint_saver(task_results, completed_trials=completed_trials)

        return wrapped_saver

    def _write_checkpoint_file(
        self,
        checkpoint_path: Path,
        checkpoint: dict,
        session_id: str,
        checkpoint_type: str = "checkpoint",
    ) -> None:
        """
        Write checkpoint data to file using an atomic write pattern.

        This method first writes the checkpoint to a temporary file and then renames it to the target path.
        This approach ensures that the checkpoint file is never left in a partially written or corrupted state,
        even if the process crashes or is interrupted during the write. The atomic rename operation guarantees
        that readers will either see the old file or the fully written new file, but never a half-written file.
        If an error occurs, the temporary file is cleaned up to avoid clutter.
        """
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            with temp_path.open("wb") as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(checkpoint_path)
            logger.info(f"{checkpoint_type.title()} saved for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save {checkpoint_type}: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _save_checkpoint(
        self, session_id: str, task_results: dict[str, TaskTrialResults], **extra_data
    ) -> None:
        """
        Save checkpoint to file, including all relevant session and task state.

        This method centralizes the logic for checkpoint creation, ensuring that all necessary
        metadata (such as session ID and timestamp) is included. It delegates the actual file
        writing to an atomic method to guarantee data integrity. This design allows for robust
        recovery and resumption of long-running or multi-step processes.
        """
        checkpoint = {
            "task_results": task_results,
            "session_id": session_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **extra_data,
        }

        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )

        self._write_checkpoint_file(
            checkpoint_path, checkpoint, session_id, "checkpoint"
        )

    def _load_checkpoint(self, session_id: str) -> dict | None:
        """
        Load checkpoint from file, or recover the most recent one if not found.

        This method attempts to load a checkpoint for the given session. If the specific
        checkpoint file does not exist (e.g., due to interruption or cleanup), it searches
        for the most recent available checkpoint with the same naming pattern. This design
        increases robustness and allows for recovery from unexpected interruptions or missing files.
        """
        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )

        if not checkpoint_path.exists():
            # Search for the most recent checkpoint with same checkpoint_name
            return self._find_most_recent_checkpoint()

        try:
            with checkpoint_path.open("rb") as f:
                checkpoint = pickle.load(f)
            logger.info("Checkpoint loaded successfully")
            return checkpoint
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")
            return None

    def _find_most_recent_checkpoint(self) -> dict | None:
        """
        Find and load the most recent checkpoint file with the same checkpoint_name.

        This method scans the checkpoint directory for files matching the session pattern,
        excluding those marked as finished. It sorts the files by timestamp and loads the most
        recent one. This enables recovery from interruptions and ensures that progress is not lost
        if the latest checkpoint file is missing or incomplete. It is a fallback mechanism for robust
        checkpoint management.
        """

        # Pattern to match checkpoint files with same name but different session IDs
        # excluding "finished" files
        pattern = f"{self.checkpoint_name}_session_*.pkl"

        matching_files = []
        for file_path in self.checkpoint_dir.glob(pattern):
            file_name = file_path.name
            # Skip finished checkpoints
            if "finished" in file_name:
                continue

            # Extract session ID from filename
            match = re.search(r"session_(\d{8}_\d{6}_\d{6})", file_name)
            if match:
                session_timestamp = match.group(1)
                matching_files.append((file_path, session_timestamp))

        if not matching_files:
            logger.info("No existing checkpoints found")
            return None

        # Sort by session timestamp (most recent first)
        matching_files.sort(key=lambda x: x[1], reverse=True)
        most_recent_file = matching_files[0][0]

        try:
            with most_recent_file.open("rb") as f:
                checkpoint = pickle.load(f)
            logger.info(f"Loaded most recent checkpoint: {most_recent_file.name}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Error loading most recent checkpoint: {e}")
            return None

    def _save_finished_checkpoint(
        self, session_id: str, task_results: dict[str, TaskTrialResults]
    ) -> None:
        """
        Save the final checkpoint with a 'finished' suffix to mark completion.

        This method creates a checkpoint file that is clearly marked as finished, making it easy
        to distinguish between in-progress and completed runs. This helps prevent accidental
        resumption of already completed sessions and provides a clear audit trail for completed
        benchmarks. The atomic write pattern is used for reliability.
        """
        checkpoint = {
            "task_results": task_results,
            "session_id": session_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "status": "finished",
        }

        finished_checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_finished_{session_id}.pkl"
        )

        self._write_checkpoint_file(
            finished_checkpoint_path, checkpoint, session_id, "final checkpoint"
        )

    def _remove_original_checkpoint(self, session_id: str) -> None:
        """
        Remove the original (unfinished) checkpoint file after completion.

        This method deletes the in-progress checkpoint file once a finished checkpoint has been
        written. This prevents confusion between incomplete and completed runs, and helps keep
        the checkpoint directory clean. The try/except block ensures that errors during removal
        do not interrupt the main workflow.
        """
        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )

        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Original checkpoint removed for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to remove original checkpoint: {e}")

    def _save_benchmark_report(
        self, result: BenchmarkResult, run_name: str | None, tool_verbosity: str | None
    ) -> None:
        """
        Save benchmark report to a JSON file.

        If no run name is provided, generates a filename in the format:
        {model}-{agent}-{env}-{verbosity}-{timestamp}.json

        Args:
            result: The BenchmarkResult to save
            run_name: Name for the benchmark run. If None, uses "unknown_env"
            tool_verbosity: The verbosity level used (or None)
        """
        if run_name is not None:
            filename = run_name if run_name.endswith(".json") else f"{run_name}.json"
        else:
            # Extract components for filename
            model_name = getattr(self.agent, "model", "unknown_model")
            # Sanitize model name for filesystem safety
            model_name = sanitize_model_name(model_name)

            agent_name = self.agent.__class__.__name__

            verbosity_str = tool_verbosity if tool_verbosity else "None"

            # Add timestamp
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

            # Construct filename
            filename = f"{model_name}-{agent_name}-{verbosity_str}-{timestamp}.json"

        # Save the report
        try:
            result.generate_report(filename)
            logger.info(f"Benchmark report saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save benchmark report: {e}")
