"""Command-line entry points for running Corral benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from temporalio.client import Client

import corral.orchestration as orchestration
from corral.environment_loader import ENVIRONMENT_NAMES, load_environment_group
from corral.persistence import JSONLStateStore
from corral.run import CorralRunner


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Lazy import metadata for one selectable concrete agent."""

    module: str
    attribute: str
    extra: str | None = None
    default_kwargs: tuple[tuple[str, Any], ...] = ()


AGENT_DEFINITIONS: Mapping[str, AgentDefinition] = MappingProxyType(
    {
        "ai-scientist": AgentDefinition(
            "corral.agents.ai_scientist.agent",
            "AIScientistAgent",
        ),
        "claude-code": AgentDefinition(
            "corral.agents.claude_code",
            "ClaudeCodeAgent",
            extra="claude",
        ),
        "codex": AgentDefinition(
            "corral.agents.codex",
            "CodexAgent",
            extra="codex",
        ),
        "llm-planner": AgentDefinition(
            "corral.agents.llm_planner",
            "LLMPlanner",
        ),
        "openhands": AgentDefinition(
            "corral.agents.openhands",
            "OpenHandsAgent",
            extra="openhands",
        ),
        "react": AgentDefinition("corral.agents.react", "ReActAgent"),
        "reflexion": AgentDefinition(
            "corral.agents.reflexion_agent",
            "ReflexionAgent",
        ),
        "terminus": AgentDefinition("corral.agents.terminus", "TerminusAgent"),
        "tool-calling": AgentDefinition(
            "corral.agents.tool_calling",
            "ToolCallingAgent",
            # Provider-native function tools and the reasoning mode enabled by
            # default for some GPT models cannot be combined reliably.
            default_kwargs=(("reasoning_effort", "none"),),
        ),
    }
)
AGENT_NAMES = tuple(AGENT_DEFINITIONS)


def _agent_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


_AGENT_ALIASES = {
    alias: canonical
    for canonical, definition in AGENT_DEFINITIONS.items()
    for alias in (
        _agent_key(canonical),
        _agent_key(definition.attribute),
    )
}


def normalise_agent_name(name: str) -> str:
    """Resolve CLI spellings and class names to a canonical agent name."""
    canonical = _AGENT_ALIASES.get(_agent_key(name))
    if canonical is None:
        known = ", ".join(AGENT_NAMES)
        raise ValueError(f"unknown agent {name!r}; choose one of {known}")
    return canonical


def _agent_argument(raw: str) -> str:
    try:
        return normalise_agent_name(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return value


def _optional_timeout(raw: str) -> float | None:
    if raw.casefold() in {"none", "null"}:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timeout must be a positive number or 'none'"
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("timeout must be greater than zero")
    return value


def _load_agent_class(name: str) -> type[Any]:
    definition = AGENT_DEFINITIONS[name]
    module = import_module(definition.module)
    agent_class = getattr(module, definition.attribute, None)
    if not isinstance(agent_class, type):
        raise TypeError(
            f"registered agent {name!r} did not resolve to a class: "
            f"{definition.module}:{definition.attribute}"
        )
    return agent_class


def _ai_scientist_config(kwargs: dict[str, Any]) -> None:
    profile = kwargs.pop("config_profile", "default")
    raw_config = kwargs.get("config")
    if raw_config is None:
        if profile == "default":
            return
        raw_config = {}
    if not isinstance(raw_config, Mapping):
        raise TypeError("AI Scientist config must be a JSON object")
    config_module = import_module("corral.agents.ai_scientist.config")
    if profile == "default":
        config_class = config_module.AIScientistConfig
    elif profile == "sakana":
        config_class = config_module.SakanaAIScientistConfig
    else:
        raise ValueError("AI Scientist config_profile must be 'default' or 'sakana'")
    kwargs["config"] = config_class.model_validate(dict(raw_config))


def _common_agent_kwargs(
    agent_class: type[Any],
    kwargs: dict[str, Any],
    *,
    model: str | None,
    api_endpoint: str | None,
    temperature: float | None,
) -> None:
    signature = inspect.signature(agent_class)
    parameters = signature.parameters
    accepts_extra = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    common = {
        "model": model,
        "api_endpoint": api_endpoint,
        "temperature": temperature,
    }
    for key, value in common.items():
        if value is None:
            continue
        if key not in parameters and not accepts_extra:
            option = key.replace("_", "-")
            raise ValueError(f"{agent_class.__name__} does not support --{option}")
        # Explicit, named CLI options take precedence over --agent-kwargs.
        kwargs[key] = value


def create_agent(
    name: str,
    *,
    model: str | None = None,
    api_endpoint: str | None = None,
    temperature: float | None = None,
    agent_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Construct any public concrete Corral agent from CLI-safe values."""
    canonical = normalise_agent_name(name)
    definition = AGENT_DEFINITIONS[canonical]
    kwargs = dict(definition.default_kwargs)
    kwargs.update(dict(agent_kwargs or {}))

    if model is None:
        model = kwargs.pop("model", None)
    else:
        kwargs.pop("model", None)
    if api_endpoint is None:
        api_endpoint = kwargs.pop("api_endpoint", None)
    else:
        kwargs.pop("api_endpoint", None)
    if temperature is None:
        temperature = kwargs.pop("temperature", None)
    else:
        kwargs.pop("temperature", None)

    if canonical == "reflexion":
        actor_name = kwargs.pop("actor", "tool-calling")
        actor_kwargs = kwargs.pop("actor_kwargs", {})
        if not isinstance(actor_name, str):
            raise TypeError("Reflexion actor must be an agent name")
        if normalise_agent_name(actor_name) == "reflexion":
            raise ValueError("Reflexion cannot wrap another Reflexion agent")
        if not isinstance(actor_kwargs, Mapping):
            raise TypeError("Reflexion actor_kwargs must be a JSON object")
        actor = create_agent(
            actor_name,
            model=model,
            api_endpoint=api_endpoint,
            temperature=temperature,
            agent_kwargs=actor_kwargs,
        )
        agent_class = _load_agent_class(canonical)
        kwargs["actor"] = actor
        kwargs.setdefault("reflection_model", model or getattr(actor, "model", None))
        if api_endpoint is not None:
            kwargs["api_endpoint"] = api_endpoint
        if temperature is not None:
            kwargs["reflection_temperature"] = temperature
        return agent_class(**kwargs)

    agent_class = _load_agent_class(canonical)
    if canonical == "ai-scientist":
        _ai_scientist_config(kwargs)
    _common_agent_kwargs(
        agent_class,
        kwargs,
        model=model,
        api_endpoint=api_endpoint,
        temperature=temperature,
    )
    return agent_class(**kwargs)


def _list_tasks(environments: Mapping[str, Any]) -> None:
    lines = []
    for task_id, environment in environments.items():
        task = environment.current_task
        dependencies = ", ".join(sorted(task.dependencies())) or "-"
        description = " ".join(task.description.strip().split())
        if len(description) > 120:
            description = f"{description[:117]}..."
        lines.append(f"{task_id}\tdependencies={dependencies}\t{description}")
    sys.stdout.write("\n".join(lines) + "\n")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return slug[:120] or uuid4().hex


def _print_results(run_id: str, result: Any) -> None:
    lines = [f"Run {run_id} completed:"]
    for task_id, task_results in result.task_results.items():
        for trial in task_results.trials:
            status = "failed" if trial.error_message else "completed"
            score = "-" if trial.score is None else f"{trial.score:g}"
            output = json.dumps(trial.output, ensure_ascii=False, default=str)
            lines.append(
                f"- {task_id} [{trial.trial_id}]: {status}, "
                f"score={score}, output={output}"
            )
    sys.stdout.write("\n".join(lines) + "\n")


def _configured_agent(
    args: argparse.Namespace,
    *,
    agent_name: str | None = None,
    agent_kwargs: Mapping[str, Any] | None = None,
) -> tuple[str, Any, str]:
    canonical = normalise_agent_name(agent_name or args.agent)
    configured_kwargs = dict(getattr(args, "agent_kwargs", {}) or {})
    configured_kwargs.update(dict(agent_kwargs or {}))
    agent = create_agent(
        canonical,
        model=args.model,
        api_endpoint=args.api_endpoint,
        temperature=args.temperature,
        agent_kwargs=configured_kwargs,
    )
    model = getattr(agent, "model", None)
    if not isinstance(model, str) or not model:
        raise ValueError(f"{type(agent).__name__} did not define a model")
    return canonical, agent, model


async def _connect_temporal(args: argparse.Namespace) -> Client:
    try:
        return await Client.connect(
            args.temporal_address,
            namespace=args.temporal_namespace,
        )
    except Exception as exc:
        raise ConnectionError(
            f"could not connect to Temporal at {args.temporal_address!r}; "
            "start it with 'temporal server start-dev' or pass "
            "--temporal-address"
        ) from exc


def _activity_policy(args: argparse.Namespace) -> Any:
    return orchestration.ActivityPolicy(
        start_to_close_seconds=args.activity_timeout,
        heartbeat_timeout_seconds=args.heartbeat_timeout,
        maximum_attempts=args.max_attempts,
    )


async def run_benchmark(
    args: argparse.Namespace,
    *,
    agent_name: str | None = None,
    agent_kwargs: Mapping[str, Any] | None = None,
) -> int:
    """Run one CLI-configured benchmark through the normal Temporal path."""
    environments = load_environment_group(
        args.environment,
        env_kwargs=args.env_kwargs,
    )
    if args.list_tasks:
        _list_tasks(environments)
        return 0

    canonical_agent, agent, model = _configured_agent(
        args,
        agent_name=agent_name,
        agent_kwargs=agent_kwargs,
    )

    run_id = args.run_id or (
        f"{_slug(canonical_agent)}-{_slug(args.environment)}-{uuid4().hex[:12]}"
    )
    task_queue = args.task_queue or f"corral-{_slug(run_id)}"
    store = JSONLStateStore(Path(args.state_file).expanduser().resolve())
    registry = orchestration.RuntimeRegistry(
        agents={canonical_agent: agent},
        environments=environments,
    )

    try:
        client = await _connect_temporal(args)

        activities = orchestration.CorralActivities(store, registry)
        async with orchestration.create_worker(
            client,
            task_queue=task_queue,
            activities=activities,
            max_concurrent_activities=args.max_parallel,
        ):
            runner = CorralRunner(
                orchestration.TemporalBenchmarkExecutor(
                    client,
                    task_queue=task_queue,
                ),
                environments=environments,
                agent_id=canonical_agent,
                model=model,
                max_iterations=args.max_iterations,
                state_store=store,
            )
            result = await runner.run(
                run_id,
                task_ids=args.tasks or None,
                trials_per_task=args.trials,
                max_parallel=args.max_parallel,
                max_parallel_per_task=args.max_parallel_per_task,
                enable_surrender=args.enable_surrender,
                evaluate=not args.no_evaluate,
                verbose=args.verbose,
                activity_policy=_activity_policy(args),
            )
    finally:
        registry.close()
        store.close()

    result.generate_report(args.report)
    _print_results(run_id, result)
    return int(any(trial.error_message for trial in result.all_results))


async def run_task(args: argparse.Namespace) -> int:
    """Run exactly one independent task without benchmark evaluation/reporting."""
    environments = load_environment_group(
        args.environment,
        env_kwargs=args.env_kwargs,
    )
    try:
        environment = environments[args.task]
    except KeyError as exc:
        known = ", ".join(environments)
        raise ValueError(
            f"unknown task {args.task!r} for environment {args.environment!r}; "
            f"choose one of {known}"
        ) from exc

    dependencies = tuple(sorted(environment.current_task.dependencies()))
    if dependencies:
        listed = ", ".join(dependencies)
        raise ValueError(
            f"task {args.task!r} depends on {listed}; `corral run` executes one "
            "independent task, so use `corral bench` for dependency graphs"
        )

    canonical_agent, agent, model = _configured_agent(args)
    execution_id = args.execution_id or (
        f"run-{_slug(canonical_agent)}-{_slug(args.environment)}-"
        f"{_slug(args.task)}-{uuid4().hex[:12]}"
    )
    task_queue = args.task_queue or f"corral-{_slug(execution_id)}"
    store = JSONLStateStore(Path(args.state_file).expanduser().resolve())
    registry = orchestration.RuntimeRegistry(
        agents={canonical_agent: agent},
        environments={args.task: environment},
    )

    try:
        client = await _connect_temporal(args)
        activities = orchestration.CorralActivities(store, registry)
        async with orchestration.create_worker(
            client,
            task_queue=task_queue,
            activities=activities,
            max_concurrent_activities=1,
        ):
            state = await orchestration.execute_task(
                executor=orchestration.TemporalTaskExecutor(
                    client,
                    state_store=store,
                    task_queue=task_queue,
                ),
                task=orchestration.TaskWorkflowInput(
                    execution_id=execution_id,
                    task_id=args.task,
                    environment_id=args.task,
                    agent_id=canonical_agent,
                    model=model,
                    max_iterations=args.max_iterations,
                    enable_surrender=args.enable_surrender,
                    evaluate=False,
                    activity_policy=_activity_policy(args),
                ),
            )
    finally:
        registry.close()
        store.close()

    answer = json.dumps(state.submission, ensure_ascii=False)
    sys.stdout.write(
        f"Run {execution_id} completed:\n"
        f"- task: {args.task}\n"
        f"- status: {state.runtime.status}\n"
        f"- answer: {answer}\n"
        f"- state: {state.state_hash}\n"
    )
    return int(state.runtime.status == "failed")


def _add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    environment = parser.add_argument_group("environment")
    environment.add_argument(
        "--environment",
        required=True,
        choices=ENVIRONMENT_NAMES,
        help="Registered environment to run.",
    )
    environment.add_argument(
        "--env-kwargs",
        type=_json_object,
        default={},
        metavar="JSON",
        help="JSON object configuring the selected environment.",
    )


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    agent = parser.add_argument_group("agent")
    agent.add_argument(
        "--agent",
        required=True,
        type=_agent_argument,
        metavar="NAME",
        help=f"Agent to run. Available: {', '.join(AGENT_NAMES)}.",
    )
    agent.add_argument(
        "--model",
        help="Model identifier. If omitted, use the selected agent's default.",
    )
    agent.add_argument("--api-endpoint")
    agent.add_argument("--temperature", type=float)
    agent.add_argument(
        "--agent-kwargs",
        type=_json_object,
        default={},
        metavar="JSON",
        help="Additional constructor options for the selected agent.",
    )
    agent.add_argument("--max-iterations", type=int, default=20)
    agent.add_argument("--enable-surrender", action="store_true")


def _add_temporal_arguments(
    execution: Any,
    *,
    default_state_file: str,
) -> None:
    execution.add_argument("--task-queue")
    execution.add_argument("--temporal-address", default="localhost:7233")
    execution.add_argument("--temporal-namespace", default="default")
    execution.add_argument(
        "--activity-timeout",
        type=_optional_timeout,
        default=None,
        metavar="SECONDS|none",
        help="Start-to-Close timeout; default: none.",
    )
    execution.add_argument(
        "--heartbeat-timeout",
        type=_optional_timeout,
        default=None,
        metavar="SECONDS|none",
        help="Heartbeat timeout; default: none.",
    )
    execution.add_argument("--max-attempts", type=int, default=3)
    execution.add_argument("--state-file", default=default_state_file)


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        default=[],
        help=(
            "Task ID to run; repeat for multiple. If omitted, all tasks run. "
            "Dependencies are included automatically."
        ),
    )
    parser.add_argument("--list-tasks", action="store_true")
    _add_environment_arguments(parser)
    _add_agent_arguments(parser)

    execution = parser.add_argument_group("execution")
    execution.add_argument("--trials", type=int, default=1)
    execution.add_argument("--max-parallel", type=int, default=1)
    execution.add_argument("--max-parallel-per-task", type=int, default=1)
    execution.add_argument("--no-evaluate", action="store_true")
    execution.add_argument("--run-id")
    _add_temporal_arguments(
        execution,
        default_state_file=str(Path(".corral") / "benchmark-states.jsonl"),
    )
    execution.add_argument("--report")
    execution.add_argument("--verbose", action="store_true")


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, help="Independent task ID to run.")
    _add_environment_arguments(parser)
    _add_agent_arguments(parser)

    execution = parser.add_argument_group("execution")
    execution.add_argument("--execution-id")
    _add_temporal_arguments(
        execution,
        default_state_file=str(Path(".corral") / "run-states.jsonl"),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``corral`` command parser."""
    parser = argparse.ArgumentParser(
        prog="corral",
        description="Run scientific agents and benchmarks with Corral.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="Run one independent task without benchmark evaluation.",
    )
    _add_run_arguments(run)
    run.set_defaults(_handler=run_task)
    bench = commands.add_parser(
        "bench",
        help="Run an agent against a registered environment.",
    )
    _add_benchmark_arguments(bench)
    bench.set_defaults(_handler=run_benchmark)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the installed ``corral`` command."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(args._handler(args))
    except KeyboardInterrupt:
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
