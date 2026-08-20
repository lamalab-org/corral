"""Tests for the installed benchmark CLI and legacy script compatibility."""

import argparse
import asyncio
import importlib.util
from pathlib import Path

import pytest
from temporalio.testing import WorkflowEnvironment

from corral import cli
from corral.agents.ai_scientist import AIScientistConfig, SakanaAIScientistConfig
from corral.agents.ai_scientist.agent import AIScientistAgent
from corral.agents.llm_planner import LLMPlanner
from corral.agents.react import ReActAgent
from corral.agents.reflexion_agent import ReflexionAgent
from corral.agents.schema import AgentOutcome
from corral.agents.terminus import TerminusAgent
from corral.agents.tool_calling import ToolCallingAgent
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.environment import Environment, Toolset
from corral.core.task import InputRef, TaskDefinition


def _load_legacy_script():
    path = Path(__file__).parents[1] / "run_scripts" / "run_tool_calling.py"
    specification = importlib.util.spec_from_file_location(
        "corral_test_run_tool_calling",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


run_tool_calling = _load_legacy_script()

EXPECTED_AGENTS = (
    "ai-scientist",
    "claude-code",
    "codex",
    "llm-planner",
    "openhands",
    "react",
    "reflexion",
    "terminus",
    "tool-calling",
)


class SubmitAgent:
    model = "test-model"

    async def run_session(self, session):
        result = await session.execute(
            Action(name=SUBMIT_ANSWER_TOOL_NAME, arguments={"answer": "42"})
        )
        assert result.success is True
        return AgentOutcome(status="completed", answer="42")


def _unexpected_score(_answer):
    raise AssertionError("corral run must not invoke the task scorer")


def _environment(task_id, *dependencies):
    return Environment(
        task_id,
        TaskDefinition(
            name=task_id,
            description=f"Run {task_id}",
            tools=[],
            scoring_fn=_unexpected_score,
            submission_format={"answer": "string"},
            input_map={dependency: InputRef(dependency) for dependency in dependencies},
            resolve_answer=False,
        ),
        toolset=Toolset(workspace_factory=None),
    )


def test_every_public_concrete_agent_is_registered():
    assert cli.AGENT_NAMES == EXPECTED_AGENTS
    assert {definition.attribute for definition in cli.AGENT_DEFINITIONS.values()} == {
        "AIScientistAgent",
        "ClaudeCodeAgent",
        "CodexAgent",
        "LLMPlanner",
        "OpenHandsAgent",
        "ReActAgent",
        "ReflexionAgent",
        "TerminusAgent",
        "ToolCallingAgent",
    }


@pytest.mark.parametrize(
    ("name", "agent_class"),
    [
        ("ai-scientist", AIScientistAgent),
        ("llm-planner", LLMPlanner),
        ("react", ReActAgent),
        ("terminus", TerminusAgent),
        ("tool-calling", ToolCallingAgent),
    ],
)
def test_create_agent_constructs_core_agents(name, agent_class):
    agent = cli.create_agent(name, model="openai/test-model")

    assert isinstance(agent, agent_class)
    assert agent.model == "openai/test-model"


def test_create_agent_accepts_class_and_separator_aliases():
    assert cli.normalise_agent_name("ToolCallingAgent") == "tool-calling"
    assert cli.normalise_agent_name("tool_calling") == "tool-calling"
    assert cli.normalise_agent_name("ClaudeCodeAgent") == "claude-code"
    assert cli.normalise_agent_name("AI_Scientist") == "ai-scientist"


def test_tool_calling_uses_cli_safe_reasoning_default_and_allows_override():
    default_agent = cli.create_agent("tool-calling")
    overridden_agent = cli.create_agent(
        "tool-calling",
        agent_kwargs={"reasoning_effort": "low"},
    )

    assert default_agent.kwargs["reasoning_effort"] == "none"
    assert overridden_agent.kwargs["reasoning_effort"] == "low"


def test_reflexion_builds_a_configurable_actor():
    agent = cli.create_agent(
        "reflexion",
        model="openai/test-model",
        agent_kwargs={"actor": "react", "actor_kwargs": {"seed": 4}},
    )

    assert isinstance(agent, ReflexionAgent)
    assert isinstance(agent.actor, ReActAgent)
    assert agent.actor.model == "openai/test-model"
    assert agent.actor.kwargs["seed"] == 4
    assert agent.model == "openai/test-model"


def test_reflexion_accepts_common_values_from_agent_kwargs():
    agent = cli.create_agent(
        "reflexion",
        agent_kwargs={
            "model": "openai/actor-model",
            "reflection_model": "openai/reflection-model",
        },
    )

    assert agent.actor.model == "openai/actor-model"
    assert agent.model == "openai/reflection-model"


def test_ai_scientist_builds_config_from_json_values():
    agent = cli.create_agent(
        "ai-scientist",
        agent_kwargs={"config": {"parallel_llm_workers": 1}},
    )

    assert isinstance(agent.config, AIScientistConfig)
    assert agent.config.parallel_llm_workers == 1


def test_ai_scientist_supports_sakana_config_profile():
    agent = cli.create_agent(
        "ai-scientist",
        agent_kwargs={"config_profile": "sakana"},
    )

    assert isinstance(agent.config, SakanaAIScientistConfig)


@pytest.mark.parametrize("name", ["claude-code", "codex", "openhands"])
def test_optional_agents_are_loaded_only_when_selected(monkeypatch, name):
    definition = cli.AGENT_DEFINITIONS[name]
    requested = []

    class FakeAgent:
        def __init__(self, model="test-default", **kwargs):
            self.model = model
            self.kwargs = kwargs

    class FakeModule:
        pass

    module = FakeModule()
    setattr(module, definition.attribute, FakeAgent)

    def fake_import(module_name):
        requested.append(module_name)
        return module

    monkeypatch.setattr(cli, "import_module", fake_import)

    agent = cli.create_agent(name, model="test-model")

    assert isinstance(agent, FakeAgent)
    assert agent.model == "test-model"
    assert requested == [definition.module]


def test_benchmark_parser_accepts_agent_environment_model_and_json_options():
    args = cli.build_parser().parse_args(
        [
            "bench",
            "--agent",
            "ReActAgent",
            "--environment",
            "samplemath",
            "--model",
            "openai/test-model",
            "--task",
            "task1",
            "--env-kwargs",
            '{"level": 2}',
            "--agent-kwargs",
            '{"seed": 7}',
        ]
    )

    assert args.agent == "react"
    assert args.environment == "samplemath"
    assert args.model == "openai/test-model"
    assert args.tasks == ["task1"]
    assert args.env_kwargs == {"level": 2}
    assert args.agent_kwargs == {"seed": 7}


def test_run_parser_requires_one_task_and_has_non_benchmark_defaults():
    args = cli.build_parser().parse_args(
        [
            "run",
            "--agent",
            "ReActAgent",
            "--environment",
            "samplemath",
            "--task",
            "task1",
            "--model",
            "openai/test-model",
        ]
    )

    assert args.command == "run"
    assert args.agent == "react"
    assert args.task == "task1"
    assert args.model == "openai/test-model"
    assert args.commit_file == ".corral/run-commits.sqlite3"
    assert not hasattr(args, "trials")
    assert not hasattr(args, "report")


def test_run_rejects_unknown_task(monkeypatch):
    monkeypatch.setattr(
        cli,
        "load_environment_group",
        lambda *args, **kwargs: {"task1": _environment("task1")},
    )
    args = cli.build_parser().parse_args(
        [
            "run",
            "--agent",
            "react",
            "--environment",
            "samplemath",
            "--task",
            "missing",
        ]
    )

    with pytest.raises(ValueError, match="unknown task 'missing'"):
        asyncio.run(cli.run_task(args))


def test_run_rejects_task_dependencies(monkeypatch):
    monkeypatch.setattr(
        cli,
        "load_environment_group",
        lambda *args, **kwargs: {"task2": _environment("task2", "task1")},
    )
    args = cli.build_parser().parse_args(
        [
            "run",
            "--agent",
            "react",
            "--environment",
            "samplemath",
            "--task",
            "task2",
        ]
    )

    with pytest.raises(ValueError, match="use `corral bench` for dependency graphs"):
        asyncio.run(cli.run_task(args))


def test_run_executes_one_task_without_a_benchmark(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        cli,
        "load_environment_group",
        lambda *args, **kwargs: {"task1": _environment("task1")},
    )
    monkeypatch.setattr(cli, "create_agent", lambda *args, **kwargs: SubmitAgent())
    args = cli.build_parser().parse_args(
        [
            "run",
            "--agent",
            "react",
            "--environment",
            "samplemath",
            "--task",
            "task1",
            "--execution-id",
            "cli-direct-task",
            "--commit-file",
            str(tmp_path / "run-commits.sqlite3"),
        ]
    )

    async def scenario():
        async with await WorkflowEnvironment.start_time_skipping() as temporal:

            async def connect(*args, **kwargs):
                return temporal.client

            monkeypatch.setattr(cli.Client, "connect", connect)
            return await cli.run_task(args)

    result = asyncio.run(scenario())

    assert result == 0
    output = capsys.readouterr().out
    assert "Run cli-direct-task completed:" in output
    assert "- task: task1" in output
    assert "- status: submitted" in output
    assert '- answer: "42"' in output


def test_unknown_agent_error_lists_available_agents():
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(
            [
                "bench",
                "--agent",
                "missing",
                "--environment",
                "samplemath",
            ]
        )

    assert exc_info.value.code == 2


def test_benchmark_is_not_a_command_name():
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["benchmark"])

    assert exc_info.value.code == 2


def test_legacy_script_delegates_to_tool_calling_agent(monkeypatch):
    calls = []

    async def fake_run_benchmark(args, *, agent_name=None, agent_kwargs=None):
        calls.append((args, agent_name, agent_kwargs))
        return 7

    monkeypatch.setattr(run_tool_calling, "run_benchmark", fake_run_benchmark)
    args = argparse.Namespace()

    result = asyncio.run(run_tool_calling.run(args))

    assert result == 7
    assert calls == [(args, "tool-calling", None)]


def test_legacy_script_retains_existing_defaults():
    args = run_tool_calling.build_parser().parse_args(["--environment", "samplemath"])

    assert args.model == "openai/gpt-5.6-terra"
    assert args.temperature == 1.0
    assert args.max_iterations == 20
    assert args.trials == 1
