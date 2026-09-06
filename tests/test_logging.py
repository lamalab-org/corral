"""Contract tests for Corral's structured logging facade."""

from __future__ import annotations

import ast
import asyncio
import io
import json
import logging
from pathlib import Path

import pytest

from corral.observability import LoggingObserver, Observation, ObservationContext
from corral.observability.langfuse import observer_from_env
from corral.report.logging import (
    LoggingConfig,
    LogSinkConfig,
    configure_logging,
    event,
    log_context,
    logger,
)

TASK_ENVIRONMENT_MODULES = (
    "tasks/afm/src/env.py",
    "tasks/catalyst/src/catalyst/env.py",
    "tasks/corral_md/src/corral_md/env.py",
    "tasks/ml/src/ml/env.py",
    "tasks/resistor_network/src/resistor_network/env.py",
    "tasks/retrosynthesis/retrosynthesis/env.py",
    "tasks/samplemath/samplemath/env_subtask.py",
    "tasks/spectra_elucidation/spectra_elucidation/env.py",
    "tasks/wetlab/wetlab/env.py",
)


@pytest.fixture(autouse=True)
def _restore_default_logging():
    yield
    configure_logging()


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def _capture(*, level: str = "DEBUG"):
    records = []
    configure_logging(
        LoggingConfig(
            console=False,
            sinks=(LogSinkConfig(records.append, level=level),),
        )
    )
    return records


def test_configuration_is_idempotent_without_duplicate_handlers():
    output = io.StringIO()
    config = LoggingConfig(
        console=False,
        sinks=(LogSinkConfig(output, level="INFO"),),
    )
    configure_logging(config)
    configure_logging(config)

    event("INFO", "logging.idempotent", subsystem="observability")

    assert output.getvalue().count("logging.idempotent") == 2  # event + message


def test_console_and_json_formats_include_the_event_schema():
    console = io.StringIO()
    configure_logging(
        LoggingConfig(
            console=False,
            sinks=(LogSinkConfig(console, format="console"),),
        )
    )
    event("INFO", "task.started", subsystem="runtime", task_id="task-a")
    assert "runtime | task.started | task.started" in console.getvalue()

    structured = io.StringIO()
    configure_logging(
        LoggingConfig(
            console=False,
            sinks=(LogSinkConfig(structured, format="json"),),
        )
    )
    event("INFO", "task.completed", subsystem="runtime", task_id="task-a")
    payload = json.loads(structured.getvalue())
    assert (
        payload["record"]["extra"].items()
        >= {
            "event": "task.completed",
            "subsystem": "runtime",
            "task_id": "task-a",
        }.items()
    )


def test_sink_routing_filters_by_subsystem():
    runtime_output = io.StringIO()
    rag_output = io.StringIO()
    configure_logging(
        LoggingConfig(
            console=False,
            sinks=(
                LogSinkConfig(runtime_output, subsystems=("runtime",)),
                LogSinkConfig(rag_output, subsystems=("rag",)),
            ),
        )
    )

    event("INFO", "task.started", subsystem="runtime")
    event("INFO", "rag.database_started", subsystem="rag")

    assert "task.started" in runtime_output.getvalue()
    assert "rag.database_started" not in runtime_output.getvalue()
    assert "rag.database_started" in rag_output.getvalue()
    assert "task.started" not in rag_output.getvalue()


@pytest.mark.anyio()
async def test_context_is_isolated_across_concurrent_tasks():
    records = _capture()

    async def emit_for(execution_id: str) -> None:
        with log_context(execution_id=execution_id):
            await asyncio.sleep(0)
            event("INFO", "context.test", subsystem="runtime")

    await asyncio.gather(emit_for("execution-a"), emit_for("execution-b"))

    context_records = [
        item.record
        for item in records
        if item.record["extra"]["event"] == "context.test"
    ]
    assert {item["extra"]["execution_id"] for item in context_records} == {
        "execution-a",
        "execution-b",
    }


def test_sensitive_fields_and_payloads_are_redacted_centrally():
    records = _capture()

    event(
        "DEBUG",
        "agent.tool_call",
        subsystem="agent",
        authorization="Bearer top-secret",
        token="secret-token",
        prompt="private benchmark prompt",
        nested={"api_key": "sk-1234567890"},
    )

    extra = records[-1].record["extra"]
    assert extra["authorization"] == "[REDACTED]"
    assert extra["token"] == "[REDACTED]"
    assert extra["prompt"] == "[REDACTED]"
    assert extra["nested"]["api_key"] == "[REDACTED]"


def test_payload_opt_in_keeps_payloads_but_never_credentials():
    records = []
    configure_logging(
        LoggingConfig(
            console=False,
            payloads=True,
            sinks=(LogSinkConfig(records.append, level="DEBUG"),),
        )
    )

    event(
        "DEBUG",
        "agent.tool_call",
        subsystem="agent",
        prompt="diagnostic prompt",
        arguments={"query": "value", "api_key": "sk-1234567890"},
    )

    extra = records[-1].record["extra"]
    assert extra["prompt"] == "diagnostic prompt"
    assert extra["arguments"] == {
        "query": "value",
        "api_key": "[REDACTED]",
    }
    assert any(
        item.record["extra"]["event"] == "logging.payloads_enabled" for item in records
    )


def test_normal_configuration_does_not_render_tracebacks():
    output = io.StringIO()
    configure_logging(
        LoggingConfig(
            console=False,
            diagnostics=False,
            sinks=(LogSinkConfig(output),),
        )
    )

    try:
        raise ValueError("concise only")
    except ValueError:
        logger.opt(exception=True).error("operation failed")

    rendered = output.getvalue()
    assert "operation failed" in rendered
    assert "Traceback" not in rendered


def test_standard_library_records_are_intercepted_with_schema():
    records = _capture()

    logging.getLogger("dependency.example").warning("dependency degraded")

    record = records[-1].record
    assert record["extra"]["event"] == "dependency.log"
    assert record["extra"]["subsystem"] == "dependency"
    assert record["message"] == "dependency degraded"


def test_logging_observer_emits_one_failure_isolated_operation_record():
    records = _capture()
    observer = LoggingObserver()
    context = ObservationContext(execution_id="execution-a", task_id="task-a")
    span = observer.start(Observation(name="task.run", context=context))
    span.end(ValueError("Invalid parameter: response_format"))

    operation_records = [
        item.record
        for item in records
        if item.record["extra"]["event"] == "observation.task.run.completed"
    ]
    assert len(operation_records) == 1
    assert operation_records[0]["level"].name == "WARNING"
    assert operation_records[0]["extra"]["error_message"] == (
        "Invalid parameter: response_format"
    )


def test_operation_failure_warns_without_affecting_runtime_logging():
    records = _capture()
    observer = LoggingObserver()
    context = ObservationContext(execution_id="execution-a", task_id="task-a")

    span = observer.start(Observation(name="task.run", context=context))
    span.end(ConnectionError("temporary outage"))

    operation_records = [
        item.record
        for item in records
        if item.record["extra"]["event"] == "observation.task.run.completed"
    ]
    assert len(operation_records) == 1
    assert operation_records[0]["level"].name == "WARNING"
    assert operation_records[0]["extra"]["error_type"] == "ConnectionError"


def test_langfuse_initialization_failure_warns_and_keeps_local_observer(
    monkeypatch,
):
    records = _capture()
    monkeypatch.setenv("CORRAL_LANGFUSE_ENABLED", "true")

    def fail():
        raise ConnectionError("backend unavailable")

    monkeypatch.setattr("corral.observability.langfuse.LangfuseObserver", fail)

    observer = observer_from_env()

    assert isinstance(observer, LoggingObserver)
    warnings = [
        item.record
        for item in records
        if item.record["extra"]["event"] == "observability.initialization_failed"
    ]
    assert len(warnings) == 1
    assert warnings[0]["level"].name == "WARNING"


def test_corral_modules_use_only_the_logging_facade():
    package_root = Path(__file__).parents[1] / "src" / "corral"
    violations: list[str] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "loguru"
                and path.name != "logging.py"
            ):
                violations.append(str(path.relative_to(package_root)))
            if isinstance(node, ast.Import):
                violations.extend(
                    str(path.relative_to(package_root))
                    for alias in node.names
                    if alias.name == "logging" and path.name != "logging.py"
                )
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "logging"
                and node.level == 0
                and path.name != "logging.py"
            ):
                violations.append(str(path.relative_to(package_root)))
    assert violations == []


def test_task_environments_follow_the_logging_convention():
    repository_root = Path(__file__).parents[1]
    direct_logging_imports: list[str] = []
    direct_logger_calls: list[str] = []

    for relative_path in TASK_ENVIRONMENT_MODULES:
        path = repository_root / relative_path
        tree = ast.parse(path.read_text(), filename=str(path))
        environment_events: list[tuple[str, str, ast.Call]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                "logging",
                "loguru",
            }:
                direct_logging_imports.append(relative_path)
            if isinstance(node, ast.Import) and any(
                alias.name in {"logging", "loguru"} for alias in node.names
            ):
                direct_logging_imports.append(relative_path)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                direct_logger_calls.append(relative_path)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "event"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[1], ast.Constant)
            ):
                environment_events.append(
                    (str(node.args[0].value), str(node.args[1].value), node)
                )

        event_names = [name for _, name, _ in environment_events]
        assert event_names.count("environment.started") == 1, relative_path
        assert event_names.count("environment.completed") == 1, relative_path
        assert event_names.count("environment.failed") == 1, relative_path

        lifecycle_events = {
            name: (level, call)
            for level, name, call in environment_events
            if name
            in {
                "environment.started",
                "environment.completed",
                "environment.failed",
            }
        }
        assert lifecycle_events["environment.started"][0] == "INFO", relative_path
        assert lifecycle_events["environment.completed"][0] == "INFO", relative_path
        assert lifecycle_events["environment.failed"][0] == "ERROR", relative_path
        for terminal_name in ("environment.completed", "environment.failed"):
            keywords = {
                keyword.arg for keyword in lifecycle_events[terminal_name][1].keywords
            }
            assert {"duration_ms", "status"} <= keywords, relative_path

        prompt_events = [
            (level, name)
            for level, name, call in environment_events
            if any(keyword.arg == "prompt" for keyword in call.keywords)
        ]
        assert all(level == "DEBUG" for level, _ in prompt_events), relative_path

    assert direct_logging_imports == []
    assert direct_logger_calls == []
