"""Corral's structured, redacting Loguru configuration facade.

Application modules import :data:`logger` from here instead of configuring
Loguru themselves.  This keeps event fields, correlation context, redaction,
and standard-library interception consistent across the process.
"""

from __future__ import annotations

import json
import logging as stdlib_logging
import os
import re
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from loguru import logger as _loguru_logger

from corral.core.errors import concise_error_message

LogFormat = Literal["console", "json"]
LogSink = str | Path | Any

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = (
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "hiddenargument",
    "password",
    "privatekey",
    "secret",
)
_SENSITIVE_TOKEN_KEYS = {
    "accesstoken",
    "authtoken",
    "idtoken",
    "refreshtoken",
    "token",
}
_PAYLOAD_KEYS = {
    "arguments",
    "input",
    "messages",
    "output",
    "payload",
    "prompt",
    "reasoning",
    "result",
    "toolarguments",
    "toolresult",
}
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_KEY_PATTERN = re.compile(r"(?i)\b(?:sk|pk|api)[-_][A-Za-z0-9_-]{8,}\b")

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[subsystem]}</cyan> | "
    "<magenta>{extra[event]}</magenta> | "
    "<level>{message}</level>\n"
)


def _normalise_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _sensitive_key(key: object) -> bool:
    normalised = _normalise_key(key)
    return normalised in _SENSITIVE_TOKEN_KEYS or any(
        part in normalised for part in _SENSITIVE_KEY_PARTS
    )


def _payload_key(key: object) -> bool:
    return _normalise_key(key) in _PAYLOAD_KEYS


def redact_sensitive_data(value: Any, *, include_payloads: bool = False) -> Any:
    """Recursively redact credentials and opt-in diagnostic payloads.

    Redaction is applied in the logging patcher, so it also covers records from
    dependencies that enter through the standard-library interception bridge.
    Credential-shaped values remain redacted even when payload logging is on.
    """

    if isinstance(value, Mapping):
        return {
            str(key): (
                _REDACTED
                if _sensitive_key(key) or (_payload_key(key) and not include_payloads)
                else redact_sensitive_data(item, include_payloads=include_payloads)
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple | list | set | frozenset):
        return [
            redact_sensitive_data(item, include_payloads=include_payloads)
            for item in value
        ]
    if isinstance(value, str):
        redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
        return _KEY_PATTERN.sub(_REDACTED, redacted)
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class LogSinkConfig:
    """One Loguru sink, optionally restricted to selected subsystems."""

    sink: LogSink
    level: str = "INFO"
    format: LogFormat = "console"
    subsystems: tuple[str, ...] | None = None
    exclude_subsystems: tuple[str, ...] = ()
    enqueue: bool = False
    colorize: bool | None = None


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Process logging configuration with environment-compatible defaults."""

    level: str = "INFO"
    format: LogFormat = "console"
    console: bool = True
    file: str | Path | None = None
    file_level: str = "DEBUG"
    payloads: bool = False
    diagnostics: bool = False
    intercept_stdlib: bool = True
    sinks: tuple[LogSinkConfig, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> LoggingConfig:
        """Read the documented ``CORRAL_LOG_*`` environment variables."""

        raw_format = os.getenv("CORRAL_LOG_FORMAT", "console").strip().lower()
        log_format: LogFormat = "json" if raw_format == "json" else "console"
        return cls(
            level=os.getenv("CORRAL_LOG_LEVEL", "INFO"),
            format=log_format,
            file=os.getenv("CORRAL_LOG_FILE") or None,
            file_level=os.getenv("CORRAL_LOG_FILE_LEVEL", "DEBUG"),
            payloads=_env_bool("CORRAL_LOG_PAYLOADS"),
            diagnostics=_env_bool("CORRAL_LOG_DIAGNOSTICS"),
        )


class InterceptHandler(stdlib_logging.Handler):
    """Forward standard-library records to Loguru without changing severity."""

    def emit(self, record: stdlib_logging.LogRecord) -> None:
        try:
            level: str | int = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = stdlib_logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == stdlib_logging.__file__:
            frame = frame.f_back
            depth += 1
        fields = {
            key: record.__dict__[key]
            for key in (
                "activity",
                "attempt",
                "benchmark_run_id",
                "duration_ms",
                "error_message",
                "error_type",
                "event",
                "execution_id",
                "job_id",
                "status",
                "subsystem",
                "task_id",
                "workflow_id",
                "workflow_run_id",
            )
            if key in record.__dict__
        }
        fields.setdefault("subsystem", _stdlib_subsystem(record.name))
        fields.setdefault("event", "dependency.log")
        fields["stdlib_logger"] = record.name
        _loguru_logger.bind(**fields).opt(
            depth=depth,
            exception=record.exc_info,
        ).log(level, record.getMessage())


def _stdlib_subsystem(name: str) -> str:
    if name.startswith("temporalio"):
        return "orchestration"
    if name.startswith("corral"):
        return "runtime"
    return "dependency"


_configuration_lock = threading.RLock()
_handler_ids: list[int] = []
_intercept_handler: InterceptHandler | None = None
_configured_once = False
_include_payloads = False
_diagnostics = False


def _patch_record(record: dict[str, Any]) -> None:
    extra = record["extra"]
    extra.setdefault("event", "log.message")
    if extra.get("subsystem") in {None, "application"}:
        extra["subsystem"] = _corral_subsystem(str(record.get("name", "")))
    record["extra"] = redact_sensitive_data(
        extra,
        include_payloads=_include_payloads,
    )
    record["message"] = redact_sensitive_data(
        record["message"],
        include_payloads=_include_payloads,
    )
    if not _diagnostics:
        record["exception"] = None


def _corral_subsystem(module_name: str) -> str:
    if module_name.startswith("corral.orchestration"):
        return "orchestration"
    if module_name.startswith("corral.observability"):
        return "observability"
    if module_name.startswith("corral.persistence"):
        return "persistence"
    if module_name.startswith(("corral.evaluation", "corral.report")):
        return "evaluation"
    if module_name == "corral.utils.rag":
        return "rag"
    if module_name.startswith(("corral.backend", "corral.utils")):
        return "tool"
    if module_name.startswith("corral.agents"):
        return "agent"
    if module_name.startswith("corral"):
        return "runtime"
    return "application"


def _sink_filter(
    included: tuple[str, ...] | None,
    excluded: tuple[str, ...],
) -> Callable[[dict[str, Any]], bool]:
    included_set = frozenset(included or ())
    excluded_set = frozenset(excluded)

    def allow(record: dict[str, Any]) -> bool:
        subsystem = str(record["extra"].get("subsystem", "application"))
        return (
            not included_set or subsystem in included_set
        ) and subsystem not in excluded_set

    return allow


def _add_sink(config: LogSinkConfig, *, diagnostics: bool) -> int:
    log_format = (
        _CONSOLE_FORMAT + ("{exception}\n" if diagnostics else "")
        if config.format == "console"
        else "{message}"
    )
    return _loguru_logger.add(
        config.sink,
        level=config.level.upper(),
        format=log_format,
        filter=_sink_filter(config.subsystems, config.exclude_subsystems),
        serialize=config.format == "json",
        enqueue=config.enqueue,
        colorize=config.colorize,
        backtrace=False,
        diagnose=False,
    )


def _install_stdlib_intercept(enabled: bool) -> None:
    global _intercept_handler  # noqa: PLW0603
    root = stdlib_logging.getLogger()
    if _intercept_handler is not None:
        root.removeHandler(_intercept_handler)
        _intercept_handler = None
    if enabled:
        _intercept_handler = InterceptHandler()
        _intercept_handler.setLevel(stdlib_logging.NOTSET)
        root.addHandler(_intercept_handler)
        if root.level > stdlib_logging.DEBUG:
            root.setLevel(stdlib_logging.DEBUG)


def configure_logging(
    config: LoggingConfig | None = None,
    **overrides: Any,
) -> tuple[int, ...]:
    """Configure Corral sinks idempotently and return their handler IDs.

    Reconfiguration removes only handlers previously installed by Corral, so
    embedding applications can keep independent Loguru sinks. Keyword
    overrides are applied with :func:`dataclasses.replace` for concise setup.
    """

    global _configured_once, _diagnostics, _include_payloads  # noqa: PLW0603
    selected = config or LoggingConfig.from_env()
    if overrides:
        selected = replace(selected, **overrides)

    with _configuration_lock:
        for handler_id in _handler_ids:
            with _ignore_missing_handler():
                _loguru_logger.remove(handler_id)
        _handler_ids.clear()

        if not _configured_once:
            # Loguru's preconfigured stderr handler is not owned by Corral and
            # would otherwise duplicate every record after the first setup.
            with _ignore_missing_handler():
                _loguru_logger.remove(0)
            _configured_once = True

        _include_payloads = selected.payloads
        _diagnostics = selected.diagnostics
        _loguru_logger.configure(
            extra={"event": "log.message"},
            patcher=_patch_record,
        )

        sink_configs: list[LogSinkConfig] = list(selected.sinks)
        if selected.console:
            sink_configs.insert(
                0,
                LogSinkConfig(
                    sink=sys.stderr,
                    level=selected.level,
                    format=selected.format,
                ),
            )
        if selected.file is not None:
            sink_configs.append(
                LogSinkConfig(
                    sink=selected.file,
                    level=selected.file_level,
                    format=selected.format,
                    enqueue=True,
                )
            )
        _handler_ids.extend(
            _add_sink(sink, diagnostics=selected.diagnostics) for sink in sink_configs
        )
        _install_stdlib_intercept(selected.intercept_stdlib)

        if selected.payloads:
            event(
                "WARNING",
                "logging.payloads_enabled",
                subsystem="observability",
                message="Diagnostic payload logging is enabled and may contain sensitive data",
            )
        return tuple(_handler_ids)


@contextmanager
def _ignore_missing_handler() -> Iterator[None]:
    with suppress(ValueError):
        yield


def disable_logging() -> None:
    """Remove Corral-owned sinks and the standard-library interception bridge."""

    with _configuration_lock:
        for handler_id in _handler_ids:
            with _ignore_missing_handler():
                _loguru_logger.remove(handler_id)
        _handler_ids.clear()
        _install_stdlib_intercept(False)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Bind correlation fields for the current thread or async task only."""

    safe_fields = redact_sensitive_data(fields, include_payloads=_include_payloads)
    with _loguru_logger.contextualize(**safe_fields):
        yield


def exception_fields(exc: BaseException) -> dict[str, str]:
    """Return the standard concise fields for a handled exception."""

    return {
        "error_type": type(exc).__name__,
        "error_message": concise_error_message(exc),
    }


def event(
    level: str,
    name: str,
    *,
    subsystem: str,
    message: str | None = None,
    **fields: Any,
) -> None:
    """Emit one schema-compliant structured event."""

    _loguru_logger.bind(event=name, subsystem=subsystem, **fields).log(
        level.upper(),
        message or name,
    )


def json_record(record: Mapping[str, Any]) -> str:
    """Return a stable JSON representation useful to custom sink adapters."""

    return json.dumps(redact_sensitive_data(record), default=str, sort_keys=True)


# Public facade used throughout Corral. Configuration happens lazily at module
# import to preserve the project's historical default of visible INFO logs.
logger = _loguru_logger
configure_logging()


__all__ = [
    "InterceptHandler",
    "LogSinkConfig",
    "LoggingConfig",
    "configure_logging",
    "disable_logging",
    "event",
    "exception_fields",
    "json_record",
    "log_context",
    "logger",
    "redact_sensitive_data",
]
