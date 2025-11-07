"""
Centralized logging configuration for Corral framework.

This module provides a flexible logging setup using loguru that allows different
handlers to be configured for different subsystems (agents, backend, router, utils, report).

Example usage:
    >>> from corral.logging_config import setup_logging, get_logger
    >>> 
    >>> # Basic setup with console output
    >>> setup_logging()
    >>> 
    >>> # Advanced setup with separate log files for subsystems
    >>> setup_logging(
    ...     level="INFO",
    ...     log_dir="./logs",
    ...     subsystem_files={
    ...         "agents": "agents.log",
    ...         "backend": "backend.log",
    ...         "router": "router.log",
    ...     }
    ... )
    >>> 
    >>> # Get a logger for a specific subsystem
    >>> logger = get_logger("agents")
    >>> logger.info("Agent started")
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger


# Store original logger for module-level loggers
_module_loggers: dict[str, Any] = {}


def setup_logging(
    level: str = "INFO",
    console: bool = True,
    log_file: str | None = None,
    log_dir: str | None = None,
    subsystem_files: dict[str, str] | None = None,
    format_string: str | None = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    compression: str = "zip",
) -> None:
    """
    Configure logging for the Corral framework.

    This function sets up loguru with optional handlers for console output,
    general log files, and subsystem-specific log files.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    console : bool, default=True
        Whether to output logs to console (stderr)
    log_file : str or None, default=None
        Path to a general log file. If None, no general file logging.
    log_dir : str or None, default=None
        Directory for log files. Used with subsystem_files.
    subsystem_files : dict[str, str] or None, default=None
        Dictionary mapping subsystem names to log file names.
        Example: {"agents": "agents.log", "backend": "backend.log"}
        Logs will be filtered to only include messages from that subsystem.
    format_string : str or None, default=None
        Custom format string for log messages. If None, uses default format.
    rotation : str, default="10 MB"
        When to rotate log files (e.g., "500 MB", "1 day", "1 week")
    retention : str, default="1 week"
        How long to keep rotated log files
    compression : str, default="zip"
        Compression format for rotated logs (e.g., "zip", "gz", "bz2")

    Examples
    --------
    Basic console logging:
        >>> setup_logging()

    Console + file logging:
        >>> setup_logging(level="DEBUG", log_file="corral.log")

    Subsystem-specific logging:
        >>> setup_logging(
        ...     level="INFO",
        ...     log_dir="./logs",
        ...     subsystem_files={
        ...         "agents": "agents.log",
        ...         "backend": "backend.log",
        ...         "router": "router.log",
        ...         "utils": "utils.log",
        ...         "report": "report.log",
        ...     }
        ... )

    Custom format:
        >>> setup_logging(
        ...     format_string="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        ...                  "<level>{level: <8}</level> | "
        ...                  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        ...                  "<level>{message}</level>"
        ... )
    """
    # Remove default handler
    logger.remove()

    # Default format if none provided
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler if requested
    if console:
        logger.add(
            sys.stderr,
            format=format_string,
            level=level,
            colorize=True,
        )

    # Add general log file handler if requested
    if log_file:
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
        )

    # Add subsystem-specific handlers
    if subsystem_files and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        for subsystem, filename in subsystem_files.items():
            file_path = log_path / filename

            # Create a filter function for this subsystem
            def make_filter(subsystem_name: str):
                def filter_func(record):
                    # Check if the logger name starts with the subsystem module path
                    return record["name"].startswith(f"corral.{subsystem_name}")

                return filter_func

            logger.add(
                str(file_path),
                format=format_string,
                level=level,
                filter=make_filter(subsystem),
                rotation=rotation,
                retention=retention,
                compression=compression,
            )


def get_logger(subsystem: str | None = None):
    """
    Get a logger instance, optionally bound to a specific subsystem.

    This function returns the global loguru logger. If a subsystem is specified,
    the logger's context will be bound to include the subsystem name in the logs.

    Parameters
    ----------
    subsystem : str or None, default=None
        Name of the subsystem (e.g., "agents", "backend", "router", "utils", "report")

    Returns
    -------
    logger
        A loguru logger instance

    Examples
    --------
    >>> from corral.logging_config import get_logger
    >>> logger = get_logger("agents")
    >>> logger.info("Starting agent execution")

    Notes
    -----
    The subsystem parameter is used to help filter logs when subsystem-specific
    log files are configured via setup_logging().
    """
    if subsystem:
        # Return a logger bound with the subsystem context
        # The binding doesn't change the logger itself, but adds context
        return logger.bind(subsystem=subsystem)
    return logger


def add_file_handler(
    filepath: str,
    level: str = "INFO",
    format_string: str | None = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    compression: str = "zip",
    filter_func=None,
) -> int:
    """
    Add a custom file handler to the logger.

    This allows you to add additional log handlers after initial setup.

    Parameters
    ----------
    filepath : str
        Path to the log file
    level : str, default="INFO"
        Logging level for this handler
    format_string : str or None, default=None
        Custom format string. If None, uses a default format.
    rotation : str, default="10 MB"
        When to rotate the log file
    retention : str, default="1 week"
        How long to keep rotated files
    compression : str, default="zip"
        Compression format for rotated logs
    filter_func : callable or None, default=None
        Optional filter function to determine which records to log

    Returns
    -------
    int
        Handler ID that can be used to remove this handler later

    Examples
    --------
    >>> handler_id = add_file_handler("debug.log", level="DEBUG")
    >>> # ... do some logging ...
    >>> remove_handler(handler_id)
    """
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    handler_id = logger.add(
        filepath,
        format=format_string,
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        filter=filter_func,
    )
    return handler_id


def remove_handler(handler_id: int) -> None:
    """
    Remove a handler by its ID.

    Parameters
    ----------
    handler_id : int
        The handler ID returned by add_file_handler()

    Examples
    --------
    >>> handler_id = add_file_handler("temp.log")
    >>> remove_handler(handler_id)
    """
    logger.remove(handler_id)


def set_level(level: str) -> None:
    """
    Change the logging level for all handlers.

    Note: This removes all existing handlers and re-adds a console handler
    with the new level. For more control, use setup_logging() or add custom handlers.

    Parameters
    ----------
    level : str
        New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Examples
    --------
    >>> set_level("DEBUG")
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=level,
        colorize=True,
    )
