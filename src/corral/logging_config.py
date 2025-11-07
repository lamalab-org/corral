"""
Centralized logging configuration for Corral framework.

This module provides a simple logging setup using loguru for console and file logging.

Example usage:
    >>> from corral.logging_config import setup_logging
    >>>
    >>> # Basic setup with console output
    >>> setup_logging()
    >>>
    >>> # Setup with file logging
    >>> setup_logging(level="INFO", log_file="benchmark.log")
"""

import sys

from loguru import logger


def setup_logging(
    level: str = "INFO",
    console: bool = True,
    log_file: str | None = None,
    format_string: str | None = None,
    rotation: str | None = None,
    retention: str | None = None,
    compression: str | None = None,
) -> None:
    """
    Configure logging for the Corral framework.

    This function sets up loguru with optional handlers for console output
    and file logging. By default, logs are not rotated or deleted to preserve
    important benchmark data.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    console : bool, default=True
        Whether to output logs to console (stderr)
    log_file : str or None, default=None
        Path to a log file. If None, no file logging.
    format_string : str or None, default=None
        Custom format string for log messages. If None, uses default format.
    rotation : str or None, default=None
        When to rotate log files (e.g., "500 MB", "1 day", "1 week").
        Default is None (no rotation) to preserve benchmark data.
    retention : str or None, default=None
        How long to keep rotated log files. Default is None (keep forever).
    compression : str or None, default=None
        Compression format for rotated logs (e.g., "zip", "gz", "bz2").
        Default is None (no compression).

    Examples
    --------
    Basic console logging:
        >>> setup_logging()

    Console + file logging (no rotation - preserves benchmark data):
        >>> setup_logging(level="INFO", log_file="benchmark.log")

    With rotation (for non-benchmark logs):
        >>> setup_logging(
        ...     level="DEBUG",
        ...     log_file="server.log",
        ...     rotation="100 MB",
        ...     retention="30 days",
        ...     compression="gz"
        ... )

    Custom format:
        >>> setup_logging(
        ...     format_string=(
        ...         "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        ...         "<level>{level: <8}</level> | "
        ...         "<level>{message}</level>"
        ...     )
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

    # Add log file handler if requested
    if log_file:
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
        )


def add_file_handler(
    filepath: str,
    level: str = "INFO",
    format_string: str | None = None,
    rotation: str | None = None,
    retention: str | None = None,
    compression: str | None = None,
    filter_func=None,
) -> int:
    """
    Add a custom file handler to the logger.

    This allows you to add additional log handlers after initial setup.
    By default, logs are not rotated to preserve important data.

    Parameters
    ----------
    filepath : str
        Path to the log file
    level : str, default="INFO"
        Logging level for this handler
    format_string : str or None, default=None
        Custom format string. If None, uses a default format.
    rotation : str or None, default=None
        When to rotate the log file (e.g., "100 MB", "1 day").
        Default is None (no rotation).
    retention : str or None, default=None
        How long to keep rotated files. Default is None (keep forever).
    compression : str or None, default=None
        Compression format for rotated logs (e.g., "zip", "gz").
        Default is None (no compression).
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

    return logger.add(
        filepath,
        format=format_string,
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        filter=filter_func,
    )


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
    Change the logging level by resetting to a simple console handler.

    **Warning**: This function removes all existing handlers (including any
    custom file handlers or subsystem-specific handlers) and replaces them
    with a single console handler at the specified level. This is a
    destructive operation.

    For more fine-grained control, use `setup_logging()` with your desired
    configuration, or manually manage handlers with `add_file_handler()`
    and `remove_handler()`.

    Parameters
    ----------
    level : str
        New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Examples
    --------
    >>> set_level("DEBUG")

    Notes
    -----
    This is a convenience function for simple use cases. If you need to
    preserve existing handler configurations while changing the level,
    consider using `setup_logging()` again with your desired parameters.
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
