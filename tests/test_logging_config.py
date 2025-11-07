"""Tests for the logging configuration module."""


import pytest
from loguru import logger

from corral import add_file_handler, get_logger, remove_handler, setup_logging


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset logger before each test."""
    # Remove all handlers
    logger.remove()
    yield
    # Clean up after test
    logger.remove()


def test_setup_logging_console_only():
    """Test basic console logging setup."""
    setup_logging(level="INFO", console=True)

    # Logger should have at least one handler
    # We can't directly check handlers but we can verify it doesn't raise
    logger.info("Test message")


def test_setup_logging_with_file(tmp_path):
    """Test logging to a file."""
    log_file = tmp_path / "test.log"

    setup_logging(level="DEBUG", console=False, log_file=str(log_file))

    logger.info("Test message")
    logger.debug("Debug message")

    # Verify file was created and contains messages
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content
    assert "Debug message" in content


def test_setup_logging_with_subsystems(tmp_path):
    """Test subsystem-specific logging."""
    setup_logging(
        level="INFO",
        console=False,
        log_dir=str(tmp_path),
        subsystem_files={
            "agents": "agents.log",
            "backend": "backend.log",
        },
    )

    # Log from different "subsystems" - we simulate this by logging
    logger.info("General message")

    # Verify log directory was created
    assert tmp_path.exists()
    assert (tmp_path / "agents.log").exists()
    assert (tmp_path / "backend.log").exists()


def test_get_logger():
    """Test getting a logger instance."""
    setup_logging(level="INFO")

    # Get default logger
    default_logger = get_logger()
    assert default_logger is not None

    # Get subsystem logger
    agents_logger = get_logger("agents")
    assert agents_logger is not None


def test_add_and_remove_file_handler(tmp_path):
    """Test adding and removing file handlers dynamically."""
    log_file = tmp_path / "dynamic.log"

    setup_logging(level="INFO", console=False)

    # Add a file handler
    handler_id = add_file_handler(str(log_file), level="DEBUG")

    logger.debug("Debug message")
    logger.info("Info message")

    # Verify file was created
    assert log_file.exists()
    content = log_file.read_text()
    assert "Debug message" in content
    assert "Info message" in content

    # Remove the handler
    remove_handler(handler_id)

    # Note: After removing, new messages won't be written to the file
    # but we can't easily test this without breaking the file handler


def test_custom_format(tmp_path):
    """Test custom log format."""
    log_file = tmp_path / "custom_format.log"

    custom_format = "{level} | {message}"

    setup_logging(
        level="INFO",
        console=False,
        log_file=str(log_file),
        format_string=custom_format,
    )

    logger.info("Test message")

    content = log_file.read_text()
    assert "INFO" in content
    assert "Test message" in content


def test_log_rotation_params(tmp_path):
    """Test that rotation parameters are accepted."""
    log_file = tmp_path / "rotating.log"

    # Should not raise any errors
    setup_logging(
        level="INFO",
        console=False,
        log_file=str(log_file),
        rotation="1 MB",
        retention="1 week",
        compression="zip",
    )

    logger.info("Test message")
    assert log_file.exists()


def test_add_file_handler_with_filter(tmp_path):
    """Test adding a file handler with a custom filter."""
    log_file = tmp_path / "filtered.log"

    setup_logging(level="DEBUG", console=False)

    # Filter that only allows ERROR and above
    def error_filter(record):
        return record["level"].name in ["ERROR", "CRITICAL"]

    handler_id = add_file_handler(
        str(log_file), level="DEBUG", filter_func=error_filter
    )

    logger.debug("Debug message")
    logger.info("Info message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Verify only error messages are in the file
    content = log_file.read_text()
    assert "Debug message" not in content
    assert "Info message" not in content
    assert "Error message" in content
    assert "Critical message" in content

    remove_handler(handler_id)


def test_multiple_handlers(tmp_path):
    """Test that multiple handlers can coexist."""
    log_file1 = tmp_path / "log1.log"
    log_file2 = tmp_path / "log2.log"

    setup_logging(level="INFO", console=False, log_file=str(log_file1))

    # Add another handler
    handler_id = add_file_handler(str(log_file2), level="DEBUG")

    logger.info("Test message")
    logger.debug("Debug message")

    # Both files should exist
    assert log_file1.exists()
    assert log_file2.exists()

    # log1 should have only INFO and above
    content1 = log_file1.read_text()
    assert "Test message" in content1

    # log2 should have DEBUG and above
    content2 = log_file2.read_text()
    assert "Test message" in content2
    assert "Debug message" in content2

    remove_handler(handler_id)
