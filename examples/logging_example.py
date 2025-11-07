#!/usr/bin/env python3
"""
Example script demonstrating Corral's logging framework.

This script shows different logging configurations and patterns.
"""

import time
from pathlib import Path

from corral import add_file_handler, get_logger, remove_handler, setup_logging
from loguru import logger


def example_basic_logging():
    """Example 1: Basic console logging"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Console Logging")
    print("=" * 60)

    setup_logging(level="INFO")

    logger.info("This is an info message")
    logger.debug("This debug message won't appear (level is INFO)")
    logger.warning("This is a warning")
    logger.error("This is an error")
    logger.success("This is a success message")


def example_file_logging():
    """Example 2: Logging to a file"""
    print("\n" + "=" * 60)
    print("Example 2: File Logging")
    print("=" * 60)

    # Create a logs directory
    Path("./example_logs").mkdir(exist_ok=True)

    setup_logging(
        level="DEBUG",
        console=True,
        log_file="./example_logs/general.log",
    )

    logger.info("Logging to both console and file")
    logger.debug("This debug message appears in both")
    logger.warning("Check ./example_logs/general.log for the full log")


def example_subsystem_logging():
    """Example 3: Subsystem-specific logging"""
    print("\n" + "=" * 60)
    print("Example 3: Subsystem-Specific Logging")
    print("=" * 60)

    Path("./example_logs").mkdir(exist_ok=True)

    setup_logging(
        level="INFO",
        console=True,
        log_dir="./example_logs",
        subsystem_files={
            "agents": "agents.log",
            "backend": "backend.log",
            "router": "router.log",
        },
    )

    # Simulate different subsystem loggers
    # Note: In actual code, these would be in their respective modules
    # We're importing from those modules here to simulate it

    # Import from actual modules to demonstrate filtering
    try:
        from corral.agents.utils import logger as agents_logger

        agents_logger.info("This goes to agents.log")
    except ImportError:
        logger.info("Could not import agents module")

    try:
        from corral.backend.server import logger as backend_logger

        backend_logger.info("This goes to backend.log")
    except ImportError:
        logger.info("Could not import backend module")

    try:
        from corral.router.routes import logger as router_logger

        router_logger.info("This goes to router.log")
    except ImportError:
        logger.info("Could not import router module")

    logger.info("Check ./example_logs/ for subsystem-specific logs")


def example_custom_handler():
    """Example 4: Adding custom handlers dynamically"""
    print("\n" + "=" * 60)
    print("Example 4: Dynamic Custom Handlers")
    print("=" * 60)

    Path("./example_logs").mkdir(exist_ok=True)

    setup_logging(level="INFO", console=True)

    # Add a temporary debug handler
    logger.info("Adding a temporary debug handler")
    handler_id = add_file_handler(
        "./example_logs/debug_session.log", level="DEBUG", rotation="1 MB"
    )

    logger.debug("This appears in debug_session.log but not console")
    logger.info("This appears in both console and debug_session.log")

    # Remove the handler
    logger.info("Removing the debug handler")
    remove_handler(handler_id)

    logger.debug("This debug message won't be saved to debug_session.log anymore")


def example_custom_format():
    """Example 5: Custom log format"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Log Format")
    print("=" * 60)

    setup_logging(
        level="INFO",
        console=True,
        format_string=(
            "{time:HH:mm:ss} | " "{level: <8} | " "{function}:{line} - " "{message}"
        ),
    )

    logger.info("This message has a custom format")
    logger.warning("Notice the simplified timestamp and layout")


def example_error_filtering():
    """Example 6: Filtering only errors to a separate file"""
    print("\n" + "=" * 60)
    print("Example 6: Error-Only Filtering")
    print("=" * 60)

    Path("./example_logs").mkdir(exist_ok=True)

    setup_logging(level="DEBUG", console=True)

    # Add a handler that only logs errors
    def error_filter(record):
        return record["level"].name in ["ERROR", "CRITICAL"]

    error_handler = add_file_handler(
        "./example_logs/errors_only.log", level="DEBUG", filter_func=error_filter
    )

    logger.debug("This won't go to errors_only.log")
    logger.info("This won't go to errors_only.log either")
    logger.error("But this WILL go to errors_only.log")
    logger.critical("And so will this")

    logger.info("Check ./example_logs/errors_only.log - it only has errors!")

    remove_handler(error_handler)


def example_production_setup():
    """Example 7: Production-ready configuration"""
    print("\n" + "=" * 60)
    print("Example 7: Production Configuration")
    print("=" * 60)

    Path("./example_logs").mkdir(exist_ok=True)

    setup_logging(
        level="INFO",
        console=True,
        log_file="./example_logs/production.log",
        log_dir="./example_logs",
        subsystem_files={
            "agents": "agents.log",
            "backend": "backend.log",
            "router": "router.log",
            "utils": "utils.log",
            "report": "report.log",
        },
        rotation="10 MB",
        retention="1 week",
        compression="zip",
    )

    logger.info("Production logging configured")
    logger.info("- Console output enabled")
    logger.info("- General log: ./example_logs/production.log")
    logger.info("- Subsystem logs in ./example_logs/")
    logger.info("- Rotation: 10 MB")
    logger.info("- Retention: 1 week")
    logger.info("- Compression: zip")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print(" Corral Logging Framework Examples")
    print("=" * 70)

    examples = [
        example_basic_logging,
        example_file_logging,
        example_subsystem_logging,
        example_custom_handler,
        example_custom_format,
        example_error_filtering,
        example_production_setup,
    ]

    for example in examples:
        try:
            example()
            time.sleep(0.5)  # Brief pause between examples
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")

    print("\n" + "=" * 70)
    print(" All examples completed!")
    print("=" * 70)
    print("\nLog files created in ./example_logs/")
    print("You can explore them to see how different configurations work.")


if __name__ == "__main__":
    main()
