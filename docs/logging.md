# Logging Framework

Corral uses [loguru](https://github.com/Delgan/loguru) as its logging framework, providing a flexible and powerful logging system that can be easily configured for different use cases.

## Quick Start

### Basic Usage

```python
from corral import setup_logging

# Initialize logging with default settings (console output at INFO level)
setup_logging()
```

### Using the Logger

After setup, you can use loguru's logger directly in any module:

```python
from loguru import logger

logger.info("This is an info message")
logger.debug("This is a debug message")
logger.warning("This is a warning")
logger.error("This is an error")
```

## Configuration Options

### Console Logging

Enable or disable console output:

```python
from corral import setup_logging

# With console output (default)
setup_logging(console=True, level="INFO")

# Without console output
setup_logging(console=False, level="INFO")
```

### File Logging

Log to a single file:

```python
from corral import setup_logging

setup_logging(
    level="INFO",
    log_file="corral.log",
    rotation="10 MB",      # Rotate when file reaches 10 MB
    retention="1 week",    # Keep rotated logs for 1 week
    compression="zip"      # Compress rotated logs
)
```

### Subsystem-Specific Logging

Corral has several subsystems that can log to separate files for better organization:

- `agents`: Agent execution and decision-making
- `backend`: Task environment and server operations
- `router`: API routing and interface management
- `utils`: Utility functions and tools
- `report`: Reporting and result logging

Configure subsystem-specific log files:

```python
from corral import setup_logging

setup_logging(
    level="INFO",
    console=True,
    log_dir="./logs",
    subsystem_files={
        "agents": "agents.log",
        "backend": "backend.log",
        "router": "router.log",
        "utils": "utils.log",
        "report": "report.log",
    }
)
```

This configuration will:
1. Output all logs to console
2. Filter and save agent-related logs to `./logs/agents.log`
3. Filter and save backend-related logs to `./logs/backend.log`
4. And so on for other subsystems

### Custom Format

Customize the log message format:

```python
from corral import setup_logging

setup_logging(
    format_string=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
)
```

### Combined Configuration

A complete example with all options:

```python
from corral import setup_logging

setup_logging(
    level="DEBUG",
    console=True,
    log_file="corral_full.log",
    log_dir="./logs",
    subsystem_files={
        "agents": "agents.log",
        "backend": "backend.log",
        "router": "router.log",
        "utils": "utils.log",
        "report": "report.log",
    },
    format_string=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    rotation="50 MB",
    retention="2 weeks",
    compression="gz"
)
```

## Advanced Usage

### Dynamic Handler Addition

Add handlers on-the-fly:

```python
from corral import add_file_handler, remove_handler

# Add a debug handler for temporary detailed logging
handler_id = add_file_handler(
    "debug.log",
    level="DEBUG",
    rotation="1 MB"
)

# ... do some work ...

# Remove the handler when done
remove_handler(handler_id)
```

### Custom Filters

Create handlers with custom filters:

```python
from corral import add_file_handler

def error_only_filter(record):
    """Only log ERROR and CRITICAL messages"""
    return record["level"].name in ["ERROR", "CRITICAL"]

handler_id = add_file_handler(
    "errors_only.log",
    level="DEBUG",  # Set low level, filter handles the rest
    filter_func=error_only_filter
)
```

### Changing Log Level at Runtime

```python
from corral import set_level

# Start with INFO level
setup_logging(level="INFO")

# ... later in your code ...

# Switch to DEBUG for detailed output
set_level("DEBUG")
```

### Using Subsystem-Specific Loggers

Get a logger bound to a specific subsystem context:

```python
from corral import get_logger

# Get a logger for the agents subsystem
agents_logger = get_logger("agents")
agents_logger.info("Agent initialized")

# Get a logger for the backend subsystem
backend_logger = get_logger("backend")
backend_logger.info("Server starting")
```

## Integration with Corral

### In Your Benchmark Code

**Important**: Do not use rotation or retention for benchmark logs. Benchmark data should be preserved.

```python
from corral import CorralRunner, CorralRouter, setup_logging
from corral.agents import ReActAgent

# Setup logging before running benchmarks
# NOTE: No rotation/retention - benchmark logs should be kept permanently
setup_logging(
    level="INFO",
    log_dir="./benchmark_logs",
    subsystem_files={
        "agents": "agents.log",
        "router": "router.log",
    }
)

# Run your benchmark
interface = CorralRouter("http://localhost:8000")
agent = ReActAgent(model="gpt-4o", max_iterations=10)
runner = CorralRunner(interface, agent)
result = runner.bench()
```

### In Custom Agents

```python
from loguru import logger
from corral.agents import BaseAgent

class MyCustomAgent(BaseAgent):
    def run_agent(self, interface, task_id, **kwargs):
        logger.info(f"Starting custom agent for task {task_id}")
        # ... your agent logic ...
        logger.debug("Processing step 1")
        logger.debug("Processing step 2")
        logger.info("Agent completed successfully")
```

## Log Rotation and Management

Loguru supports several rotation options:

- **Size-based**: `"10 MB"`, `"1 GB"`
- **Time-based**: `"12:00"` (daily at noon), `"1 week"`, `"1 day"`
- **Function-based**: Pass a custom function

Retention options:

- **Time-based**: `"1 week"`, `"30 days"`
- **Count-based**: `3` (keep 3 rotated files)
- **Function-based**: Pass a custom function

## Common Patterns

### Benchmark Setup

**Important**: For benchmark logs, do NOT use rotation or retention. Benchmark results 
are valuable data that should be preserved permanently, not rotated away.

```python
from corral import setup_logging

# Benchmark logging - NO rotation/retention
setup_logging(
    level="INFO",
    console=True,
    log_dir="./benchmark_logs",
    subsystem_files={
        "agents": "agents.log",
        "backend": "backend.log",
        "router": "router.log",
        "utils": "utils.log",
        "report": "report.log",
    }
    # NOTE: No rotation, retention, or compression for benchmark logs
)
```

### Production Server Setup

For production server/service deployments with detailed logging (non-benchmark):

```python
from corral import setup_logging

setup_logging(
    level="INFO",
    console=True,
    log_file="corral_production.log",
    log_dir="./logs",
    subsystem_files={
        "agents": "agents.log",
        "backend": "backend.log",
        "router": "router.log",
        "utils": "utils.log",
        "report": "report.log",
    },
    rotation="100 MB",
    retention="30 days",
    compression="gz"
)
```

### Development Setup

For development with verbose output:

```python
from corral import setup_logging

setup_logging(
    level="DEBUG",
    console=True,
    log_file="corral_dev.log",
    rotation="10 MB",
    retention="3 days"
)
```

### Testing Setup

For testing with minimal output:

```python
from corral import setup_logging

setup_logging(
    level="WARNING",
    console=True
)
```

## Troubleshooting

### No Logs Appearing

If you're not seeing logs, ensure:
1. You've called `setup_logging()` before using the logger
2. The log level is appropriate for the messages you're logging
3. Console output is enabled if you expect to see logs in terminal

### Too Much Output

If you're getting too many logs:
1. Increase the log level: `setup_logging(level="WARNING")`
2. Disable console output: `setup_logging(console=False)`
3. Use subsystem-specific files to separate concerns

### Log Files Not Created

If log files aren't being created:
1. Check that the directory exists or can be created
2. Verify you have write permissions
3. Check the file paths in your configuration

## API Reference

### `setup_logging()`

Main function to configure the logging system.

**Parameters:**
- `level` (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `console` (bool): Enable console output
- `log_file` (str | None): Path to general log file
- `log_dir` (str | None): Directory for subsystem log files
- `subsystem_files` (dict[str, str] | None): Mapping of subsystem names to log files
- `format_string` (str | None): Custom log format
- `rotation` (str): When to rotate log files
- `retention` (str): How long to keep rotated logs
- `compression` (str): Compression format for rotated logs

### `get_logger(subsystem)`

Get a logger instance, optionally bound to a subsystem.

**Parameters:**
- `subsystem` (str | None): Name of the subsystem

**Returns:**
- Logger instance

### `add_file_handler()`

Add a custom file handler to the logger.

**Parameters:**
- `filepath` (str): Path to log file
- `level` (str): Logging level
- `format_string` (str | None): Custom format
- `rotation` (str): Rotation policy
- `retention` (str): Retention policy
- `compression` (str): Compression format
- `filter_func` (callable | None): Optional filter function

**Returns:**
- Handler ID (int)

### `remove_handler(handler_id)`

Remove a handler by its ID.

**Parameters:**
- `handler_id` (int): Handler ID from add_file_handler()

### `set_level(level)`

Change the logging level for all handlers.

**Parameters:**
- `level` (str): New logging level

## Additional Resources

- [Loguru Documentation](https://loguru.readthedocs.io/)
- [Python Logging Best Practices](https://docs.python-guide.org/writing/logging/)
