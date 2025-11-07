# Logging Framework

Corral uses [loguru](https://github.com/Delgan/loguru) as its logging framework, providing a simple and powerful logging system.

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

**Important**: By default, logs are NOT rotated or deleted to preserve benchmark data.

Log to a file without rotation (recommended for benchmarks):

```python
from corral import setup_logging

# Benchmark logging - no rotation (default)
setup_logging(
    level="INFO",
    log_file="benchmark.log"
)
```

Log to a file with rotation (for server logs only):

```python
from corral import setup_logging

# Server logs with rotation
setup_logging(
    level="INFO",
    log_file="server.log",
    rotation="100 MB",      # Rotate when file reaches 100 MB
    retention="30 days",    # Keep rotated logs for 30 days
    compression="gz"        # Compress rotated logs
)
```

### Custom Format

Customize the log message format:

```python
from corral import setup_logging

setup_logging(
    format_string=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{message}"
    )
)
```

## Integration with Corral

### In Your Benchmark Code

**Important**: Do not use rotation or retention for benchmark logs.

```python
from corral import CorralRunner, CorralRouter, setup_logging
from corral.agents import ReActAgent

# Setup logging before running benchmarks
# NOTE: No rotation/retention - benchmark logs should be kept permanently
setup_logging(
    level="INFO",
    log_file="benchmark.log"
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

## Advanced Usage

### Dynamic Handler Addition

Add handlers on-the-fly:

```python
from corral import add_file_handler

# Add a debug handler
handler_id = add_file_handler(
    "debug.log",
    level="DEBUG"
)

# ... do some work ...

# Remove the handler when done
from corral.logging_config import remove_handler
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

## Common Patterns

### Benchmark Setup

For benchmark logs, do NOT use rotation or retention:

```python
from corral import setup_logging

# Benchmark logging - NO rotation/retention
setup_logging(
    level="INFO",
    console=True,
    log_file="benchmark.log"
    # NOTE: No rotation, retention, or compression
)
```

### Development Setup

For development with verbose output:

```python
from corral import setup_logging

setup_logging(
    level="DEBUG",
    console=True,
    log_file="dev.log"
)
```

### Server Setup (Non-Benchmark)

For production server/service logs (NOT benchmarks), you can use rotation:

```python
from corral import setup_logging

setup_logging(
    level="INFO",
    console=True,
    log_file="server.log",
    rotation="100 MB",
    retention="30 days",
    compression="gz"
)
```

## API Reference

### `setup_logging()`

Main function to configure the logging system.

**Parameters:**
- `level` (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: "INFO"
- `console` (bool): Enable console output. Default: True
- `log_file` (str | None): Path to log file. Default: None
- `format_string` (str | None): Custom log format. Default: None
- `rotation` (str | None): When to rotate log files. Default: None (no rotation)
- `retention` (str | None): How long to keep rotated logs. Default: None (keep forever)
- `compression` (str | None): Compression format for rotated logs. Default: None

### `add_file_handler()`

Add a custom file handler to the logger.

**Parameters:**
- `filepath` (str): Path to log file
- `level` (str): Logging level. Default: "INFO"
- `format_string` (str | None): Custom format. Default: None
- `rotation` (str | None): Rotation policy. Default: None
- `retention` (str | None): Retention policy. Default: None
- `compression` (str | None): Compression format. Default: None
- `filter_func` (callable | None): Optional filter function. Default: None

**Returns:**
- Handler ID (int)

## Best Practices

### For Benchmarks
- ✅ **DO** use simple file logging without rotation
- ✅ **DO** keep logs permanently to preserve benchmark data
- ❌ **DON'T** use rotation, retention, or compression

### For Server Logs
- ✅ **DO** use rotation to manage disk space
- ✅ **DO** set appropriate retention policies
- ✅ **DO** compress old logs to save space

## Troubleshooting

### No Logs Appearing

If you're not seeing logs:
1. Ensure you've called `setup_logging()` before using the logger
2. Check the log level is appropriate for the messages you're logging
3. Verify console output is enabled if you expect to see logs in terminal

### Too Much Output

If you're getting too many logs:
1. Increase the log level: `setup_logging(level="WARNING")`
2. Disable console output: `setup_logging(console=False)`

## Additional Resources

- [Loguru Documentation](https://loguru.readthedocs.io/)
- [Python Logging Best Practices](https://docs.python-guide.org/writing/logging/)
