"""Dynamic metrics loader for loading custom metrics from Python files.

This module provides utilities to load custom metric classes from Python files,
enabling users to define their own metrics and use them with the CLI or Docker runner.

Example metrics file (my_metrics.py):
    ```python
    from corral.report.metrics import Metric, MetricMetadata


    class MyCustomMetric(Metric):
        @property
        def metadata(self):
            return MetricMetadata(
                name="my_custom_metric",
                display_name="My Custom Metric",
                description="A custom metric example",
            )

        def calculate(self, context):
            return len(context.all_task_ids) * 100


    # Export metrics to be loaded
    METRICS = [MyCustomMetric()]
    ```

Usage:
    ```python
    from corral.report.metrics.loader import load_metrics_from_file

    metrics = load_metrics_from_file("path/to/my_metrics.py")
    runner = CorralRunner(interface, agent, metrics=metrics)
    ```
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from corral.report.metrics.base import Metric


def load_metrics_from_file(
    file_path: str | Path,
    metrics_var: str = "METRICS",
    auto_discover: bool = True,
) -> list[Metric]:
    """Load metrics from a Python file.

    The file can export metrics in two ways:
    1. Define a `METRICS` list (or custom variable name) containing metric instances
    2. If auto_discover is True, automatically find all Metric subclass instances

    Args:
        file_path: Path to the Python file containing metric definitions.
        metrics_var: Name of the variable containing the list of metrics.
                    Defaults to "METRICS".
        auto_discover: If True and metrics_var is not found, automatically
                      discover Metric instances in the module. Defaults to True.

    Returns:
        List of Metric instances loaded from the file.

    Raises:
        FileNotFoundError: If the metrics file doesn't exist.
        ImportError: If the file cannot be imported.
        ValueError: If no metrics are found in the file.

    Example:
        >>> metrics = load_metrics_from_file("custom_metrics.py")
        >>> print(f"Loaded {len(metrics)} metrics")
    """
    from corral.report.metrics.base import Metric

    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    if not file_path.suffix == ".py":
        raise ValueError(f"Metrics file must be a Python file (.py): {file_path}")

    logger.info(f"Loading metrics from: {file_path}")

    # Create a unique module name to avoid conflicts
    module_name = f"_corral_custom_metrics_{file_path.stem}_{id(file_path)}"

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # Clean up on failure
        sys.modules.pop(module_name, None)
        raise ImportError(f"Error executing metrics file {file_path}: {e}") from e

    metrics: list[Metric] = []

    # Try to get metrics from the specified variable
    if hasattr(module, metrics_var):
        metrics_list = getattr(module, metrics_var)
        if not isinstance(metrics_list, list | tuple):
            raise ValueError(
                f"'{metrics_var}' in {file_path} must be a list or tuple, "
                f"got {type(metrics_list).__name__}"
            )
        for item in metrics_list:
            if isinstance(item, Metric):
                metrics.append(item)
            else:
                logger.warning(
                    f"Skipping non-Metric item in {metrics_var}: {type(item).__name__}"
                )

    # Auto-discover metrics if enabled and none found via variable
    if not metrics and auto_discover:
        logger.debug(f"'{metrics_var}' not found or empty, auto-discovering metrics...")
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)
            if isinstance(attr, Metric):
                metrics.append(attr)
                logger.debug(f"Auto-discovered metric: {attr.metadata.name}")

    if not metrics:
        raise ValueError(
            f"No metrics found in {file_path}. "
            f"Define a '{metrics_var}' list or instantiate Metric subclasses."
        )

    logger.info(
        f"Loaded {len(metrics)} metrics: "
        f"{', '.join(m.metadata.name for m in metrics)}"
    )

    return metrics


def load_metrics_from_module(
    module_path: str,
    metrics_var: str = "METRICS",
) -> list[Metric]:
    """Load metrics from an installed Python module.

    This allows loading metrics from installed packages rather than file paths.

    Args:
        module_path: Dot-separated module path (e.g., "mypackage.metrics").
        metrics_var: Name of the variable containing the list of metrics.

    Returns:
        List of Metric instances from the module.

    Raises:
        ImportError: If the module cannot be imported.
        ValueError: If no metrics are found.

    Example:
        >>> metrics = load_metrics_from_module("mypackage.custom_metrics")
    """
    from corral.report.metrics.base import Metric

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}': {e}") from e

    if not hasattr(module, metrics_var):
        raise ValueError(f"Module '{module_path}' has no '{metrics_var}' attribute")

    metrics_list = getattr(module, metrics_var)
    if not isinstance(metrics_list, list | tuple):
        raise ValueError(f"'{metrics_var}' in {module_path} must be a list or tuple")

    metrics = [m for m in metrics_list if isinstance(m, Metric)]

    if not metrics:
        raise ValueError(
            f"No valid Metric instances found in '{module_path}.{metrics_var}'"
        )

    logger.info(
        f"Loaded {len(metrics)} metrics from {module_path}: "
        f"{', '.join(m.metadata.name for m in metrics)}"
    )

    return metrics


def validate_metrics_file(file_path: str | Path) -> dict[str, any]:
    """Validate a metrics file without fully loading it.

    Useful for pre-flight checks before running a benchmark.

    Args:
        file_path: Path to the metrics file to validate.

    Returns:
        Dictionary with validation results:
        - valid: bool indicating if the file is valid
        - metrics_count: number of metrics found
        - metric_names: list of metric names
        - errors: list of error messages (empty if valid)

    Example:
        >>> result = validate_metrics_file("my_metrics.py")
        >>> if result["valid"]:
        ...     print(f"Found {result['metrics_count']} metrics")
    """
    result = {
        "valid": False,
        "metrics_count": 0,
        "metric_names": [],
        "errors": [],
    }

    try:
        metrics = load_metrics_from_file(file_path)
        result["valid"] = True
        result["metrics_count"] = len(metrics)
        result["metric_names"] = [m.metadata.name for m in metrics]
    except FileNotFoundError as e:
        result["errors"].append(f"File not found: {e}")
    except ImportError as e:
        result["errors"].append(f"Import error: {e}")
    except ValueError as e:
        result["errors"].append(f"Validation error: {e}")
    except Exception as e:
        result["errors"].append(f"Unexpected error: {e}")

    return result
