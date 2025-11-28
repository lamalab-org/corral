"""
Data validation utilities
"""

from typing import Any, dict, list

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> dict[str, Any]:
    """
    Validate that a dataframe has required columns

    Args:
        df: DataFrame to validate
        required_columns: list of required column names

    Returns:
        dictionary with validation results
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    return {
        "is_valid": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "has_data": len(df) > 0,
        "row_count": len(df),
        "column_count": len(df.columns),
    }


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Check if dataframe has required columns, raise error if not

    Args:
        df: DataFrame to check
        required_columns: list of required column names

    Raises:
        ValueError: If required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def check_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Check data quality metrics

    Args:
        df: DataFrame to check

    Returns:
        dictionary with quality metrics
    """
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
    }


def validate_trace_data(
    traces_df: pd.DataFrame, steps_df: pd.DataFrame, tools_df: pd.DataFrame
) -> dict[str, Any]:
    """
    Validate trace data consistency

    Args:
        traces_df: Trace-level dataframe
        steps_df: Step-level dataframe
        tools_df: Tool-level dataframe

    Returns:
        dictionary with validation results
    """
    validation = {
        "traces_valid": True,
        "steps_valid": True,
        "tools_valid": True,
        "issues": [],
    }

    # Check trace IDs
    trace_ids_traces = set(traces_df["trace_id"].unique())
    trace_ids_steps = set(steps_df["trace_id"].unique())
    trace_ids_tools = set(tools_df["trace_id"].unique())

    # Check for orphaned steps
    orphaned_steps = trace_ids_steps - trace_ids_traces
    if orphaned_steps:
        validation["steps_valid"] = False
        validation["issues"].append(
            f"Steps with no matching trace: {len(orphaned_steps)}"
        )

    # Check for orphaned tools
    orphaned_tools = trace_ids_tools - trace_ids_traces
    if orphaned_tools:
        validation["tools_valid"] = False
        validation["issues"].append(
            f"Tools with no matching trace: {len(orphaned_tools)}"
        )

    # Check for traces with no steps
    traces_no_steps = trace_ids_traces - trace_ids_steps
    if traces_no_steps:
        validation["issues"].append(f"Traces with no steps: {len(traces_no_steps)}")

    # Check for traces with no tools
    traces_no_tools = trace_ids_traces - trace_ids_tools
    if traces_no_tools:
        validation["issues"].append(f"Traces with no tools: {len(traces_no_tools)}")

    validation["overall_valid"] = (
        validation["traces_valid"]
        and validation["steps_valid"]
        and validation["tools_valid"]
    )

    return validation
