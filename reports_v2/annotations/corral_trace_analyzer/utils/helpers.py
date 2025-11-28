"""
Helper utilities
"""

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


def save_results(results: Any, filepath: str, format: str = "json") -> None:
    """
    Save results to file

    Args:
        results: Results to save (dict, DataFrame, etc.)
        filepath: Path to save to
        format: Format ('json', 'pickle', 'csv', 'parquet')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with filepath.open("w") as f:
            json.dump(results, f, indent=2, default=str)

    elif format == "pickle":
        with filepath.open("wb") as f:
            pickle.dump(results, f)

    elif format == "csv" and isinstance(results, pd.DataFrame):
        results.to_csv(filepath, index=False)

    elif format == "parquet" and isinstance(results, pd.DataFrame):
        results.to_parquet(filepath, index=False)

    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str, format: str = "json") -> Any:
    """
    Load results from file

    Args:
        filepath: Path to load from
        format: Format ('json', 'pickle', 'csv', 'parquet')

    Returns:
        Loaded results
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if format == "json":
        with filepath.open() as f:
            return json.load(f)

    elif format == "pickle":
        with filepath.open("rb") as f:
            return pickle.load(f)

    elif format == "csv":
        return pd.read_csv(filepath)

    elif format == "parquet":
        return pd.read_parquet(filepath)

    else:
        raise ValueError(f"Unsupported format: {format}")


def export_to_csv(
    dataframes: dict[str, pd.DataFrame], output_dir: str, prefix: str = ""
) -> None:
    """
    Export multiple dataframes to CSV files

    Args:
        dataframes: dictionary of {name: dataframe}
        output_dir: Output directory
        prefix: Optional prefix for filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in dataframes.items():
        filename = f"{prefix}{name}.csv" if prefix else f"{name}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {name} to {filepath}")


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Create a summary of a dataframe

    Args:
        df: DataFrame to summarize

    Returns:
        dictionary with summary statistics
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict()
        if len(df.select_dtypes(include="number").columns) > 0
        else {},
    }

    return summary


def filter_significant_correlations(
    correlation_df: pd.DataFrame, min_abs_corr: float = 0.3, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Filter correlation results to significant ones

    Args:
        correlation_df: DataFrame with correlation results
        min_abs_corr: Minimum absolute correlation
        alpha: Significance level

    Returns:
        Filtered DataFrame
    """
    if (
        "abs_correlation" not in correlation_df.columns
        or "p_value" not in correlation_df.columns
    ):
        return correlation_df

    return correlation_df[
        (correlation_df["abs_correlation"] >= min_abs_corr)
        & (correlation_df["p_value"] < alpha)
    ]


def create_analysis_report(
    correlation_results: dict[str, Any],
    interaction_results: dict[str, Any],
    output_path: str,
) -> None:
    """
    Create a text report of analysis results

    Args:
        correlation_results: Results from correlation analysis
        interaction_results: Results from interaction analysis
        output_path: Path to save report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("CORRAL TRACE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Correlation results
        f.write("CORRELATION ANALYSIS\n")
        f.write("-" * 80 + "\n\n")

        if "tier1" in correlation_results:
            f.write("Tier 1 Analyses (High Priority):\n\n")

            for analysis_name, results in correlation_results["tier1"].items():
                if isinstance(results, pd.DataFrame) and len(results) > 0:
                    f.write(f"\n{analysis_name}:\n")
                    f.write(results.head(5).to_string() + "\n\n")

        if "tier2" in correlation_results:
            f.write("\nTier 2 Analyses:\n\n")

            for analysis_name, results in correlation_results["tier2"].items():
                if isinstance(results, pd.DataFrame) and len(results) > 0:
                    f.write(f"\n{analysis_name}:\n")
                    f.write(results.head(5).to_string() + "\n\n")

        # Interaction results
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERACTION ANALYSIS\n")
        f.write("-" * 80 + "\n\n")

        if "main_effects" in interaction_results:
            f.write("Main Effects:\n\n")

            for factor, results in interaction_results["main_effects"].items():
                if "f_statistic" in results:
                    f.write(f"\n{factor}:\n")
                    f.write(f"  F-statistic: {results['f_statistic']:.4f}\n")
                    f.write(f"  p-value: {results['p_value']:.4e}\n")
                    f.write(f"  Significant: {results['significant']}\n")
                    f.write(f"  Effect size (eta²): {results['eta_squared']:.4f}\n\n")

        if "two_way_interactions" in interaction_results:
            f.write("\nTwo-Way Interactions:\n\n")

            for interaction_name, results in interaction_results[
                "two_way_interactions"
            ].items():
                if "factor1" in results:
                    f.write(f"\n{interaction_name}:\n")
                    f.write(
                        f"  Factor 1 ({results['factor1']}) p-value: {results.get('factor1_p_value', 'N/A')}\n"
                    )
                    f.write(
                        f"  Factor 2 ({results['factor2']}) p-value: {results.get('factor2_p_value', 'N/A')}\n"
                    )
                    f.write(
                        f"  Interaction detected: {results.get('interaction_detected', False)}\n\n"
                    )

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Report saved to {output_path}")
