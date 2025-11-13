"""Scoring functions for kinetic fitting tasks."""

import json
from pathlib import Path
from typing import Dict, Any


def calculate_score(work_dir: str) -> float:
    """Calculate score based on phenomenological trends.

    Args:
        work_dir: Working directory containing fit results

    Returns:
        Score between 0 and 1 based on trend reproduction
    """
    results_path = Path(work_dir) / "fit_results.json"

    if not results_path.exists():
        return 0.0

    try:
        with open(results_path) as f:
            results = json.load(f)

        # Get the best score from the results
        return results.get("best_phenomenological_score", 0.0)

    except Exception:
        return 0.0
