"""Statistical tests for intervention experiment results.

For each (env, agent, intervention, num_steps) condition, tests whether
the success rate is significantly different from the baseline using
a two-proportion z-test with Bonferroni correction.

Reads from analysis/results.csv (output of aggregate_results.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INTERVENTION_ROOT


def two_proportion_ztest(
    successes1: int, n1: int, successes2: int, n2: int
) -> tuple[float, float]:
    """Two-proportion z-test.

    Tests H0: p1 == p2 vs. H1: p1 != p2.
    Returns (z_stat, p_value).
    """
    p1 = successes1 / n1 if n1 > 0 else 0
    p2 = successes2 / n2 if n2 > 0 else 0

    # Pooled proportion
    p_pool = (successes1 + successes2) / (n1 + n2)

    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0

    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run statistical tests for all conditions vs. baseline."""
    results = []

    groups = df.groupby(["env", "agent"])
    for (env, agent), group_df in groups:
        # Get baseline
        baseline = group_df[group_df["intervention"] == "none"]
        if baseline.empty:
            continue

        baseline_n = len(baseline)
        baseline_successes = baseline["success"].sum()
        baseline_sr = baseline_successes / baseline_n

        # Test each intervention condition
        interventions = group_df[group_df["intervention"] != "none"]
        conditions = interventions.groupby(["intervention", "num_steps"])

        for (intervention, num_steps), cond_df in conditions:
            cond_n = len(cond_df)
            cond_successes = cond_df["success"].sum()
            cond_sr = cond_successes / cond_n

            z_stat, p_value = two_proportion_ztest(
                int(cond_successes), cond_n, int(baseline_successes), baseline_n
            )

            results.append(
                {
                    "env": env,
                    "agent": agent,
                    "intervention": intervention,
                    "num_steps": num_steps,
                    "baseline_sr": round(baseline_sr, 3),
                    "condition_sr": round(cond_sr, 3),
                    "delta": round(cond_sr - baseline_sr, 3),
                    "baseline_n": baseline_n,
                    "condition_n": cond_n,
                    "z_stat": round(z_stat, 3),
                    "p_value": p_value,
                }
            )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    # Bonferroni correction
    n_tests = len(results_df)
    results_df["p_bonferroni"] = np.minimum(results_df["p_value"] * n_tests, 1.0)
    results_df["significant_005"] = results_df["p_bonferroni"] < 0.05
    results_df["significant_001"] = results_df["p_bonferroni"] < 0.01

    return results_df.sort_values(["env", "agent", "intervention", "num_steps"])


def main():
    results_path = INTERVENTION_ROOT / "analysis" / "results.csv"
    if not results_path.exists():
        logger.info(f"ERROR: {results_path} not found. Run aggregate_results.py first.")
        sys.exit(1)

    results_df = pd.read_csv(results_path)
    test_results = run_tests(results_df)

    if test_results.empty:
        logger.info("No test results (missing baseline conditions?).")
        return

    # Save
    output_path = INTERVENTION_ROOT / "analysis" / "statistical_tests.csv"
    test_results.to_csv(output_path, index=False)
    logger.info(f"Saved {len(test_results)} test results to {output_path}")

    # Print summary
    logger.info("\n=== Statistical Tests Summary ===")
    logger.info(f"Total tests: {len(test_results)}")
    logger.info(
        f"Significant (p < 0.05, Bonferroni): {test_results['significant_005'].sum()}"
    )
    logger.info(
        f"Significant (p < 0.01, Bonferroni): {test_results['significant_001'].sum()}"
    )

    logger.info("\n=== Significant Results ===")
    sig = test_results[test_results["significant_005"]]
    if sig.empty:
        logger.info("No significant results.")
    else:
        cols = [
            "env",
            "agent",
            "intervention",
            "num_steps",
            "baseline_sr",
            "condition_sr",
            "delta",
            "p_bonferroni",
        ]
        logger.info(sig[cols].to_string(index=False))

    logger.info("\n=== Full Results ===")
    cols = [
        "env",
        "agent",
        "intervention",
        "num_steps",
        "baseline_sr",
        "condition_sr",
        "delta",
        "p_bonferroni",
        "significant_005",
    ]
    logger.info(test_results[cols].to_string(index=False))


if __name__ == "__main__":
    main()
