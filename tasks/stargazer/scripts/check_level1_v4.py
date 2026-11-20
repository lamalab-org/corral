"""Reproduce the blind recovery checks for the four Level 1 v4 replacements."""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
from stargazer.identifiability import BankRules, fit_counts
from stargazer.models import load_task, simulate_keplerian_rv
from stargazer.score import evaluate_submission

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
TASKS = ("seed111_diff3", "seed630238_diff4", "seed610696_diff5", "seed611761_diff5")


def check_trial(job):
    task_id, index = job
    task = load_task(DATA / f"{task_id}.json", source="synthetic")
    if index:
        # Keep the observation times, quoted errors and orbit fixed across draws.
        obs = task.observations
        noise = task.config["noise"]
        assert not noise["gp"]["use_gp"]
        seed = task.metadata["seed"]
        rng = np.random.default_rng(np.random.SeedSequence([26112073, seed, index]))
        sigma = np.hypot(obs.sigmas_ms, noise["sigma_jitter_ms"])
        signal = simulate_keplerian_rv(
            task.truth_planets, np.asarray(obs.times_days), task.star_mass_sun
        )
        offsets = {
            instrument.label: instrument.gamma_ms for instrument in task.instruments
        }
        signal += task.config["star"].get("gamma_ms", 0.0) + np.array(
            [offsets[label] for label in obs.instruments]
        )
        task = replace(
            task, observations=replace(obs, rvs_ms=tuple(signal + rng.normal(0, sigma)))
        )

    rules = BankRules()
    fits = fit_counts(
        task.public_fit_context(),
        starts=rules.starts,
        max_planets=rules.max_planets,
        max_nfev=rules.max_nfev,
        rng=np.random.default_rng(42 + index),
    )
    best = min(fits, key=lambda fit: fit["bic"])
    result = evaluate_submission(task, best["candidate"], rules.criteria())
    return {
        "task_id": task_id,
        "trial": index,
        "passed": int(result.success and best["optimizer"]["converged"]),
        "count": best["count"],
        "match": result.match_score,
        "rms_ms": result.rms_ms,
        "delta_bic_per_point": result.delta_bic_per_point,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--draws", type=int, choices=range(101), metavar="0..100", default=100
    )
    args = parser.parse_args()
    indices = [0, *range(101, 101 + args.draws)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = pool.map(check_trial, product(TASKS, indices))
        first = next(rows)
        writer = csv.DictWriter(sys.stdout, fieldnames=first, lineterminator="\n")
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)
