"""Reproduce the Stargazer mini-v2 selection from the recorded trial indexes."""

# ruff: noqa: T201

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from corral_mini_common import rank_cost_tie_counts, report_strata, row_spearman, select

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "src/corral/subsets/corral_mini_v2.json").read_text())
    refresh = manifest["stargazer_refresh"]
    sources = {}
    for name in ("full", "additions"):
        key = f"{name}_trials_csv"
        path = ROOT / refresh[key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != refresh[f"{key}_sha256"]:
            raise ValueError(f"{path} differs from the recorded trial index")
        sources[name] = pd.read_csv(path)

    bank_dir = ROOT / "tasks/stargazer/environments"
    bank_tasks = {
        level: set(
            json.loads(
                (bank_dir / f"level_{level}/tasks_json/task_bank.json").read_text()
            )[0]["task_ids"]
        )
        for level in (1, 2)
    }
    full = sources["full"]
    agents = sorted((full.agent + "__" + full.model).unique())
    mixed = refresh["mixed_column"]
    full_column = f"{mixed['agent']}__{mixed['full_model']}"
    additions_column = f"{mixed['agent']}__{mixed['additions_model']}"
    additions = sources["additions"]
    added_tasks = bank_tasks[1] - set(full.loc[full.level == 1, "task_id"])
    additions = additions[
        additions.level.eq(1)
        & additions.task_id.isin(added_tasks)
        & (additions.agent + "__" + additions.model).isin(
            (set(agents) - {full_column}) | {additions_column}
        )
    ]
    trials = pd.concat(
        [
            full[full.level.eq(level) & full.task_id.isin(ids)]
            for level, ids in bank_tasks.items()
        ]
        + [additions],
        ignore_index=True,
    )
    if trials.duplicated(["level", "task_id", "agent", "model", "trial"]).any():
        raise ValueError("Duplicate Stargazer trials")
    strata = report_strata(
        trials,
        ROOT / "corral_mini/prices.json",
        agents,
        manifest["score_metric"],
        "stargazer",
        agent_columns={additions_column: full_column},
    )
    indices = np.arange(len(agents))
    for level, bank in bank_tasks.items():
        stratum = strata["stargazer", level]
        if set(stratum.tasks) != bank:
            raise ValueError(
                f"Stargazer Level {level} reports differ from the task bank"
            )
        k = manifest["tasks_per_level"]
        exclude_saturated = manifest["exclude_saturated"]
        selected = select(
            stratum,
            k,
            manifest["method"],
            indices,
            exclude_saturated=exclude_saturated,
            seed=manifest["seed"],
        )
        ids = stratum.tasks[selected].tolist()
        expected = manifest["tasks_by_environment"]["stargazer"][f"level_{level}"]
        if ids != expected:
            raise ValueError(
                f"Stargazer Level {level} selected {ids}, expected {expected}"
            )
        full_scores = stratum.passes.mean(axis=0)
        mini_scores = stratum.passes[selected].mean(axis=0)
        rho = row_spearman(mini_scores[None], full_scores)[0]
        cost = stratum.costs[selected].sum()
        counts = rank_cost_tie_counts(stratum, k, indices, exclude_saturated)
        print(f"Level {level} {' '.join(ids)}")
        print(f"  Spearman {rho:.6f}, estimated cost ${cost:.2f}, {counts}")


if __name__ == "__main__":
    main()
