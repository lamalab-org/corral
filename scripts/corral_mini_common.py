"""Shared loading, task selection and scoring for corral-mini.

Terms used throughout:
- stratum: one (environment, level), scored with Pass@5 per task and agent.
- agent: one agent/model combination ("subject" in the report tables).
- material loss: a pair of agents at least MATERIAL_GAP apart on the full
  stratum that ties or flips order on the mini subset.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = "jablonkagroup/corral-reports-october-runs-summary-v1"
TASKS_FILE = "tasks/train-00000-of-00001.parquet"
EXCLUDED_ENVIRONMENTS = ("afm",)
METHODS = ("random", "profile", "rank", "rank_guarded", "cost_aware", "rank_cost")
SCORE_GUARD = 0.02
MATERIAL_GAP = 0.05


@dataclass
class Stratum:
    tasks: np.ndarray  # task IDs
    passes: np.ndarray  # Pass@5, tasks x agents
    costs: np.ndarray  # estimated USD for five trials, tasks x agents
    solved: np.ndarray  # solved in at least one of five trials, tasks x agents


def load(
    revision: str,
    tasks_parquet: Path | None,
    prices_path: Path,
    score_column: str = "pass_at_5",
):
    """Return strata keyed by (environment, level), agent names and source info."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision must be a full 40-character commit hash")
    if tasks_parquet is None:
        from huggingface_hub import hf_hub_download

        tasks_parquet = Path(
            hf_hub_download(
                REPO, filename=TASKS_FILE, repo_type="dataset", revision=revision
            )
        )
    data = pd.read_parquet(tasks_parquet)
    if score_column not in {"pass_at_5", "pass_hat_5"}:
        raise ValueError(f"unknown score column: {score_column}")
    data = data[~data.environment.isin(EXCLUDED_ENVIRONMENTS)].copy()
    if (data.trials != 5).any():
        raise ValueError("expected five trials per task and agent")
    data["agent_model"] = data.agent + "__" + data.model
    if data.duplicated(["environment", "level", "task_id", "agent_model"]).any():
        raise ValueError("duplicate task rows for an agent")

    prices = json.loads(prices_path.read_text())
    if missing := set(data.model) - set(prices):
        raise ValueError(f"missing model prices: {sorted(missing)}")
    data["usd"] = (
        data.total_input_tokens * data.model.map(lambda m: prices[m]["input"])
        + data.total_output_tokens * data.model.map(lambda m: prices[m]["output"])
    ) / 1_000_000

    agents = sorted(data.agent_model.unique())
    strata = {}
    for (environment, level), group in data.groupby(["environment", "level"]):
        passes = group.pivot_table(score_column, "task_id", "agent_model")
        costs = group.pivot_table("usd", "task_id", "agent_model")
        solved = group.pivot_table("passes", "task_id", "agent_model") > 0
        if list(passes.columns) != agents or passes.isna().any().any():
            raise ValueError(f"{environment}/{level}: every agent must run every task")
        strata[(str(environment), int(level))] = Stratum(
            passes.index.to_numpy(),
            passes.to_numpy(float),
            costs.to_numpy(float),
            solved.to_numpy(bool),
        )
    source = {
        "repo": REPO,
        "revision": revision,
        "file": TASKS_FILE,
        "sha256": hashlib.sha256(Path(tasks_parquet).read_bytes()).hexdigest(),
    }
    return dict(sorted(strata.items())), agents, source


def load_afm_strata(
    trials_csv: Path,
    prices_path: Path,
    agents: list[str],
    score_column: str = "pass_hat_5",
) -> dict:
    """Load the newer AFM reports as five-trial Pass^5 or Pass@5 strata."""
    if score_column not in {"pass_hat_5", "pass_at_5"}:
        raise ValueError(f"unknown score column: {score_column}")
    data = pd.read_csv(trials_csv)
    data["agent_model"] = data.agent + "__" + data.model
    prices = json.loads(prices_path.read_text())
    data["usd"] = (
        data.input_tokens * data.model.map(lambda m: prices[m]["input"])
        + data.output_tokens * data.model.map(lambda m: prices[m]["output"])
    ) / 1_000_000
    grouped = data.groupby(["level", "task_id", "agent_model"], as_index=False).agg(
        passes=("success", "sum"), trials=("success", "size"), usd=("usd", "sum")
    )
    if (grouped.trials != 5).any():
        raise ValueError("AFM reports must have five trials per task and agent")
    grouped["pass_hat_5"] = (grouped.passes / 5) ** 5
    grouped["pass_at_5"] = 1 - (1 - grouped.passes / 5) ** 5
    strata = {}
    for level, group in grouped.groupby("level"):
        scores = group.pivot_table(score_column, "task_id", "agent_model")
        costs = group.pivot_table("usd", "task_id", "agent_model")
        solved = group.pivot_table("passes", "task_id", "agent_model") > 0
        if list(scores.columns) != agents or scores.isna().any().any():
            raise ValueError(f"AFM level {level}: missing agent data")
        strata[("afm", int(level))] = Stratum(
            scores.index.to_numpy(),
            scores.to_numpy(float),
            costs.to_numpy(float),
            solved.to_numpy(bool),
        )
    return strata


@cache
def combinations(n: int, k: int) -> np.ndarray:
    return np.asarray(list(itertools.combinations(range(n), k)), dtype=np.int32)


def row_spearman(rows: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row with target; constant vectors count as tied."""
    a = rankdata(rows, axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b = rankdata(target)
    b -= b.mean()
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b)
    if np.linalg.norm(b) == 0:
        return (np.linalg.norm(a, axis=1) == 0).astype(float)
    return np.divide(a @ b, norms, out=np.zeros(len(a)), where=norms > 0)


def select(
    stratum: Stratum,
    k: int,
    method: str,
    agents: np.ndarray,
    rng: np.random.Generator | None = None,
    exclude_saturated: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Return the task indexes chosen for one stratum, fitting only `agents`.

    `rng` draws the random method. The others enumerate every k-task
    combination and break exact ties with a permutation seeded by `seed`.

    With `exclude_saturated`, tasks every one of `agents` solves are skipped.
    A level with fewer than k other tasks takes all of them and fills the
    remaining slots from saturated tasks, chosen by the same method.
    """
    n = len(stratum.tasks)
    unsaturated = (
        ~stratum.solved[:, agents].all(axis=1)
        if exclude_saturated
        else np.ones(n, dtype=bool)
    )
    must_use = min(k, int(unsaturated.sum()))
    if method == "random":
        chosen = rng.choice(np.flatnonzero(unsaturated), size=must_use, replace=False)
        top_up = rng.choice(
            np.flatnonzero(~unsaturated), size=k - must_use, replace=False
        )
        return np.sort(np.concatenate([chosen, top_up]))
    combos = combinations(n, k)
    combos = combos[unsaturated[combos].sum(axis=1) == must_use]
    passes = stratum.passes[:, agents]
    target = passes.mean(axis=0)
    error = passes[combos].mean(axis=1) - target
    mse = np.mean(error**2, axis=1)
    worst = np.max(abs(error), axis=1)
    guarded = worst <= worst.min() + SCORE_GUARD + 1e-12
    cost = stratum.costs[:, agents].sum(axis=1)[combos].sum(axis=1)
    rho = row_spearman(target + error, target)
    # Rounded so float noise does not hide a tie. Last key sorts first.
    mse, worst, cost, rho = (np.round(x, 9) for x in (mse, worst, cost, rho))
    keys = {
        "profile": (mse,),
        "cost_aware": (mse, cost),
        "rank": (mse, worst, -rho),
        "rank_guarded": (mse, worst, -rho),
        "rank_cost": (mse, cost, -rho),
    }
    if method not in keys:
        raise ValueError(f"unknown method: {method}")
    tie_break = np.random.default_rng(seed).permutation(len(combos))
    order = np.lexsort((tie_break, *keys[method]))
    if method in {"cost_aware", "rank_guarded"}:
        order = order[guarded[order]]
    return combos[order[0]]


def rank_cost_tie_counts(
    stratum: Stratum, k: int, agents: np.ndarray, exclude_saturated: bool = True
) -> dict[str, int]:
    """Count eligible sets tied at each Rank + cost decision step.

    Matches select(): Spearman is primary, estimated cost is secondary, and
    mean squared score error is tertiary. Criteria are rounded to 9 decimals.
    """
    n = len(stratum.tasks)
    unsaturated = (
        ~stratum.solved[:, agents].all(axis=1)
        if exclude_saturated
        else np.ones(n, dtype=bool)
    )
    must_use = min(k, int(unsaturated.sum()))
    combos = combinations(n, k)
    combos = combos[unsaturated[combos].sum(axis=1) == must_use]
    passes = stratum.passes[:, agents]
    target = passes.mean(axis=0)
    mini = passes[combos].mean(axis=1)
    rho = np.round(row_spearman(mini, target), 9)
    best_rank = rho == rho.max()
    costs = np.round(stratum.costs[:, agents].sum(axis=1)[combos].sum(axis=1), 9)
    best_cost = best_rank & (costs == costs[best_rank].min())
    mse = np.round(np.mean((mini - target) ** 2, axis=1), 9)
    best_mse = best_cost & (mse == mse[best_cost].min())
    return {
        "eligible_sets": len(combos),
        "sets_tied_at_best_rank": int(best_rank.sum()),
        "sets_tied_after_cost": int(best_cost.sum()),
        "sets_tied_after_score_error": int(best_mse.sum()),
    }


def material_losses(full: np.ndarray, mini: np.ndarray, involving=None) -> int:
    """Count agent pairs that are clearly apart on full and tie or flip on mini."""
    count = 0
    for i, j in itertools.combinations(range(len(full)), 2):
        if involving is not None and i not in involving and j not in involving:
            continue
        gap = full[i] - full[j]
        count += abs(gap) >= MATERIAL_GAP and gap * (mini[i] - mini[j]) <= 0
    return int(count)


def evaluate(strata: dict, selection: dict) -> dict:
    """Score a selection against the full strata, using every agent.

    `selection` maps each (environment, level) to chosen task indexes. Spearman
    is averaged only over levels where agents' full scores differ.
    """
    level_rho, worst_error, levels_lost, saturated_used = [], 0.0, 0, 0
    full_overall, mini_overall, shifts = [], [], []
    for (environment, level), stratum in strata.items():
        full = stratum.passes.mean(axis=0)
        chosen = selection[(environment, level)]
        mini = stratum.passes[chosen].mean(axis=0)
        full_overall.append(full)
        mini_overall.append(mini)
        worst_error = max(worst_error, float(np.max(abs(mini - full))))
        shifts.extend(mini - full)
        levels_lost += material_losses(full, mini) > 0
        saturated_used += int(stratum.solved[chosen].all(axis=1).sum())
        if np.std(full) > 0:  # a level every agent scores equally carries no ranking
            level_rho.append(float(row_spearman(mini[None], full)[0]))
    return {
        "mean_level_spearman": float(np.mean(level_rho)),
        "level_spearman": level_rho,
        "levels_with_loss": levels_lost,
        "saturated_tasks": saturated_used,
        "worst_level_error": worst_error,
        # Signed mini - full Pass@5 over every agent and level; < 0 means harder.
        "mean_score_shift": float(np.mean(shifts)),
        "overall_spearman": float(
            row_spearman(
                np.mean(mini_overall, axis=0)[None], np.mean(full_overall, axis=0)
            )[0]
        ),
        "usd": float(
            sum(stratum.costs[selection[key]].sum() for key, stratum in strata.items())
        ),
    }


def held_out_model_losses(
    strata: dict,
    agents: list[str],
    k: int,
    method: str,
    selection=None,
    exclude_saturated: bool = False,
) -> int:
    """Levels where a model left out of selection loses a pair on the mini subset.

    Each model is withheld in turn (all agents using it), tasks are selected on
    the remaining agents, and only pairs involving the withheld agents count.
    A fixed `selection` (random sampling) is reused for every model.
    """
    lost = set()
    for model in sorted({agent.split("__", 1)[1] for agent in agents}):
        held = {i for i, a in enumerate(agents) if a.endswith(f"__{model}")}
        train = np.array([i for i in range(len(agents)) if i not in held])
        for key, stratum in strata.items():
            chosen = (
                selection[key]
                if selection is not None
                else select(
                    stratum, k, method, train, exclude_saturated=exclude_saturated
                )
            )
            full = stratum.passes.mean(axis=0)
            mini = stratum.passes[chosen].mean(axis=0)
            if material_losses(full, mini, involving=held):
                lost.add(key)
    return len(lost)
