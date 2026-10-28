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


def load(revision: str, tasks_parquet: Path | None, prices_path: Path):
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
        passes = group.pivot_table("pass_at_5", "task_id", "agent_model")
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
) -> np.ndarray:
    """Return the task indexes chosen for one stratum, fitting only `agents`.

    Deterministic methods enumerate every k-task combination. Ties go to the
    lexicographically first combination.

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
    if method == "profile":
        order = np.argsort(mse, kind="stable")
    elif method == "cost_aware":
        cost = stratum.costs[:, agents].sum(axis=1)[combos].sum(axis=1)
        order = np.lexsort((mse, cost))
    elif method in {"rank", "rank_guarded"}:
        rho = row_spearman(target + error, target)
        order = np.lexsort((mse, worst, -rho))
    elif method == "rank_cost":
        rho = row_spearman(target + error, target)
        cost = stratum.costs[:, agents].sum(axis=1)[combos].sum(axis=1)
        order = np.lexsort((mse, cost, -rho))
    else:
        raise ValueError(f"unknown method: {method}")
    if method in {"cost_aware", "rank_guarded"}:
        order = order[guarded[order]]
    return combos[order[0]]


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
