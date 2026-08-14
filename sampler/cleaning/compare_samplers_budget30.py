"""Compare stratified_proportional_nested_min2 vs Bayes-IRT-greedy-nested-min2
at budget=30 on the cleaned pool, resistor folded into the unified budget
(not kept whole/excluded, per this round's request).

Two evaluation metrics, since they can disagree (and did, here):
  - "raw" (evaluate_subset): mini-set mean success rate per subject vs
    full-pool mean success rate per subject. What every earlier sweep in
    this project used.
  - "theta-recovery": re-estimate each subject's ability (MLE, item a/b held
    fixed at their full-pool posterior-mean values) using ONLY the selected
    items, compared to the reference theta from the full 93-item Bayesian
    fit. This is the metric the Fisher-information criterion actually
    targets -- included because the two metrics gave different answers and
    reporting only the more favorable one would be misleading.

Produces the manifests for both methods and a JSON of everything needed for
the comparison report.

Usage:
    python compare_samplers_budget30.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from bayesian_irt_selection import (
    build_bayes_item_order_per_env_level,
    make_stratified_bayes_sampler_nested,
)
from subsampling_core import SAMPLERS, evaluate_subset, load_matrix

sys.path.insert(0, str(Path(__file__).parent))
from plot_icc import (
    DEFAULT_EXCLUDE_MODELS,
    DEFAULT_SOURCE,
    DEFAULT_TRACE,
    load_fit,
)

DATA_DIR = Path(__file__).parent.parent / "data"
BUDGET = 30
MIN_PER_STRATUM = 2
N_REPEATS = 200
IRT_N_DRAWS = 200


def theta_recovery_eval(sel_items, item_idx, a_mean, b_mean, K, N, theta_full):
    n_subj = K.shape[0]
    idx = [item_idx[it] for it in sel_items]
    a_s, b_s = a_mean[idx], b_mean[idx]
    recovered = np.empty(n_subj)
    for j in range(n_subj):
        k_s, n_s = K[j, idx], N[j, idx]

        def neg_log_post(theta, a_s=a_s, b_s=b_s, k_s=k_s, n_s=n_s):
            logits = a_s * (theta - b_s)
            p = np.clip(1 / (1 + np.exp(-logits)), 1e-9, 1 - 1e-9)
            ll = np.sum(k_s * np.log(p) + (n_s - k_s) * np.log(1 - p))
            return -ll + 0.5 * theta**2

        recovered[j] = minimize_scalar(neg_log_post, bounds=(-6, 6), method="bounded").x
    mse = float(np.mean((recovered - theta_full) ** 2))
    rho = float(spearmanr(recovered, theta_full)[0])
    return mse, rho, recovered


def main() -> None:
    # ground truth for the "raw" metric: same 8-subject pool the IRT fit used
    # (fair comparison basis for both methods; matches what the info
    # criterion was built against)
    matrix, col_env = load_matrix(
        value="success",
        exclude_environments=None,
        exclude_models=DEFAULT_EXCLUDE_MODELS,
        data_path=Path(DEFAULT_SOURCE),
    )
    all_items = list(matrix.columns)
    logger.info(
        f"{len(all_items)} candidate items, {matrix.shape[0]} subjects, budget={BUDGET}"
    )

    trace, subjects, item_list, K, N, item_env = load_fit(
        DEFAULT_TRACE, Path(DEFAULT_SOURCE), [], DEFAULT_EXCLUDE_MODELS
    )
    theta_full = trace.posterior["theta"].mean(dim=["chain", "draw"]).to_numpy()
    a_mean = trace.posterior["a"].mean(dim=["chain", "draw"]).to_numpy()
    b_mean = trace.posterior["b"].mean(dim=["chain", "draw"]).to_numpy()
    item_idx = {it: i for i, it in enumerate(item_list)}

    # ---- method 1: stratified_proportional_nested_min2, 200 repeats ----
    sampler1 = SAMPLERS["stratified_proportional_nested_min2"]
    rng = np.random.default_rng(0)
    rows1 = []
    for rep in range(N_REPEATS):
        sel = sampler1(all_items, BUDGET, rng, col_env)
        m = evaluate_subset(matrix, col_env, sel)
        t_mse, t_rho, _ = theta_recovery_eval(
            sel, item_idx, a_mean, b_mean, K, N, theta_full
        )
        rows1.append(
            {
                "repeat": rep,
                "raw_mse": m["global_mse"],
                "raw_spearman": m["global_spearman"],
                "theta_mse": t_mse,
                "theta_spearman": t_rho,
            }
        )
    df1 = pd.DataFrame(rows1)
    logger.info(
        f"stratified_proportional_nested_min2: raw_mse median={df1.raw_mse.median():.5f}, "
        f"theta_mse median={df1.theta_mse.median():.5f}"
    )
    # representative single manifest: the repeat closest to median raw_mse, for reproducibility
    rep_idx = (df1.raw_mse - df1.raw_mse.median()).abs().idxmin()
    rng_repro = np.random.default_rng(0)
    for _ in range(int(df1.loc[rep_idx, "repeat"]) + 1):
        sel1_final = sampler1(all_items, BUDGET, rng_repro, col_env)

    # ---- method 2: Bayes-IRT-greedy nested, deterministic ----
    orders = build_bayes_item_order_per_env_level(
        trace, item_list, item_env, n_draws_used=IRT_N_DRAWS, seed=0
    )
    sampler2 = make_stratified_bayes_sampler_nested(
        orders, allocation_mode="proportional", min_per_stratum=MIN_PER_STRATUM
    )
    sel2 = sampler2(all_items, BUDGET, None, col_env)
    m2 = evaluate_subset(matrix, col_env, sel2)
    t_mse2, t_rho2, recovered2 = theta_recovery_eval(
        sel2, item_idx, a_mean, b_mean, K, N, theta_full
    )
    logger.info(
        f"bayes_irt_greedy_nested_min2: raw_mse={m2['global_mse']:.5f}, theta_mse={t_mse2:.5f}"
    )

    # ---- per-subject theta recovery detail for method 2 (the mechanism plot) ----
    per_subject = [
        {
            "subject": s,
            "theta_full": float(theta_full[j]),
            "theta_recovered": float(recovered2[j]),
        }
        for j, s in enumerate(subjects)
    ]

    # ---- allocation (identical by construction between the two methods) ----
    def item_key(it):
        env, level, task = it.split("|")
        return env, level, task

    def to_rows(sel, status_label):
        return [
            {"environment": e, "level": lv, "task": t, "status": status_label}
            for e, lv, t in map(item_key, sel)
        ]

    df_strat = pd.DataFrame(to_rows(sel1_final, "sampled"))
    df_bayes = pd.DataFrame(to_rows(sel2, "sampled"))
    alloc_strat = (
        df_strat.groupby(["environment", "level"]).size().rename("n").reset_index()
    )
    alloc_bayes = (
        df_bayes.groupby(["environment", "level"]).size().rename("n").reset_index()
    )

    # ---- b (difficulty) of selected items, both methods, vs full pool ----
    def b_of(sel):
        return [float(b_mean[item_idx[it]]) for it in sel]

    b_full = [float(x) for x in b_mean]
    b_strat = b_of(sel1_final)
    b_bayes = b_of(sel2)

    # ---- overlap between the two selections ----
    overlap = sorted(set(sel1_final) & set(sel2))

    # ---- save manifests ----
    df_strat.sort_values(["environment", "level", "task"]).to_csv(
        DATA_DIR / "corral_mini_selected_tasks_budget30_stratified_min2.csv",
        index=False,
    )
    df_bayes.sort_values(["environment", "level", "task"]).to_csv(
        DATA_DIR / "corral_mini_selected_tasks_budget30_bayes_irt_min2.csv", index=False
    )
    df1.to_csv(DATA_DIR / "sweep_budget30_stratified_min2.csv", index=False)

    out = {
        "budget": BUDGET,
        "min_per_stratum": MIN_PER_STRATUM,
        "n_repeats": N_REPEATS,
        "stratified_raw_mse": df1["raw_mse"].tolist(),
        "stratified_raw_spearman": df1["raw_spearman"].tolist(),
        "stratified_theta_mse": df1["theta_mse"].tolist(),
        "stratified_theta_spearman": df1["theta_spearman"].tolist(),
        "bayes_raw_mse": m2["global_mse"],
        "bayes_raw_spearman": m2["global_spearman"],
        "bayes_theta_mse": t_mse2,
        "bayes_theta_spearman": t_rho2,
        "alloc_strat": alloc_strat.to_dict("records"),
        "alloc_bayes": alloc_bayes.to_dict("records"),
        "b_full": b_full,
        "b_strat": b_strat,
        "b_bayes": b_bayes,
        "per_subject": per_subject,
        "n_overlap": len(overlap),
        "n_selected": len(sel1_final),
        "subject_theta_ability": [
            {"subject": s, "theta": float(theta_full[j])}
            for j, s in enumerate(subjects)
        ],
    }
    out_path = Path(
        "/private/tmp/claude-501/-Users-n0w0f-git-corral-rebuttal-corral/3486a884-ca3b-4f7e-aec0-dec6593dc91b/scratchpad/compare_samplers_budget30.json"
    )
    out_path.write_text(json.dumps(out))
    logger.success(f"Saved comparison data -> {out_path}")
    logger.success(
        "Saved manifests -> corral_mini_selected_tasks_budget30_{stratified,bayes_irt}_min2.csv"
    )


if __name__ == "__main__":
    main()
