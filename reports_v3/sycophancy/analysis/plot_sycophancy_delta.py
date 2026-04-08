"""Plot sycophancy experiment: delta success rate vs baseline per environment.

Uses lama_aesthetics style with range_frame, ONE_COL sizing.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH
from lama_aesthetics.plotutils import range_frame

lama_aesthetics.get_style("main")

# ─── Config ──────────────────────────────────────────────────────────────────
POSITIVE_COLOR = "#16476A"
NEGATIVE_COLOR = "#BF092F"
BASELINE_COLOR = "#7A7A7A"

SYCOPHANCY_ROOT = Path(__file__).resolve().parent.parent
INTERVENTION_ROOT = SYCOPHANCY_ROOT.parent / "intervention"

ENVS = ["spectra", "wetlab", "retrosynthesis", "resistor", "ml", "catalyst"]
ENV_LABELS = {
    "spectra": "Spectra elucid.",
    "wetlab": "Inorg. analysis",
    "resistor": "Circuit inference",
    "ml": "ML property.",
    "catalyst": "Adsorp. surface",
    "retrosynthesis": "Retro. planning",
}

# ─── Load baseline data ─────────────────────────────────────────────────────
baseline_results = defaultdict(lambda: {"success": 0, "total": 0})
with open(INTERVENTION_ROOT / "analysis" / "results.csv") as f:
    for row in csv.DictReader(f):
        if row["intervention"] != "none" or row["agent"] != "react":
            continue
        env = row["env"]
        task_id = row["task_id"]
        baseline_results[(env, task_id)]["total"] += 1
        if row["success"] == "True":
            baseline_results[(env, task_id)]["success"] += 1

# ─── Load sycophancy data ───────────────────────────────────────────────────
syco_results = defaultdict(lambda: {"success": 0, "total": 0})
for env in ENVS:
    report_path = (
        SYCOPHANCY_ROOT
        / "runs"
        / env
        / "react"
        / "syco_misleading"
        / f"syco_{env}_react_misleading_report.json"
    )
    if not report_path.exists():
        continue
    report = json.loads(report_path.read_text())
    for task_id, task_data in report["task_results"].items():
        trials = (
            task_data if isinstance(task_data, list) else task_data.get("trials", [])
        )
        for t in trials:
            if isinstance(t, dict):
                syco_results[(env, task_id)]["total"] += 1
                if t.get("success"):
                    syco_results[(env, task_id)]["success"] += 1

# ─── Compute per-environment deltas ─────────────────────────────────────────


N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)


def bootstrap_delta(base_outcomes, syco_outcomes, n_boot=N_BOOTSTRAP, sample_size=None):
    """Bootstrap the delta in success rate between two conditions.

    Samples `sample_size` trials (with replacement) from each condition,
    computes success rate difference, repeats n_boot times.

    Returns: observed delta, 2.5th percentile, 97.5th percentile (all in %).
    """
    base = np.array(base_outcomes)
    syco = np.array(syco_outcomes)
    n = sample_size or min(len(base), len(syco))

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        b_sample = RNG.choice(base, size=n, replace=True)
        s_sample = RNG.choice(syco, size=n, replace=True)
        deltas[i] = s_sample.mean() - b_sample.mean()

    observed = syco.mean() - base.mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return observed * 100, lo * 100, hi * 100


env_data = []
for env in ENVS:
    syco_tasks = {tid for (e, tid) in syco_results if e == env}
    base_tasks = {tid for (e, tid) in baseline_results if e == env}
    common_tasks = syco_tasks & base_tasks

    if not common_tasks:
        continue

    # Build per-trial outcome arrays (1=success, 0=fail)
    base_outcomes = []
    for t in common_tasks:
        s = baseline_results[(env, t)]["success"]
        n = baseline_results[(env, t)]["total"]
        base_outcomes.extend([1] * s + [0] * (n - s))

    syco_outcomes = []
    for t in common_tasks:
        s = syco_results[(env, t)]["success"]
        n = syco_results[(env, t)]["total"]
        syco_outcomes.extend([1] * s + [0] * (n - s))

    delta, ci_lo, ci_hi = bootstrap_delta(base_outcomes, syco_outcomes)

    env_data.append(
        {
            "env": env,
            "label": ENV_LABELS[env],
            "delta": delta,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "base_n": len(base_outcomes),
            "syco_n": len(syco_outcomes),
        }
    )

# ─── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

x = np.arange(len(env_data))
deltas = [d["delta"] for d in env_data]
# Asymmetric error bars from bootstrap CI
lower_errs = [d["delta"] - d["ci_lo"] for d in env_data]
upper_errs = [d["ci_hi"] - d["delta"] for d in env_data]
colors = [NEGATIVE_COLOR if d < 0 else POSITIVE_COLOR for d in deltas]
labels = [d["label"] for d in env_data]

bars = ax.bar(x, deltas, color=colors, width=0.6, edgecolor="white", linewidth=0.3)
ax.errorbar(
    x,
    deltas,
    yerr=[lower_errs, upper_errs],
    fmt="none",
    ecolor="black",
    capsize=2.5,
    linewidth=0.7,
)

ax.axhline(0, color=BASELINE_COLOR, linewidth=0.6, linestyle="-", zorder=0)

for i, d in enumerate(env_data):
    va = "bottom" if d["delta"] >= 0 else "top"
    offset = 1.2 if d["delta"] >= 0 else -1.2
    ax.text(
        i,
        d["delta"] + offset,
        f'{d["delta"]:+.1f}%',
        ha="center",
        va=va,
        fontsize=5.5,
        fontweight="bold",
    )

ax.set_ylabel("$\\Delta$ Success Rate (%)")

y_range = np.array(
    [d["ci_lo"] for d in env_data] + [d["ci_hi"] for d in env_data] + deltas
)
range_frame(ax, x, y_range, pad=0.08)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=5.5, rotation=35, ha="right")

fig.tight_layout()

out_dir = Path(__file__).resolve().parent / "figures"
fig.savefig(out_dir / "sycophancy_delta.png", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "sycophancy_delta.pdf", bbox_inches="tight")
print("Saved to analysis/figures/sycophancy_delta.{png,pdf}")
