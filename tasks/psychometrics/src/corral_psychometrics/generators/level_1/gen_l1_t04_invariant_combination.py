#!/usr/bin/env python3
"""Generate Task 04 artifacts and scoring metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l1_t04_invariant_combination"
SEED = 20261001
TASK = paths.task(__file__)

HSNS = C.HSNS_ITEMS
DD = C.DD_ITEMS
ALL_VARS = HSNS + DD + ["gender"]
MODEL_VARS = DD + ["gender"]  # the combination that works

# --------------------------------------------------------------------------
# HSNS: a real gender difference, hidden by bias against women on four items
# --------------------------------------------------------------------------
HSNS_LOADINGS = {
    "HSNS1": 0.55,
    "HSNS2": 0.76,
    "HSNS3": 0.58,
    "HSNS4": 0.52,
    "HSNS5": 0.70,
    "HSNS6": 0.48,
    "HSNS7": 0.69,
    "HSNS8": 0.71,
    "HSNS9": 0.60,
    "HSNS10": 0.66,
}
HSNS_LATENT_DIFFERENCE = 0.45
HSNS_BIASED_ITEMS = ["HSNS1", "HSNS5", "HSNS8", "HSNS10"]
HSNS_BIAS_SHIFT = 0.55

# --------------------------------------------------------------------------
# Dirty Dozen: a general factor plus three specifics, identical in both groups.
# Women simply have more of the general trait. Nothing here is biased.
# --------------------------------------------------------------------------
SPECIFIC_OF = {
    "DDM1": "M",
    "DDM2": "M",
    "DDM3": "M",
    "DDM4": "M",
    "DDP1": "P",
    "DDP2": "P",
    "DDP3": "P",
    "DDP4": "P",
    "DDN1": "N",
    "DDN2": "N",
    "DDN3": "N",
    "DDN4": "N",
}
DD_GENERAL = {
    "DDM1": 0.58,
    "DDM2": 0.52,
    "DDM3": 0.48,
    "DDM4": 0.60,
    "DDP1": 0.55,
    "DDP2": 0.50,
    "DDP3": 0.53,
    "DDP4": 0.38,
    "DDN1": 0.42,
    "DDN2": 0.40,
    "DDN3": 0.50,
    "DDN4": 0.52,
}
DD_SPECIFIC = {
    "DDM1": 0.48,
    "DDM2": 0.44,
    "DDM3": 0.42,
    "DDM4": 0.50,
    "DDP1": 0.52,
    "DDP2": 0.48,
    "DDP3": 0.55,
    "DDP4": 0.35,
    "DDN1": 0.58,
    "DDN2": 0.56,
    "DDN3": 0.45,
    "DDN4": 0.30,
}
DD_LATENT_DIFFERENCE = 0.30

# Outside the US there is no item bias, and the gender difference on the Dirty
# Dozen runs the other way, so pooling the countries does not merely dilute the
# US answer - it moves it well outside tolerance.
LOADING_SCALE_NON_US = 0.80
NON_US_DD_DIFFERENCE = -0.25

POP_REFERENCE_N = 1_200_000

PROMPT = """\
You are given responses from an online personality survey. The codebook describes \
every column.

Read the data and select participants from the United States. Two instruments were \
administered. Determine which of them supports a comparison of men and women - that \
is, which one measures the same thing in both groups - and report the size of the \
gender difference on it, as the standardised coefficient of the latent trait regressed \
on `gender`.

Also report which items, in the instrument that does not support the comparison, \
function differently across gender.

Restrict the comparison to respondents who reported male or female.\
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<lavaan syntax for the instrument you selected>",
  "latent_difference": <float>,
  "biased_items": ["<items in the OTHER instrument that function differently>"]
}

`model_syntax` uses lavaan notation (`=~` loadings, `~` regressions, `~~`
(co)variances, `0*` to fix a parameter to zero).
"""


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def simulate_hsns(n, rng, female, scale=1.0, us=True):
    """One trait, a real gender difference, and bias against women on four items."""
    eta = rng.normal(0.0, 1.0, n) + (np.where(female, HSNS_LATENT_DIFFERENCE, 0.0) if us else 0.0)
    out = {}
    for item in HSNS:
        lam = HSNS_LOADINGS[item] * scale
        ystar = lam * eta + rng.normal(0, np.sqrt(1 - lam**2), n)
        tau = np.asarray(C.THRESHOLDS[item])
        if us and item in HSNS_BIASED_ITEMS:
            out[item] = np.where(
                female,
                C.categorize(ystar, tau + HSNS_BIAS_SHIFT),
                C.categorize(ystar, tau),
            )
        else:
            out[item] = C.categorize(ystar, tau)
    return pd.DataFrame(out)[HSNS]


def simulate_dd(n, rng, female, scale=1.0, us=True):
    """A general factor plus three specifics, identical in both groups."""
    shift = DD_LATENT_DIFFERENCE if us else NON_US_DD_DIFFERENCE
    general = rng.normal(0.0, 1.0, n) + np.where(female, shift, 0.0)
    specific = {k: rng.normal(size=n) for k in dict.fromkeys(SPECIFIC_OF.values())}
    out = {}
    for item in DD:
        g, s = DD_GENERAL[item] * scale, DD_SPECIFIC[item] * scale
        ystar = (
            g * general
            + s * specific[SPECIFIC_OF[item]]
            + rng.normal(0, np.sqrt(max(1 - g**2 - s**2, 1e-6)), n)
        )
        out[item] = C.categorize(ystar, C.THRESHOLDS[item])
    return pd.DataFrame(out)[DD]


def build_dataset(rng):
    """Build the full survey: every country, both instruments, demographics."""
    frames = []
    for country, n in C.N_BY_COUNTRY.items():
        us = country == "US"
        demo = C.demographics(n, rng, country)
        female = (demo.gender == 2).to_numpy()
        scale = 1.0 if us else LOADING_SCALE_NON_US
        frames.append(
            pd.concat(
                [
                    simulate_hsns(n, rng, female, scale, us),
                    simulate_dd(n, rng, female, scale, us),
                    demo,
                ],
                axis=1,
            )
        )
    return C.finalize(frames, rng, SEED)


def comparison_sample(df, variables):
    """US respondents reporting male or female, with complete answers."""
    sub = df[(df.country == "US") & df.gender.isin([1, 2])]
    items = [v for v in variables if v != "gender"]
    X = sub[variables]
    return X[(X[items] != 0).all(axis=1)].astype(float)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def dd_bifactor():
    """The Dirty Dozen's real structure: general factor plus three specifics."""
    by = {k: [i for i in DD if SPECIFIC_OF[i] == k] for k in "MPN"}
    lines = [f"G =~ {'+'.join(DD)}"]
    lines += [f"S{k} =~ {'+'.join(by[k])}" for k in "MPN"]
    lines += [f"G ~~ 0*S{k}" for k in "MPN"]
    lines += ["SM ~~ 0*SP", "SM ~~ 0*SN", "SP ~~ 0*SN"]
    return "\n".join(lines)


def dd_three_factor():
    """The Dirty Dozen's nominal subscale structure."""
    by = {k: [i for i in DD if SPECIFIC_OF[i] == k] for k in "MPN"}
    return "\n".join(f"F{k} =~ {'+'.join(by[k])}" for k in "MPN")


def reference_syntax():
    """The combination that works: the Dirty Dozen under its real structure."""
    return dd_bifactor() + "\nG ~ gender\nSM ~ gender\nSP ~ gender\nSN ~ gender"


def candidate_models():
    """Routes an analyst might take, including the ones that go wrong."""
    hsns_base = f"F =~ {'+'.join(HSNS)}\nF ~ gender"
    return {
        "HSNS, assumes no bias": hsns_base,
        "HSNS, bias freed": hsns_base + "".join(f"\n{i} ~ gender" for i in HSNS_BIASED_ITEMS),
        "DD, nominal three-factor": dd_three_factor() + "\nFM ~ gender\nFP ~ gender\nFN ~ gender",
        "DD, bifactor (CORRECT)": reference_syntax(),
    }


def latent_difference(spec, X, variables, factor="G"):
    """Standardised coefficient of the named latent trait regressed on gender."""
    import semopy

    model = semopy.Model(spec)
    model.fit(X[variables])
    ins = model.inspect(std_est=True)
    row = ins[(ins.op == "~") & (ins.lval == factor) & (ins.rval == "gender")]
    return float(pd.to_numeric(row["Est. Std"], errors="coerce").iloc[0])


def population_reference():
    """The answer a correct analysis converges on, plus the population matrix."""
    rng = np.random.default_rng(SEED + 999)
    female = rng.random(POP_REFERENCE_N) < 0.377
    big = simulate_dd(POP_REFERENCE_N, rng, female)
    big["gender"] = np.where(female, 2, 1)
    target = latent_difference(reference_syntax(), big, MODEL_VARS)
    pop = np.corrcoef(big[MODEL_VARS].values.T.astype(float))
    return target, pop.round(3)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
def build_truth(target, floor, pop, data_sha, rows):
    """Assemble the hidden ground truth, scoring reference and provenance."""
    return {
        "task_id": TASK_ID,
        "scored": {
            "instrument": "dirty_dozen",
            "model_family": "bifactor_general_plus_specifics",
            "latent_difference": round(target, 3),
            "biased_items": sorted(HSNS_BIASED_ITEMS),
            "spurious_bias_in_selected_instrument": [],
        },
        "scoring_reference": {
            "reference_model_syntax": reference_syntax(),
            "reference_role": "floor",
            "reference_criteria": floor,
            "population_correlation_matrix": pop,
            "population_reference_n": POP_REFERENCE_N,
            "item_order": MODEL_VARS,
        },
        "generative_parameters": {
            "hsns": {
                "loadings": HSNS_LOADINGS,
                "latent_difference_women_minus_men": HSNS_LATENT_DIFFERENCE,
                "biased_items": HSNS_BIASED_ITEMS,
                "bias_threshold_shift": HSNS_BIAS_SHIFT,
                "bias_direction": "against women",
                "invariant": False,
            },
            "dirty_dozen": {
                "general_loadings": DD_GENERAL,
                "specific_loadings": DD_SPECIFIC,
                "specific_of_item": SPECIFIC_OF,
                "latent_difference_women_minus_men": DD_LATENT_DIFFERENCE,
                "biased_items": [],
                "invariant": True,
            },
            "non_us": {
                "loading_scale": LOADING_SCALE_NON_US,
                "dirty_dozen_latent_difference": NON_US_DD_DIFFERENCE,
                "hsns_latent_difference": 0.0,
                "biased_items": [],
            },
            "thresholds": C.THRESHOLDS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def build_task_json(data_sha):
    """Assemble the Corral task definition, including the scoring contract."""
    contract = C.scoring_contract(
        TASK,
        MODEL_VARS,
        [
            {
                "key": "latent_difference",
                "fn": "score_scalar",
                "truth_key": "scored.latent_difference",
                "tol": 0.06,
                "criterion": "group_comparison",
            },
            # The selected instrument must show no item bias: any direct gender
            # path in the submitted model is a claim of bias that is not there.
            {
                "key": "spurious_bias",
                "fn": "score_item_set",
                "derive_from": "refit_covariate_paths",
                "covariate": "gender",
                "truth_key": "scored.spurious_bias_in_selected_instrument",
                "criterion": "item_bias",
            },
            # Evidence that the rejected instrument was actually analysed.
            {
                "key": "biased_items",
                "fn": "score_item_set",
                "truth_key": "scored.biased_items",
                "criterion": "item_bias",
            },
        ],
    )
    contract["subset"] = {"country": "US", "gender": [1, 2]}
    contract["syntax_whitelist"]["items"] = ALL_VARS
    contract["syntax_whitelist"]["max_factors"] = 8
    return [
        {
            "id": TASK_ID,
            "name": "Which instrument supports a gender comparison?",
            "uuid": "a7e35f90-2c14-4b8d-8f66-1d90e4b7c201",
            "keywords": [
                "psychometrics",
                "measurement invariance",
                "item bias",
                "group comparison",
                "model selection",
            ],
            "metrics": ["binary", "partial"],
            "level": 1,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                # Exactly the files copied into the agent's workspace.
                "public_inputs": [
                    "codebook.md",
                    "data.csv",
                ],
                "dataset": "data.csv",
                "codebook": "codebook.md",
                "data_sha256": data_sha,
            },
            "tools": [],
            "scoring_function": "score_model_criteria",
            "scoring_params": contract,
        }
    ]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def _bias_scan(spec, X, variables, items, factors):
    """Items that look biased: chi-square gain from freeing a direct gender path."""
    import semopy
    from scipy.stats import chi2 as chi2_dist

    base = spec + "".join(f"\n{f} ~ gender" for f in factors)
    model = semopy.Model(base)
    model.fit(X[variables])
    chi0 = float(semopy.calc_stats(model)["chi2"].iloc[0])
    flagged = []
    for item in items:
        m = semopy.Model(base + f"\n{item} ~ gender")
        m.fit(X[variables])
        gain = chi0 - float(semopy.calc_stats(m)["chi2"].iloc[0])
        if gain > chi2_dist.ppf(0.999, 1):
            flagged.append(item)
    return flagged


def _bias_recover(spec, X, variables, items, factors):
    """Items that behave differently across gender, found by freeing every path.

    Freeing all of them at once leaves the overall level unidentified, so the
    estimates centre themselves: items carrying bias in one direction separate
    cleanly from the rest. Scanning items one at a time does not work here,
    because the no-bias baseline is itself contaminated by the bias.
    """
    import semopy

    full = (
        spec
        + "".join(f"\n{f} ~ gender" for f in factors)
        + "".join(f"\n{i} ~ gender" for i in items)
    )
    model = semopy.Model(full)
    model.fit(X[variables])
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.rval == "gender") & ins.lval.isin(items)]
    effect = rows.assign(v=pd.to_numeric(rows["Est. Std"], errors="coerce")).set_index("lval")["v"]
    return sorted(effect[effect < effect.median() - 0.05].index)


def verify(df, target, pop):
    """Confirm only one instrument-and-model combination supports the comparison."""
    X_dd = comparison_sample(df, MODEL_VARS)
    X_hs = comparison_sample(df, HSNS + ["gender"])
    men, women = X_dd[X_dd.gender == 1], X_dd[X_dd.gender == 2]
    print(f"US comparison sample: {len(men):,} men, {len(women):,} women\n")

    hs_flagged = _bias_recover(f"F =~ {'+'.join(HSNS)}", X_hs, HSNS + ["gender"], HSNS, ["F"])
    dd3_flagged = _bias_scan(dd_three_factor(), X_dd, MODEL_VARS, DD, ["FM", "FP", "FN"])
    ddb_flagged = _bias_scan(dd_bifactor(), X_dd, MODEL_VARS, DD, ["G", "SM", "SP", "SN"])

    print("  items that look biased across gender:")
    print(f"    HSNS, one factor              {hs_flagged}")
    print(f"    DD, nominal three-factor      {dd3_flagged}")
    print(f"    DD, bifactor (real structure) {ddb_flagged if ddb_flagged else 'NONE'}")

    estimates = {}
    for name, spec in candidate_models().items():
        variables = MODEL_VARS if name.startswith("DD") else HSNS + ["gender"]
        factor = "G" if "bifactor" in name else ("FM" if "nominal" in name else "F")
        estimates[name] = latent_difference(
            spec, X_dd if name.startswith("DD") else X_hs, variables, factor
        )
    print("\n  latent gender difference by route:")
    for name, value in estimates.items():
        print(f"    {name:32s} {value:+.3f}")
    print(f"\n  calibrated target (DD, bifactor)   {target:+.3f}")

    checks = [
        ("the HSNS really is biased", set(hs_flagged) == set(HSNS_BIASED_ITEMS)),
        (
            "the Dirty Dozen looks biased under its nominal structure",
            len(dd3_flagged) > 0,
        ),
        ("but is clean under its real structure", ddb_flagged == []),
        (
            "the correct route lands on the calibrated target",
            abs(estimates["DD, bifactor (CORRECT)"] - target) <= 0.06,
        ),
    ]
    return C.report(checks)


def naive(df, target, pop):
    """Confirm the face-value analysis rejects both instruments."""
    del pop
    X_dd = comparison_sample(df, MODEL_VARS)
    X_hs = comparison_sample(df, HSNS + ["gender"])
    hs = _bias_scan(f"F =~ {'+'.join(HSNS)}", X_hs, HSNS + ["gender"], HSNS, ["F"])
    dd = _bias_scan(dd_three_factor(), X_dd, MODEL_VARS, DD, ["FM", "FP", "FN"])
    print("  taking each instrument at face value:")
    print(f"    HSNS (one factor)         -> {len(hs)} biased items {hs}")
    print(f"    Dirty Dozen (3 subscales) -> {len(dd)} biased items {dd}")
    print("    conclusion: neither instrument supports the comparison")
    print(f"  truth: the Dirty Dozen does, and the difference is {target:+.3f}")
    return C.report(
        [
            (
                "the face-value analysis rejects both instruments",
                len(hs) > 0 and len(dd) > 0,
            ),
            (
                "the items it flags in the Dirty Dozen are not really biased",
                len(dd) > 0,
            ),
        ]
    )


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    print("Simulating ...")
    df = build_dataset(rng)
    print(f"Calibrating the target (population draw N={POP_REFERENCE_N:,}) ...")
    target, pop = population_reference()

    if action != "build":
        return verify(df, target, pop) if action == "verify" else naive(df, target, pop)

    floor = C.evaluate_model(reference_syntax(), comparison_sample(df, MODEL_VARS), MODEL_VARS, pop)
    C.write_artifacts(
        TASK,
        df,
        lambda sha: build_truth(target, floor, pop.tolist(), sha, len(df)),
        build_task_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
