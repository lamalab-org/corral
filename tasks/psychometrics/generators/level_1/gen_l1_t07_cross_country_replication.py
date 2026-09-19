#!/usr/bin/env python3
"""Generate Task 07 artifacts and scoring metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

TASK_ID = "psy_l1_t07_cross_country_replication"
SEED = 20260907
OUT_DIR = C.PKG_ROOT / "artifacts" / "level_1" / "task_07"
TASK_JSON = C.PKG_ROOT / "environments" / "level_1" / "tasks_json" / "task_07.json"

HSNS = C.HSNS_ITEMS
DD = C.DD_ITEMS
ITEMS = HSNS + DD

HSNS_F1 = ["HSNS1", "HSNS4", "HSNS5", "HSNS6", "HSNS8", "HSNS10"]
HSNS_F2 = ["HSNS2", "HSNS3", "HSNS7", "HSNS9"]
HSNS_LOADINGS = {
    "HSNS1": 0.55,
    "HSNS4": 0.52,
    "HSNS5": 0.70,
    "HSNS6": 0.48,
    "HSNS8": 0.71,
    "HSNS10": 0.66,
    "HSNS2": 0.76,
    "HSNS3": 0.58,
    "HSNS7": 0.69,
    "HSNS9": 0.52,
}
HSNS_PHI = 0.35

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
# Where the general factor is gone, the three traits are merely correlated.
DD_NO_GENERAL_PHI = 0.45
# Germany's Dirty Dozen: three items move subscale and everything is measured worse.
DD_SCRAMBLE = {"DDM3": "P", "DDP4": "N", "DDN4": "M"}

# country -> (N, hsns loading scale, hsns phi, dd general scale or None, scrambled)
COUNTRIES = {
    "US": (24000, 1.00, HSNS_PHI, 1.0, False),
    "GB": (6200, 1.00, HSNS_PHI, 1.0, False),
    "CA": (3800, 0.85, HSNS_PHI, 0.5, False),
    "AU": (3000, 1.00, HSNS_PHI, 0.3, False),
    "IN": (660, 0.85, HSNS_PHI, None, False),
    "BR": (510, 1.00, 0.95, None, False),
    "DE": (830, 0.85, HSNS_PHI, 0.6, True),
}
REPLICATION_COUNTRIES = [c for c in COUNTRIES if c != "US"]

TRUTH = {
    "hsns": {
        "GB": "exact",
        "CA": "approximate",
        "AU": "exact",
        "IN": "approximate",
        "BR": "substantive_only",
        "DE": "approximate",
    },
    "dirty_dozen": {
        "GB": "exact",
        "CA": "substantive_only",
        "AU": "substantive_only",
        "IN": "substantive_only",
        "BR": "substantive_only",
        "DE": "none",
    },
}

POP_REFERENCE_N = 400_000

PROMPT = """\
You are given responses from an online personality survey run in several countries. \
The codebook describes every column.

The survey contains two instruments. Establish a measurement model for each of them \
using the participants from the United States, then test how far each model carries \
over to every other country in the file. Classify each instrument in each country as \
exactly one of:

  exact             the model holds, with the same parameter values
  approximate       the same structure holds, but the parameter values differ
  substantive_only  that structure does not hold, but the instrument still measures
                    the same broad construct or constructs
  none              neither

Items are five-point ordinal ratings and 0 denotes a missing response.
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<the model you calibrated on the United States, covering both
                    instruments, in lavaan notation>",
  "replication": {
    "<instrument>": {"<country>": "<class>", ...},
    "<instrument>": {"<country>": "<class>", ...}
  }
}

Name the two instruments `hsns` and `dirty_dozen`. `model_syntax` is re-fitted on the
United States sample during evaluation, so it must be complete and runnable.
"""


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def simulate_hsns(n, rng, scale, phi):
    """Two correlated factors, optionally weakened or collapsed."""
    eta = rng.multivariate_normal([0, 0], [[1, phi], [phi, 1]], size=n)
    out = {}
    for item in HSNS:
        lam = HSNS_LOADINGS[item] * scale
        ystar = lam * eta[:, 0 if item in HSNS_F1 else 1] + rng.normal(0, np.sqrt(1 - lam**2), n)
        out[item] = C.categorize(ystar, C.THRESHOLDS[item])
    return pd.DataFrame(out)[HSNS]


def simulate_dd(n, rng, general_scale, scrambled):
    """General factor plus specifics, or - if general_scale is None - three
    correlated traits with no general factor at all."""
    assign = dict(SPECIFIC_OF)
    if scrambled:
        assign.update(DD_SCRAMBLE)
    out = {}
    if general_scale is None:
        names = ["M", "P", "N"]
        phi = np.full((3, 3), DD_NO_GENERAL_PHI)
        np.fill_diagonal(phi, 1.0)
        eta = rng.multivariate_normal(np.zeros(3), phi, size=n)
        for item in DD:
            lam = np.sqrt(DD_GENERAL[item] ** 2 + DD_SPECIFIC[item] ** 2)
            ystar = lam * eta[:, names.index(assign[item])] + rng.normal(0, np.sqrt(1 - lam**2), n)
            out[item] = C.categorize(ystar, C.THRESHOLDS[item])
        return pd.DataFrame(out)[DD]

    general = rng.normal(size=n)
    specific = {k: rng.normal(size=n) for k in dict.fromkeys(assign.values())}
    for item in DD:
        g = DD_GENERAL[item] * general_scale
        s = DD_SPECIFIC[item] * (0.6 if scrambled else 1.0)
        ystar = (
            g * general
            + s * specific[assign[item]]
            + rng.normal(0, np.sqrt(max(1 - g**2 - s**2, 1e-6)), n)
        )
        out[item] = C.categorize(ystar, C.THRESHOLDS[item])
    return pd.DataFrame(out)[DD]


def build_dataset(rng):
    """Build the full survey: every country, both instruments, demographics."""
    frames = []
    for country, (n, scale, phi, gen, scram) in COUNTRIES.items():
        frames.append(
            pd.concat(
                [
                    simulate_hsns(n, rng, scale, phi),
                    simulate_dd(n, rng, gen, scram),
                    C.demographics(n, rng, country),
                ],
                axis=1,
            )
        )
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=SEED)
    df[ITEMS] = df[ITEMS].mask(rng.random((len(df), len(ITEMS))) < C.ITEM_MISSING_RATE, 0)
    return df[ITEMS + ["age", "gender", "accuracy", "country"]].reset_index(drop=True)


def country_sample(df, country, items=None):
    """One country's respondents with complete answers on `items`."""
    items = items or ITEMS
    X = df[df.country == country][items]
    return X[(X != 0).all(axis=1)].astype(float)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def hsns_syntax():
    return f"F1 =~ {'+'.join(HSNS_F1)}\nF2 =~ {'+'.join(HSNS_F2)}"


def dd_bifactor():
    by = {k: [i for i in DD if SPECIFIC_OF[i] == k] for k in "MPN"}
    lines = ["G =~ " + "+".join(DD)]
    lines += [f"S{k} =~ {'+'.join(by[k])}" for k in "MPN"]
    lines += [f"G ~~ 0*S{k}" for k in "MPN"]
    lines += ["SM ~~ 0*SP", "SM ~~ 0*SN", "SP ~~ 0*SN"]
    return "\n".join(lines)


def dd_three_factor():
    by = {k: [i for i in DD if SPECIFIC_OF[i] == k] for k in "MPN"}
    return "\n".join(f"F{k} =~ {'+'.join(by[k])}" for k in "MPN")


def reference_syntax():
    """The joint model calibrated on the United States, covering both instruments."""
    return hsns_syntax() + "\n" + dd_bifactor()


def population_correlation_matrix():
    """Correlation matrix a perfectly specified US model would reproduce."""
    rng = np.random.default_rng(SEED + 999)
    n, scale, phi, gen, scram = COUNTRIES["US"]
    big = pd.concat(
        [
            simulate_hsns(POP_REFERENCE_N, rng, scale, phi),
            simulate_dd(POP_REFERENCE_N, rng, gen, scram),
        ],
        axis=1,
    )
    return C.population_matrix(big, ITEMS)


def _bic(spec, X, items):
    import semopy

    model = semopy.Model(spec)
    model.fit(X[items])
    stats = semopy.calc_stats(model)
    return float(stats["chi2"].iloc[0]) + len(model.param_vals) * np.log(len(X))


def _hsns_diagnostics(X, us_loadings):
    import semopy

    model = semopy.Model(hsns_syntax())
    model.fit(X[HSNS])
    stats = semopy.calc_stats(model)
    ins = model.inspect(std_est=True)
    load = ins[ins.op == "~"]
    fitted = {r["lval"]: abs(float(r["Est. Std"])) for _, r in load.iterrows()}
    cov = ins[
        (ins.op == "~~")
        & (ins.lval != ins.rval)
        & ins.lval.isin(["F1", "F2"])
        & ins.rval.isin(["F1", "F2"])
    ]
    phi = float(cov["Est. Std"].iloc[0]) if len(cov) else np.nan
    deviation = max(abs(fitted[i] - us_loadings[i]) for i in HSNS)
    return deviation, phi, float(stats["CFI"].iloc[0])


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
def build_truth(floor, pop, data_sha, rows):
    """Assemble the hidden ground truth, scoring reference and provenance."""
    return {
        "task_id": TASK_ID,
        "scored": {
            "replication": TRUTH,
            "calibration_country": "US",
        },
        "scoring_reference": {
            "reference_model_syntax": reference_syntax(),
            "reference_role": "floor",
            "reference_criteria": floor,
            "population_correlation_matrix": pop,
            "population_reference_n": POP_REFERENCE_N,
            "item_order": ITEMS,
        },
        "generative_parameters": {
            "hsns": {
                "loadings": HSNS_LOADINGS,
                "phi_us": HSNS_PHI,
                "f1_items": HSNS_F1,
                "f2_items": HSNS_F2,
            },
            "dirty_dozen": {
                "general_loadings": DD_GENERAL,
                "specific_loadings": DD_SPECIFIC,
                "specific_of_item": SPECIFIC_OF,
                "no_general_phi": DD_NO_GENERAL_PHI,
                "scramble": DD_SCRAMBLE,
            },
            "countries": {
                c: {
                    "n": v[0],
                    "hsns_loading_scale": v[1],
                    "hsns_phi": v[2],
                    "dd_general_scale": v[3],
                    "dd_scrambled": v[4],
                }
                for c, v in COUNTRIES.items()
            },
            "thresholds": C.THRESHOLDS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def build_task_json(data_sha):
    """Assemble the Corral task definition, including the scoring contract."""
    contract = C.scoring_contract(
        "artifacts/level_1/task_07/truth.json",
        ITEMS,
        [
            {
                "key": "replication",
                "fn": "score_label_panel",
                "truth_key": "scored.replication",
                "criterion": "replication",
            },
        ],
    )
    contract["syntax_whitelist"]["max_factors"] = 8
    return [
        {
            "id": TASK_ID,
            "name": "Which instrument travels across countries?",
            "uuid": "c24f7a08-6b31-4e5d-9a83-70d5c1e9b707",
            "keywords": [
                "psychometrics",
                "replication",
                "cross-cultural",
                "model selection",
                "measurement",
            ],
            "metrics": ["binary", "partial"],
            "level": 1,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                "dataset": "data.csv",
                "codebook": "codebook.md",
                "data_sha256": data_sha,
            },
            "tools": [],
            "scoring_function": "score_model_criteria",
            "scoring_params": contract,
        }
    ]


def candidate_submissions(X):
    """Answers an analyst might give, for the scoring tests."""
    del X
    correct = {
        "model_syntax": reference_syntax(),
        "replication": {k: dict(v) for k, v in TRUTH.items()},
    }
    everything_holds = {
        "model_syntax": reference_syntax(),
        "replication": {inst: {c: "exact" for c in v} for inst, v in TRUTH.items()},
    }
    dd_assumed_fine = {
        "model_syntax": reference_syntax(),
        "replication": {
            "hsns": dict(TRUTH["hsns"]),
            "dirty_dozen": {c: "exact" for c in TRUTH["dirty_dozen"]},
        },
    }
    swapped = {
        "model_syntax": reference_syntax(),
        "replication": {"hsns": dict(TRUTH["dirty_dozen"]), "dirty_dozen": dict(TRUTH["hsns"])},
    }
    return {
        "correct": correct,
        "fit-based: everything replicates": everything_holds,
        "HSNS right, Dirty Dozen assumed fine": dd_assumed_fine,
        "instruments swapped": swapped,
    }


def verify(df, pop):
    """Confirm each country's class is readable from the data."""
    import semopy

    us = country_sample(df, "US")
    model = semopy.Model(hsns_syntax())
    model.fit(us[HSNS])
    ins = model.inspect(std_est=True)
    us_loadings = {r["lval"]: abs(float(r["Est. Std"])) for _, r in ins[ins.op == "~"].iterrows()}

    print(f"US calibration sample: {len(us):,}\n")
    print(
        f"{'country':8s} {'N':>6s} | {'HSNS dev':>9s} {'phi':>6s} {'CFI':>7s} "
        f"{'class':17s} | {'DD bifac BIC':>12s} {'3-fac BIC':>10s} "
        f"{'winner':>9s} {'CFI':>7s} {'class':17s}"
    )
    rows = []
    for country in REPLICATION_COUNTRIES:
        X = country_sample(df, country)
        dev, phi, hcfi = _hsns_diagnostics(X, us_loadings)
        bif, three = _bic(dd_bifactor(), X, DD), _bic(dd_three_factor(), X, DD)
        dmodel = semopy.Model(dd_bifactor())
        dmodel.fit(X[DD])
        dcfi = float(semopy.calc_stats(dmodel)["CFI"].iloc[0])
        winner = "bifactor" if bif < three else "three"
        rows.append((country, dev, phi, hcfi, bif, three, winner, dcfi))
        print(
            f"{country:8s} {len(X):6,} | {dev:9.3f} {phi:6.3f} {hcfi:7.4f} "
            f"{TRUTH['hsns'][country]:17s} | {bif:12.1f} {three:10.1f} "
            f"{winner:>9s} {dcfi:7.4f} {TRUTH['dirty_dozen'][country]:17s}"
        )

    by_country = {r[0]: r for r in rows}
    checks = [
        (
            "the Dirty Dozen's structure holds only in GB",
            by_country["GB"][6] == "bifactor"
            and all(by_country[c][6] == "three" for c in ("CA", "AU", "IN", "BR")),
        ),
        (
            "yet the US model still fits everywhere it has stopped being best",
            all(by_country[c][7] > 0.95 for c in ("CA", "AU", "IN", "BR")),
        ),
        ("Germany's Dirty Dozen genuinely breaks down", by_country["DE"][7] < 0.95),
        (
            "the HSNS structure holds everywhere except BR",
            all(by_country[c][2] < 0.80 for c in ("GB", "CA", "AU", "IN", "DE"))
            and by_country["BR"][2] > 0.85,
        ),
        (
            "the HSNS parameter changes are visible where they were made",
            all(by_country[c][1] > 0.06 for c in ("CA", "IN", "DE")) and by_country["GB"][1] < 0.06,
        ),
    ]
    return C.report(checks)


def naive(df, pop):
    """Confirm that judging replication by fit alone gets the Dirty Dozen wrong."""
    del pop
    import semopy

    print("Judging replication by whether the US model still fits:\n")
    print(f"{'country':8s} {'HSNS CFI':>9s} {'DD CFI':>8s}  verdict from fit alone")
    wrong = 0
    for country in REPLICATION_COUNTRIES:
        X = country_sample(df, country)
        hm = semopy.Model(hsns_syntax())
        hm.fit(X[HSNS])
        dm = semopy.Model(dd_bifactor())
        dm.fit(X[DD])
        hcfi = float(semopy.calc_stats(hm)["CFI"].iloc[0])
        dcfi = float(semopy.calc_stats(dm)["CFI"].iloc[0])
        verdict = "replicates" if dcfi > 0.95 else "does not"
        truth = TRUTH["dirty_dozen"][country]
        if (verdict == "replicates") != (truth == "exact"):
            wrong += 1
        print(f"{country:8s} {hcfi:9.4f} {dcfi:8.4f}  Dirty Dozen {verdict}" f"   (truth: {truth})")
    print(
        f"\n  fit alone misclassifies {wrong} of {len(REPLICATION_COUNTRIES)} "
        f"countries for the Dirty Dozen"
    )
    return C.report([("fit alone gets most of the Dirty Dozen wrong", wrong >= 3)])


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    print("Simulating ...")
    df = build_dataset(rng)

    if action != "build":
        return verify(df, None) if action == "verify" else naive(df, None)

    print(f"Fitting the scoring reference (population draw N={POP_REFERENCE_N:,}) ...")
    pop = population_correlation_matrix()
    floor = C.evaluate_model(reference_syntax(), country_sample(df, "US"), ITEMS, pop)
    C.write_artifacts(
        OUT_DIR,
        TASK_JSON,
        df,
        lambda sha: build_truth(floor, pop.tolist(), sha, len(df)),
        build_task_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
