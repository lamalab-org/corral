#!/usr/bin/env python3
"""Generate Task 07 artifacts and scoring metadata.

Generating model, United States: HSNS as two correlated factors (phi 0.35) and
the Dirty Dozen as a Machiavellianism-reference bifactor-(S-1) model.

Great Britain reproduces the US reference structure. Canada and Australia use
narcissism as the reference facet. India and Brazil have correlated first-order
facets; Germany changes reference facet and reassigns three items. HSNS changes
in several countries independently of the Dirty Dozen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l1_t07_cross_country_replication"
SEED = 20260907
TASK = paths.task(__file__)

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
N_REFERENCE_SPECIFIC_OVERRIDES = {"DDM1": 0.35, "DDM2": 0.75, "DDM3": 0.42, "DDM4": 0.66}

# Where the general factor is gone, the three traits are merely correlated.
DD_NO_GENERAL_PHI = 0.45
# Germany's Dirty Dozen: three items move subscale and everything is measured worse.
DD_SCRAMBLE = {"DDM3": "P", "DDP4": "N", "DDN4": "M"}

# country -> (N, HSNS loading scale, HSNS phi, DD reference loading scale,
#             reassigned items, DD reference facet)
COUNTRIES = {
    "US": (24000, 1.00, HSNS_PHI, 1.0, False, "M"),
    "GB": (6200, 1.00, HSNS_PHI, 1.0, False, "M"),
    "CA": (3800, 0.85, HSNS_PHI, 1.0, False, "N"),
    "AU": (3000, 1.00, HSNS_PHI, 1.0, False, "N"),
    "IN": (660, 0.85, HSNS_PHI, None, False, None),
    "BR": (510, 1.00, 0.95, None, False, None),
    "DE": (3000, 0.85, HSNS_PHI, 1.0, True, "N"),
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
REFERENCE_FACETS = {"US": "M", "GB": "M", "CA": "N", "AU": "N"}
REFERENCE_ANSWERS = {country: REFERENCE_FACETS.get(country) for country in COUNTRIES}

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
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<the model you calibrated on the United States, covering both
                    instruments, in semopy syntax>",
  "replication": {
    "hsns": {"<country>": "<class>", ...},
    "dirty_dozen": {"<country>": "<class>", ...}
  },
  "reference_facets": {"<country>": "<M|P|N> or null", ...}
}

Name the two instruments `hsns` and `dirty_dozen`. `model_syntax` must be complete
and runnable so the fit on the United States sample can be reproduced.
For every country, report the reference facet of a defensible Dirty Dozen
reference-facet model: Machiavellianism (`M`), psychopathy (`P`), or narcissism (`N`).
Use `null` when no canonical reference facet is defensible.
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
        out[item] = C.continuous_response(ystar, C.INTERCEPTS[item])
    return pd.DataFrame(out)[HSNS]


def simulate_dd(n, rng, general_scale, scrambled, reference):
    """S-1 reference facet or three correlated first-order facets."""
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
            out[item] = C.continuous_response(ystar, C.INTERCEPTS[item])
        return pd.DataFrame(out)[DD]

    if reference == "N":
        # Narcissism now defines the common factor. A strong residual M facet
        # makes this a different population, not just a relabelled US model.
        general = {
            i: DD_GENERAL[i]
            * (0.7 if SPECIFIC_OF[i] == "M" else 0.85 if SPECIFIC_OF[i] == "P" else 1.4)
            * general_scale
            for i in DD
        }
        specific = {i: DD_SPECIFIC[i] * (1.4 if SPECIFIC_OF[i] == "M" else 1.15) for i in DD}
        # Non-proportional reference/specific loadings distinguish the N-reference
        # model from a mere rotation of the M-reference model on continuous data.
        specific.update(N_REFERENCE_SPECIFIC_OVERRIDES)
    else:
        general = {i: DD_GENERAL[i] * general_scale for i in DD}
        specific = DD_SPECIFIC
    return C.bifactor_s1_block(
        n,
        rng,
        general=general,
        specific=specific,
        specific_of=assign,
        items=DD,
        intercepts=C.INTERCEPTS,
        reference=reference,
        specific_phi=0.25,
    )


def build_dataset(rng):
    """Build the full survey: every country, both instruments, demographics."""
    frames = []
    for country, (n, scale, phi, gen, scram, reference) in COUNTRIES.items():
        frames.append(
            pd.concat(
                [
                    simulate_hsns(n, rng, scale, phi),
                    simulate_dd(n, rng, gen, scram, reference),
                    C.demographics(n, rng, country),
                ],
                axis=1,
            )
        )
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=SEED)
    df[ITEMS] = df[ITEMS].mask(rng.random((len(df), len(ITEMS))) < C.ITEM_MISSING_RATE, np.nan)
    return df[ITEMS + ["age", "gender", "accuracy", "country"]].reset_index(drop=True)


def country_sample(df, country, items=None):
    """One country's respondents with complete answers on `items`."""
    items = items or ITEMS
    X = df[df.country == country][items]
    return X[X.notna().all(axis=1)].astype(float)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def hsns_syntax():
    return f"F1 =~ {'+'.join(HSNS_F1)}\nF2 =~ {'+'.join(HSNS_F2)}"


def dd_bifactor(reference="M"):
    by = {k: [i for i in DD if SPECIFIC_OF[i] == k] for k in "MPN"}
    lines = ["G =~ " + "+".join(DD)]
    specifics = [k for k in "MPN" if k != reference]
    lines += [f"S{k} =~ {'+'.join(by[k])}" for k in specifics]
    lines += [f"G ~~ 0*S{k}" for k in specifics]
    lines += [f"S{specifics[0]} ~~ S{specifics[1]}"]
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
    n, scale, phi, gen, scram, reference = COUNTRIES["US"]
    big = pd.concat(
        [
            simulate_hsns(POP_REFERENCE_N, rng, scale, phi),
            simulate_dd(POP_REFERENCE_N, rng, gen, scram, reference),
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
            "reference_facets": REFERENCE_ANSWERS,
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
                "n_reference_specific_overrides": N_REFERENCE_SPECIFIC_OVERRIDES,
                "n_reference_general_multipliers": {"M": 0.7, "P": 0.85, "N": 1.4},
                "n_reference_specific_multipliers": {"M": 1.4, "P": 1.15},
                "specific_factor_correlation": 0.25,
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
                    "dd_reference_facet": v[5],
                }
                for c, v in COUNTRIES.items()
            },
            "item_intercepts": C.INTERCEPTS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def build_task_json(data_sha):
    """Assemble the Corral task definition, including the scoring contract."""
    contract = C.scoring_contract(
        TASK,
        ITEMS,
        [
            {
                "key": "replication",
                "fn": "score_label_panel",
                "truth_key": "scored.replication",
                "criterion": "replication",
            },
            {
                "key": "reference_facets",
                "fn": "score_label_panel",
                "truth_key": "scored.reference_facets",
                "criterion": "reference_facet",
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


def candidate_submissions(X):
    """Answers an analyst might give, for the scoring tests."""
    del X
    correct = {
        "model_syntax": reference_syntax(),
        "replication": {k: dict(v) for k, v in TRUTH.items()},
        "reference_facets": dict(REFERENCE_ANSWERS),
    }
    everything_holds = {
        "model_syntax": reference_syntax(),
        "replication": {inst: {c: "exact" for c in v} for inst, v in TRUTH.items()},
        "reference_facets": dict(REFERENCE_ANSWERS),
    }
    dd_assumed_fine = {
        "model_syntax": reference_syntax(),
        "replication": {
            "hsns": dict(TRUTH["hsns"]),
            "dirty_dozen": {c: "exact" for c in TRUTH["dirty_dozen"]},
        },
        "reference_facets": dict(REFERENCE_ANSWERS),
    }
    swapped = {
        "model_syntax": reference_syntax(),
        "replication": {"hsns": dict(TRUTH["dirty_dozen"]), "dirty_dozen": dict(TRUTH["hsns"])},
        "reference_facets": dict(REFERENCE_ANSWERS),
    }
    return {
        "correct": correct,
        "fit-based: everything replicates": everything_holds,
        "HSNS right, Dirty Dozen assumed fine": dd_assumed_fine,
        "instruments swapped": swapped,
        "country reference facets assumed constant": {
            **correct,
            "reference_facets": {country: "M" for country in REFERENCE_ANSWERS},
        },
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
    reference_bic_gap = {
        country: _bic(dd_bifactor("M"), country_sample(df, country), DD)
        - _bic(dd_bifactor("N"), country_sample(df, country), DD)
        for country in REFERENCE_FACETS
    }
    de_assign = {item: DD_SCRAMBLE.get(item, SPECIFIC_OF[item]) for item in DD}
    de_syntax = "\n".join(
        f"F{k} =~ {'+'.join(item for item in DD if de_assign[item] == k)}" for k in "MPN"
    )
    de = country_sample(df, "DE")
    de_reassigned_bic = _bic(de_syntax, de, DD)
    checks = [
        (
            "the reference-facet switch is recoverable from model fit",
            reference_bic_gap["US"] < -5
            and reference_bic_gap["GB"] < -5
            and reference_bic_gap["CA"] > 5
            and reference_bic_gap["AU"] > 5,
        ),
        (
            "the US reference structure replicates in GB; IN and BR favor three traits",
            by_country["GB"][6] == "bifactor"
            and all(by_country[c][6] == "three" for c in ("IN", "BR")),
        ),
        (
            "yet the US model still fits everywhere it has stopped being best",
            all(by_country[c][7] > 0.95 for c in ("CA", "AU", "IN", "BR")),
        ),
        (
            "Germany's item reassignment is detectable",
            by_country["DE"][5] - de_reassigned_bic > 100,
        ),
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
        print(f"{country:8s} {hcfi:9.4f} {dcfi:8.4f}  Dirty Dozen {verdict}   (truth: {truth})")
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
        TASK,
        df,
        lambda sha: build_truth(floor, pop.tolist(), sha, len(df)),
        build_task_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
