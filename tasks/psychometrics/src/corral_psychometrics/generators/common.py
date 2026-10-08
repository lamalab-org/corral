"""Shared simulation, fitting, and artifact helpers for task generators."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HSNS_ITEMS = [f"HSNS{i}" for i in range(1, 11)]
DD_ITEMS = [
    "DDP1",
    "DDP2",
    "DDP3",
    "DDP4",
    "DDN1",
    "DDN2",
    "DDN3",
    "DDN4",
    "DDM1",
    "DDM2",
    "DDM3",
    "DDM4",
]
ALL_ITEMS = HSNS_ITEMS + DD_ITEMS
COLUMNS = ALL_ITEMS + ["age", "gender", "accuracy", "country"]

# Respondents per country, following the imbalance typical of open web surveys.
N_BY_COUNTRY = {
    "US": 27000,
    "GB": 6200,
    "CA": 3800,
    "AU": 3000,
    "DE": 830,
    "IN": 660,
    "SE": 630,
    "PH": 560,
    "PL": 520,
    "BR": 510,
}
ITEM_MISSING_RATE = 0.012

# Four cut points per item. A respondent's unobserved score is cut at these to
# give the 1-5 answer, which is what makes the answers skewed rather than
# symmetric.
THRESHOLDS = {
    "HSNS1": [-1.701, -0.923, -0.487, 0.509],
    "HSNS2": [-0.948, -0.253, 0.109, 0.984],
    "HSNS3": [-1.351, -0.568, -0.189, 0.737],
    "HSNS4": [-1.057, -0.032, 0.44, 1.18],
    "HSNS5": [-1.32, -0.443, 0.013, 0.902],
    "HSNS6": [-2.084, -1.288, -0.752, 0.326],
    "HSNS7": [-1.451, -0.669, -0.25, 0.876],
    "HSNS8": [-1.247, -0.383, 0.016, 0.873],
    "HSNS9": [-1.591, -0.875, -0.445, 0.606],
    "HSNS10": [-0.712, 0.158, 0.538, 1.229],
    "DDP1": [-0.659, 0.026, 0.337, 1.06],
    "DDP2": [-0.625, 0.171, 0.554, 1.209],
    "DDP3": [-1.069, -0.35, 0.063, 0.925],
    "DDP4": [-1.559, -0.969, -0.507, 0.434],
    "DDN1": [-1.607, -1.037, -0.569, 0.509],
    "DDN2": [-1.241, -0.539, -0.008, 0.941],
    "DDN3": [-1.114, -0.371, 0.083, 0.888],
    "DDN4": [-0.884, 0.046, 0.526, 1.379],
    "DDM1": [-1.011, -0.296, 0.116, 1.007],
    "DDM2": [-1.565, -0.934, -0.591, 0.612],
    "DDM3": [-1.286, -0.523, -0.17, 0.865],
    "DDM4": [-0.81, 0.057, 0.58, 1.396],
}

ITEM_TEXT = {
    "HSNS1": "I can become entirely absorbed in thinking about my personal affairs, my health, my cares or my relations to others.",
    "HSNS2": "My feelings are easily hurt by ridicule or the slighting remarks of others.",
    "HSNS3": "When I enter a room I often become self conscious and feel that the eyes of others are upon me.",
    "HSNS4": "I dislike sharing the credit of an achievement with others.",
    "HSNS5": "I feel that I have enough on my hands without worrying about other people's troubles.",
    "HSNS6": "I feel that I am temperamentally different from most people.",
    "HSNS7": "I often interpret the remarks of others in a personal way.",
    "HSNS8": "I easily become wrapped up in my own interests and forget the existence of others.",
    "HSNS9": "I dislike being with a group unless I know that I am appreciated by at least one of those present.",
    "HSNS10": 'I am secretly "put out" or annoyed when other people come to me with their troubles, asking me for my time and sympathy.',
    "DDM1": "I tend to manipulate others to get my way.",
    "DDM2": "I have used deceit or lied to get my way.",
    "DDM3": "I have used flattery to get my way.",
    "DDM4": "I tend to exploit others towards my own end.",
    "DDP1": "I tend to lack remorse.",
    "DDP2": "I tend to not be too concerned with morality or the morality of my actions.",
    "DDP3": "I tend to be callous or insensitive.",
    "DDP4": "I tend to be cynical.",
    "DDN1": "I tend to want others to admire me.",
    "DDN2": "I tend to want others to pay attention to me.",
    "DDN3": "I tend to seek prestige or status.",
    "DDN4": "I tend to expect special favors from others.",
}


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def categorize(ystar, tau):
    """Turn unobserved scores into 1-5 answers.

    ystar  one unobserved score per respondent
    tau    the four cut points for this item
    """
    return np.searchsorted(np.asarray(tau), ystar).astype(int) + 1


def correlated_block(
    n,
    rng,
    loadings,
    phi_matrix,
    factor_names,
    taus,
    scale=1.0,
    cross=None,
    resid_corr=None,
    dif=None,
    female=None,
):
    """Answers from respondents whose traits are correlated with each other.

    Each item scores a respondent out of sight, then the cut points turn that
    score into a 1-5 answer:

        ystar = lam * eta + noise

    ystar  the unobserved score, before cut points make it a 1-5 answer
    eta    where the respondent stands on the trait the item measures, drawn so
           the traits correlate the way `phi_matrix` says
    lam    the item's loading: how much of the score the trait accounts for
    noise  everything else about the item, drawn with the variance the loading
           leaves over
    tau    the item's four cut points, from `taus`

    An item carries one unit of variance, which `lam` and `noise` divide: a
    `lam` of 0.7 accounts for 0.49 of it and leaves 0.51 to `noise`. `cross`
    and `resid_corr` add further terms, each paid for out of `noise` in turn.
    Every generator in this package uses these names.

    n            how many respondents
    rng          random generator
    loadings     {item: (factor, lam)}
    phi_matrix   correlations between the factors
    factor_names names in the order phi_matrix uses
    taus         {item: tau}
    scale        multiplies every loading, to make a worse-measured group
    cross        (item, factor, loading): give one item a second loading
    resid_corr   ((item, item), covariance): let two items agree beyond the
                 factor, as near-duplicate wording makes them
    dif          {item: shift}: move an item's cut points for one group
    female       which respondents `dif` applies to
    """
    eta = rng.multivariate_normal(np.zeros(len(factor_names)), phi_matrix, size=n)
    index = {f: i for i, f in enumerate(factor_names)}
    shared = rng.normal(size=n)
    out = {}
    for item, (factor, lam) in loadings.items():
        lam *= scale
        common = lam * eta[:, index[factor]]
        explained = lam**2
        if cross and item == cross[0]:
            extra = cross[2] * scale
            common = common + extra * eta[:, index[cross[1]]]
            explained += extra**2 + 2 * extra * lam * phi_matrix[index[factor], index[cross[1]]]
        if resid_corr and item in resid_corr[0]:
            common = common + np.sqrt(resid_corr[1]) * shared
            explained += resid_corr[1]
        ystar = common + rng.normal(0, np.sqrt(max(1 - explained, 1e-6)), n)
        tau = taus[item]
        if dif and item in dif and female is not None:
            shifted = [t + dif[item] for t in tau]
            out[item] = np.where(female, categorize(ystar, shifted), categorize(ystar, tau))
        else:
            out[item] = categorize(ystar, tau)
    return pd.DataFrame(out)


def bifactor_block(n, rng, general, specific, specific_of, items, taus):
    """Answers driven by one broad trait plus a narrow one per subscale.

        ystar = broad_lam * broad + narrow_lam * narrow + noise

    ystar       the unobserved score, before cut points make it a 1-5 answer
    broad       the one trait every item measures
    narrow      the trait of the item's own subscale
    broad_lam   the item's loading on `broad`, from `general`
    narrow_lam  the item's loading on `narrow`, from `specific`
    noise       everything else about the item, drawn with `left_over`

    `broad` and `narrow` are uncorrelated, so they claim separate shares of the
    item's one unit of variance and `noise` takes `left_over`, the rest.

    general      {item: broad_lam}
    specific     {item: narrow_lam}
    specific_of  {item: which narrow trait it belongs to}
    items        column order of the result
    taus         {item: tau}
    """
    broad = rng.normal(size=n)
    # dict.fromkeys keeps first-appearance order; iterating a set here would
    # shuffle the draws between processes, because string hashing is randomised.
    narrow = {name: rng.normal(size=n) for name in dict.fromkeys(specific_of.values())}
    out = {}
    for item in items:
        broad_lam, narrow_lam = general[item], specific[item]
        left_over = 1 - broad_lam**2 - narrow_lam**2
        ystar = (
            broad_lam * broad
            + narrow_lam * narrow[specific_of[item]]
            + rng.normal(0, np.sqrt(max(left_over, 1e-6)), n)
        )
        out[item] = categorize(ystar, taus[item])
    return pd.DataFrame(out)[items]


def demographics(n, rng, country):
    """The non-item columns for one country's respondents."""
    return pd.DataFrame(
        {
            "age": np.clip(rng.lognormal(np.log(22), 0.38, n).round(), 13, 89).astype(int),
            "gender": rng.choice([1, 2, 3, 0], size=n, p=[0.61, 0.37, 0.01, 0.01]),
            "accuracy": np.clip(rng.beta(6, 1.4, n) * 100, 1, 100).round().astype(int),
            "country": country,
        }
    )


def finalize(frames, rng, seed):
    """Shuffle the per-country blocks together and drop a few answers at random.

    A dropped answer is written as 0.
    """
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=seed)
    df[ALL_ITEMS] = df[ALL_ITEMS].mask(rng.random((len(df), len(ALL_ITEMS))) < ITEM_MISSING_RATE, 0)
    return df[COLUMNS].reset_index(drop=True)


def analysis_sample(df, items, country="US"):
    """One country's respondents who answered every item."""
    X = df[df.country == country][items]
    return X[(X != 0).all(axis=1)].astype(float)


# --------------------------------------------------------------------------
# Model fitting
# --------------------------------------------------------------------------
def evaluate_model(spec, X, items, pop):
    """Fit a model and measure it.

    Returns how well it fits (CFI, RMSEA, SRMR), how many parameters it spends
    (BIC), and how close the correlations it implies come to `pop`.

    spec   model in lavaan notation
    X      one row per respondent
    items  columns to fit on
    pop    correlations a perfectly specified model would reproduce
    """
    import semopy

    model = fit(spec, X, items)
    stats = semopy.calc_stats(model)

    sigma = model.calc_sigma()[0]
    order = list(model.vars["observed"])
    pick = [order.index(i) for i in items]
    sigma = sigma[np.ix_(pick, pick)]
    scale = np.sqrt(np.diag(sigma))
    implied = sigma / np.outer(scale, scale)
    empirical = np.corrcoef(X[items].values.T)
    upper = np.triu_indices(len(items), 1)

    chi2 = float(stats["chi2"].iloc[0])
    n_par = len(model.param_vals)
    return _rounded(
        {
            "df": float(stats["DoF"].iloc[0]),
            "chi2": chi2,
            "CFI": float(stats["CFI"].iloc[0]),
            "RMSEA": float(stats["RMSEA"].iloc[0]),
            "SRMR": float(np.sqrt(((empirical[upper] - implied[upper]) ** 2).mean())),
            "BIC": float(chi2 + n_par * np.log(len(X))),
            "n_free_parameters": n_par,
            "sigma_max_abs_deviation": float(np.abs(implied[upper] - pop[upper]).max()),
            "sigma_rms_deviation": float(np.sqrt(((implied[upper] - pop[upper]) ** 2).mean())),
        }
    )


def _rounded(criteria, places=3):
    """Round to a precision a rebuild reproduces and the tolerances can tell apart."""
    return {k: (round(v, places) if isinstance(v, float) else v) for k, v in criteria.items()}


def fit(spec, X, items=None):
    """Fit a model and return it.

    spec   model in lavaan notation
    X      one row per respondent
    items  columns to fit on; all of X by default
    """
    import semopy

    model = semopy.Model(spec)
    model.fit(X if items is None else X[items])
    return model


def estimates(model):
    """A fitted model's standardised estimates, as a table."""
    return model.inspect(std_est=True)


def loadings(model, factors=None):
    """How strongly each item tracks its factor, signed.

    An item that loads on two factors is reported under the stronger one.

    factors  restrict to these factors; all of them by default
    """
    rows = estimates(model)
    rows = rows[rows.op == "~"].copy()
    if factors is not None:
        rows = rows[rows.rval.isin(factors)]
    rows["value"] = pd.to_numeric(rows["Est. Std"], errors="coerce")
    best = rows.loc[rows["value"].abs().groupby(rows.lval).idxmax()]
    return {row.lval: float(row.value) for row in best.itertuples()}


def factor_correlations(model, factors):
    """Correlations between the latent factors, keyed by the pair of names."""
    rows = estimates(model)
    rows = rows[
        (rows.op == "~~")
        & (rows.lval != rows.rval)
        & rows.lval.isin(factors)
        & rows.rval.isin(factors)
    ]
    return {frozenset((r["lval"], r["rval"])): float(r["Est. Std"]) for _, r in rows.iterrows()}


def alpha(X):
    """Cronbach's alpha of a set of items, as it would be reported."""
    values = X.values.astype(float)
    k = values.shape[1]
    return float(
        k / (k - 1) * (1 - values.var(axis=0, ddof=1).sum() / values.sum(axis=1).var(ddof=1))
    )


def population_matrix(frame, items):
    """Correlations in a very large draw from the generating model.

    The target a submission's implied correlations are measured against.
    """
    return np.corrcoef(frame[items].values.T.astype(float)).round(3)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
def write_codebook(path, df):
    """Write the dataset description the agent can see.

    Lists the items and the response scale. Which questionnaire each item
    belongs to, and how the items group into subscales, is left out.
    """
    lines = [
        "# Codebook - online personality survey",
        "",
        "Tab-separated, one row per respondent. Item responses are ratings on a "
        "five-point scale: 1 = Disagree, 3 = Neutral, 5 = Agree. **0 = missed.**",
        "",
        "Items are taken from published self-report personality scales and were "
        "presented to respondents in a single block.",
        "",
        "| item | text |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in ITEM_TEXT.items()],
        "",
        "| variable | description |",
        "|---|---|",
        "| `age` | entered as free text |",
        "| `gender` | 1 = Male, 2 = Female, 3 = Other, 0 = missed |",
        "| `accuracy` | self-rated accuracy of own responses, 0-100 |",
        "| `country` | ISO country code |",
        "",
        f"Rows: {len(df):,}. Countries: "
        + ", ".join(f"{c} ({n:,})" for c, n in df.country.value_counts().items())
        + ".",
        "",
    ]
    path.write_text("\n".join(lines))


def provenance(generator, seed, rows, data_sha):
    """How this dataset was produced, for the record."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        rev = "unknown"
    return {
        "generator": generator,
        "seed": seed,
        "generated": date.today().isoformat(),
        "git_rev": rev,
        "rows": rows,
        "data_sha256": data_sha,
        "external_data_used": None,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def scoring_contract(task, items, tier_3_claims):
    """The scoring rules for a task.

    Stages 1 and 2 are identical for every task; only stage 3 differs.

    task           the task these rules belong to
    items          variables the model must cover
    tier_3_claims  what to check the submission's answers against
    """
    return {
        "data_dir": task.artifacts_relpath,
        "truth_path": task.truth_relpath,
        "dataset": "data.csv",
        "subset": {"country": "US"},
        "items": items,
        "scorer_estimator": "ML",
        "refit_submitted_syntax": True,
        "syntax_whitelist": {"items": items, "operators": ["=~", "~~", "~"], "max_factors": 6},
        "method": "constraints_then_pareto_then_claims",
        "aggregation": "all_tiers_must_pass",
        "emit": ["score_binary", "score_partial", "checks_vector"],
        # Stage 1: is the model usable at all? Breaking one of these makes a
        # model invalid rather than worse, so it is dropped before comparison.
        "tier_1_constraints": [
            {"key": "converged"},
            {"key": "no_negative_variance"},
            {"key": "positive_df"},
            {"key": "finite_standard_errors", "max_standard_error": 10.0},
            {"key": "no_redundant_factor", "phi_max": 0.90},
            {
                "key": "no_collapsed_factor",
                "min_salient_loading": 0.30,
                "min_salient_per_factor": 2,
                "sign_reversal_at": -0.10,
            },
        ],
        # Stage 2: the generating model sets a floor. A submission must be no
        # worse than it on any of these, and is allowed to be better.
        "reference_role": "floor",
        "tier_2_comparative": [
            {"key": "CFI", "direction": "higher", "eps": 0.005, "criterion": "global_fit"},
            {"key": "RMSEA", "direction": "lower", "eps": 0.005, "criterion": "global_fit"},
            {"key": "SRMR", "direction": "lower", "eps": 0.005, "criterion": "global_fit"},
            {"key": "BIC", "direction": "lower", "eps": 10.0, "criterion": "parsimony"},
            {
                "key": "sigma_max_abs_deviation",
                "direction": "lower",
                "eps": 0.010,
                "criterion": "accuracy",
            },
        ],
        "tier_3_claims": tier_3_claims,
        "recorded_fields": ["chi_square_p", "chi_square_df"],
    }


def write_json(path, payload):
    """Write JSON, keeping the existing build stamp when nothing substantive changed.

    Returns True when the file was written.
    """

    def stable(value):
        """The payload as JSON, without the fields that change on every build.

        Comparing the serialised form, rather than the objects, keeps a tuple in
        the payload equal to the list it was last written as.
        """
        if isinstance(value, dict) and "provenance" in value:
            value = dict(value)
            value["provenance"] = {
                k: v for k, v in value["provenance"].items() if k not in {"generated", "git_rev"}
            }
        return json.dumps(value, indent=2, sort_keys=True, default=str)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = None
        if stable(existing) == stable(payload):
            return False
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return True


def write_artifacts(task, df, truth_fn, task_json_fn):
    """Write the dataset, codebook, answer key and task definition.

    truth_fn and task_json_fn are each called with the dataset's checksum,
    which is known only once the file exists.
    """
    for directory in (task.artifacts, task.truth.parent, task.definition.parent):
        directory.mkdir(parents=True, exist_ok=True)

    data_path = task.artifacts / "data.csv"
    df.to_csv(data_path, sep="\t", index=False)
    sha = data_sha256(task.artifacts)
    write_codebook(task.artifacts / "codebook.md", df)
    write_json(task.truth, truth_fn(sha))
    write_json(task.definition, task_json_fn(sha))

    print(f"\n{len(df):,} rows -> {data_path}")
    for path in (task.artifacts / "codebook.md", task.truth, task.definition):
        print(f"           {path}")


def file_sha256(path):
    """Checksum of one written file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_sha256(out_dir):
    """Checksum of the written dataset."""
    return file_sha256(out_dir / "data.csv")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def mode():
    """Read the command line. Returns "verify", "naive" or "build"."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="check the intended answer wins")
    parser.add_argument("--naive", action="store_true", help="check the obvious analysis fails")
    args = parser.parse_args()
    return "verify" if args.verify else "naive" if args.naive else "build"


def report(checks):
    """Print each check and return an exit code.

    checks  list of (description, passed)
    """
    print("\nChecks")
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1
