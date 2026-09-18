"""Scoring for the psychometrics environment.

A submission gives a model, not a number, so scoring is not a comparison of
scalars. It runs in three stages. All three must pass, and they are never
added up into a weighted total.

  Stage 1, constraints. Is the model usable at all? A negative variance, two
  factors correlated above .90, or a factor no item really loads on make a
  model invalid rather than merely worse. Such a model is dropped here,
  however well it fits.

  Stage 2, comparison. The model that generated the data sets a floor. The
  submission has to be at least as good as it on every measure - three of fit,
  one of how many parameters it spends, and one of how close the correlations
  it implies come to the truth - allowing a small margin for sampling noise.
  Beating the floor is fine and never counts against a submission.

  Stage 3, claims. The numbers the agent reported, checked against the values
  the data were generated from. Not against what the reference model happens to
  estimate, so a better model is rewarded rather than penalised.

Results:

    score_binary   1.0 only if all three stages pass
    score_partial  share of applicable checks passed; for diagnosis only
    checks_vector  {name: "PASS" | "FAIL" | "n/a"}
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# --------------------------------------------------------------------------
# Submitted-syntax validation
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ALLOWED_OPS = ("=~", "~~", "~")


class InvalidSubmission(ValueError):
    """The submitted model cannot be accepted as input."""


def validate_syntax(spec: str, items: list[str], max_factors: int = 6) -> str:
    """Check a submitted model before fitting it.

    Only known variables and a few operators are allowed, so a submission
    cannot smuggle in code or refer to columns it was not given.

    lavaan syntax is declarative rather than executable, but agent output is
    still untrusted input: only the listed observed variables, the three
    structural operators and numeric fixings are permitted.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise InvalidSubmission("empty model specification")
    if len(spec) > 8000:
        raise InvalidSubmission("model specification too long")
    for bad in ("import", "exec", "eval", "__", "open(", "system", ";"):
        if bad in spec:
            raise InvalidSubmission(f"forbidden token in specification: {bad!r}")

    lines = [ln.strip() for ln in spec.splitlines() if ln.strip()]
    if not lines:
        raise InvalidSubmission("no statements in specification")

    latents: set[str] = set()
    for line in lines:
        if not any(op in line for op in _ALLOWED_OPS):
            raise InvalidSubmission(f"line has no permitted operator: {line!r}")
        if "=~" in line:
            latents.add(line.split("=~")[0].strip())
    if len(latents) > max_factors:
        raise InvalidSubmission(
            f"{len(latents)} latent variables exceeds the limit of {max_factors}"
        )

    known = set(items) | latents
    for tok in _TOKEN.findall(spec):
        if tok not in known:
            raise InvalidSubmission(f"unknown variable in specification: {tok!r}")
    return spec


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def observed_in_spec(spec: str, known: list[str]) -> set:
    """The data columns a model actually refers to."""
    return {tok for tok in _TOKEN.findall(spec) if tok in set(known)}


def evaluate_model(
    spec: str, X: pd.DataFrame, items: list[str], pop: np.ndarray | None = None
) -> tuple[dict, Any]:
    """Fit a model and measure it. Returns (measures, fitted model)."""
    import semopy

    model = semopy.Model(spec)
    model.fit(X[items])
    stats = semopy.calc_stats(model)

    sigma = model.calc_sigma()[0]
    order = list(model.vars["observed"])
    idx = [order.index(it) for it in items]
    sigma = sigma[np.ix_(idx, idx)]
    scale = np.sqrt(np.diag(sigma))
    implied = sigma / np.outer(scale, scale)

    empirical = np.corrcoef(X[items].values.T)
    iu = np.triu_indices(len(items), 1)
    chi2 = float(stats["chi2"].iloc[0])
    n_par = len(model.param_vals)

    criteria = {
        "df": float(stats["DoF"].iloc[0]),
        "chi2": chi2,
        "CFI": float(stats["CFI"].iloc[0]),
        "RMSEA": float(stats["RMSEA"].iloc[0]),
        "SRMR": float(np.sqrt(((empirical[iu] - implied[iu]) ** 2).mean())),
        "BIC": float(chi2 + n_par * np.log(len(X))),
        "n_free_parameters": n_par,
    }
    if pop is not None:
        criteria["sigma_max_abs_deviation"] = float(np.abs(implied[iu] - pop[iu]).max())
        criteria["sigma_rms_deviation"] = float(np.sqrt(((implied[iu] - pop[iu]) ** 2).mean()))
    # Matched to the precision the reference criteria are stored at, so both
    # sides of every comparison are measured the same way.
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in criteria.items()}, model


# --------------------------------------------------------------------------
# Tier 1 - constraints
# --------------------------------------------------------------------------
def check_constraints(model, criteria: dict, params: dict) -> tuple[dict, list[str]]:
    ins = model.inspect(std_est=True)
    loadings = ins[ins.op == "~"]
    lstd = pd.to_numeric(loadings["Est. Std"], errors="coerce")
    variances = ins[(ins.op == "~~") & (ins.lval == ins.rval)]
    covariances = ins[(ins.op == "~~") & (ins.lval != ins.rval)]
    # In semopy both measurement loadings and structural regressions use `~`,
    # so anything observed on the right-hand side is a covariate, not a factor.
    latents = set(loadings.rval.unique()) - set(model.vars["observed"])
    factor_cov = covariances[covariances.lval.isin(latents) & covariances.rval.isin(latents)]
    std_err = pd.to_numeric(ins["Std. Err"], errors="coerce")

    phi_max = params.get("phi_max", 0.90)
    min_load = params.get("min_salient_loading", 0.30)
    min_per = params.get("min_salient_per_factor", 2)
    sign_at = params.get("sign_reversal_at", -0.10)
    max_se = params.get("max_standard_error", 10.0)

    degenerate: list[str] = []
    for factor in sorted(latents):
        sub = lstd[
            (loadings.rval.values == factor) & loadings.lval.isin(model.vars["observed"]).values
        ]
        if len(sub) and (sub.abs() >= min_load).sum() < min_per:
            degenerate.append(factor)
        if len(sub) and (sub < sign_at).any():
            degenerate.append(f"{factor}~sign_reversal")

    results = {
        "converged": True,
        "no_negative_variance": not (
            pd.to_numeric(variances["Estimate"], errors="coerce") < 0
        ).any(),
        "positive_df": criteria["df"] > 0,
        "finite_standard_errors": bool(std_err.notna().any() and std_err.dropna().lt(max_se).all()),
        "no_redundant_factor": (
            not (pd.to_numeric(factor_cov["Est. Std"], errors="coerce").abs() > phi_max).any()
            if len(factor_cov)
            else True
        ),
        "no_collapsed_factor": not degenerate,
    }
    return results, degenerate


# --------------------------------------------------------------------------
# Tier 2 - Pareto dominance over a floor
# --------------------------------------------------------------------------
def check_dominance(criteria: dict, floor: dict, spec: list[dict]) -> dict:
    out = {}
    for item in spec:
        key, direction, eps = item["key"], item["direction"], item["eps"]
        if key not in criteria or key not in floor:
            out[key] = None  # not applicable
            continue
        out[key] = (
            criteria[key] >= floor[key] - eps
            if direction == "higher"
            else criteria[key] <= floor[key] + eps
        )
    return out


# --------------------------------------------------------------------------
# Tier 3 - factual claims
# --------------------------------------------------------------------------
def _partition(mapping: dict) -> set[frozenset]:
    groups: dict[Any, set] = {}
    for item, factor in mapping.items():
        groups.setdefault(factor, set()).add(item)
    return {frozenset(g) for g in groups.values()}


def primary_assignment(model) -> dict:
    """Which factor each item belongs to.

    Read from the submitted model rather than asked for. An item that loads on
    two factors is assigned to the stronger one.
    """
    ins = model.inspect(std_est=True)
    loadings = ins[ins.op == "~"].copy()
    loadings["abs"] = pd.to_numeric(loadings["Est. Std"], errors="coerce").abs()
    strongest = loadings.loc[loadings.groupby("lval")["abs"].idxmax()]
    return {row.lval: row.rval for row in strongest.itertuples()}


def biased_items_from_fit(model, items, covariate: str = "gender") -> set:
    """Items the model says are answered differently between groups.

    The model already lets the covariate, usually gender, predict the trait. A
    direct path to an item on top of that is a claim that the item behaves
    differently for reasons the trait does not explain.
    """
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.rval == covariate) & ins.lval.isin(items)]
    return {row.lval for row in rows.itertuples()}


def _standardised(model, items):
    """A fitted model's loadings and its one factor correlation."""
    ins = model.inspect(std_est=True)
    load = ins[(ins.op == "~") & ins.lval.isin(items)]
    loadings = {row["lval"]: float(row["Est. Std"]) for _, row in load.iterrows()}
    cov = ins[(ins.op == "~~") & (ins.lval != ins.rval) & ~ins.lval.isin(items)]
    phi = float(cov["Est. Std"].iloc[0]) if len(cov) else None
    return loadings, phi


def deduplicate_and_refit(spec, items, data, key, identity):
    """Collapse rows the submitted key treats as one respondent, then refit.

    The key is the submission's claim about which columns identify a record as
    the same respondent as another. Applying it is what turns that claim into
    something measurable: too narrow a key merges people who merely answered
    alike, too broad a key merges nobody.

    spec      the submitted model, unchanged
    items     variables the model covers
    data      every delivered row, metadata included
    key       columns the submission says identify a respondent
    identity  columns that actually do, from the task definition

    Returns retained row count, repeats the key failed to remove, and the trait
    correlation on what is left. Returns None values when the key is unusable.
    """
    if not isinstance(key, (list, tuple)) or not key:
        return None, None, None
    if any(column not in data.columns for column in key):
        return None, None, None

    kept = data[~data.duplicated(subset=list(key), keep="first")]
    repeats = int(kept.duplicated(subset=list(identity), keep="first").sum())
    X = kept[items]
    X = X[(X != 0).all(axis=1)].astype(float)
    try:
        _, model = evaluate_model(spec, X, items)
    except Exception:  # noqa: BLE001
        return len(kept), repeats, None
    _, phi = _standardised(model, items)
    return len(kept), repeats, (round(phi, 3) if phi is not None else None)


def diagnose_holdouts(spec, items, train_fit, holdouts, thresholds):
    """Refit the submitted model in each holdout and say how far it travels.

    The frozen structure is re-estimated in every holdout and compared with the
    training solution. Items that no longer track their factor the same way
    break the measurement structure; a factor correlation that moves while the
    loadings hold leaves the structure intact but changes what it says.

    spec        the submitted model, unchanged
    items       variables the model covers
    train_fit   the model already fitted to the training data
    holdouts    name -> respondents, one frame per holdout
    thresholds  how far a loading and a factor correlation may move

    Returns name -> label, and name -> the measurements behind it.
    """
    base_loadings, base_phi = _standardised(train_fit, items)
    labels, detail = {}, {}
    for name, frame in holdouts.items():
        try:
            _, model = evaluate_model(spec, frame, items)
        except Exception:  # noqa: BLE001
            labels[name] = None
            continue
        loadings, phi = _standardised(model, items)
        loading_shift = max(abs(loadings[i] - base_loadings[i]) for i in base_loadings)
        phi_shift = abs(phi - base_phi) if phi is not None and base_phi is not None else 0.0
        if loading_shift > thresholds["loading"]:
            labels[name] = "measurement_structure_fails"
        elif phi_shift > thresholds["factor_correlation"]:
            labels[name] = "measurement_structure_holds_relations_differ"
        else:
            labels[name] = "generalizes"
        detail[name] = {
            "max_loading_shift": round(loading_shift, 3),
            "factor_correlation_shift": round(phi_shift, 3),
            "factor_correlation": round(phi, 3) if phi is not None else None,
        }
    return labels, detail


def latent_group_difference(model, items, covariate: str = "gender") -> float | None:
    """The largest group difference the model still puts on a trait.

    Read from the refitted model, so it reflects what the submission actually
    estimated rather than what it claimed. Returns None when the model gives
    the covariate no path to any trait, which leaves the question unanswered.
    """
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.rval == covariate) & ~ins.lval.isin(items)]
    values = pd.to_numeric(rows["Est. Std"], errors="coerce").abs().dropna()
    return float(values.max()) if len(values) else None


def factor_composition(model, items) -> dict:
    """Which items each factor covers.

    A submission can name its factors anything: they are matched to the answers
    by the items they cover, not by name.
    """
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & ins.lval.isin(items)]
    out: dict[str, set] = {}
    for row in rows.itertuples():
        out.setdefault(row.rval, set()).add(row.lval)
    return {name: frozenset(its) for name, its in out.items()}


def _match_correlations(reported, target, model, items, tol) -> bool:
    """Check reported correlations between factors, matched by items.

    reported  [[factor, factor, correlation], ...]
    target    [{"a": [items], "b": [items], "r": value}, ...]
    tol       how far a reported value may be from the answer
    """
    if not isinstance(reported, list) or not target:
        return False
    composition = factor_composition(model, items)
    want = {frozenset((frozenset(e["a"]), frozenset(e["b"]))): e["r"] for e in target}
    seen = {}
    for entry in reported:
        try:
            a, b, value = entry[0], entry[1], float(entry[2])
        except (TypeError, ValueError, IndexError):
            return False
        if a not in composition or b not in composition:
            return False
        seen[frozenset((composition[a], composition[b]))] = value
    if set(seen) != set(want):
        return False
    return all(abs(seen[k] - want[k]) <= tol for k in want)


def residual_pairs_from_fit(model, items) -> set:
    """Pairs of items the model says agree beyond the trait.

    These are the `~~` terms between two observed items: the model's claim that
    those two items agree for a reason the common factors do not explain.
    """
    ins = model.inspect(std_est=True)
    rows = ins[
        (ins.op == "~~") & (ins.lval != ins.rval) & ins.lval.isin(items) & ins.rval.isin(items)
    ]
    return {frozenset((row.lval, row.rval)) for row in rows.itertuples()}


def check_claims(
    submission: dict,
    truth: dict,
    spec: list[dict],
    n_latents: int,
    model=None,
    items: list[str] | None = None,
    derived: dict | None = None,
) -> dict:
    out: dict[str, bool | None] = {}
    for item in spec:
        key, fn = item["key"], item["fn"]
        if item.get("derive_from") == "refit_primary_loadings":
            reported = primary_assignment(model)
        elif item.get("derive_from") == "refit_residual_covariances":
            reported = residual_pairs_from_fit(model, items)
        elif item.get("derive_from") == "refit_covariate_paths":
            reported = biased_items_from_fit(model, items, item.get("covariate", "gender"))
        elif item.get("derive_from") == "refit_latent_group_difference":
            reported = latent_group_difference(model, items, item.get("covariate", "gender"))
        elif item.get("derive_from") in ("refit_holdouts", "dedup_by_key"):
            reported = (derived or {}).get(key)
        else:
            reported = submission.get(key)
        target = _dig(truth, item.get("truth_key", ""))

        if item.get("applicable_if") == "model_has_two_oblique_factors" and n_latents != 2:
            out[key] = None
            continue
        if reported is None and not item.get("derive_from"):
            out[key] = None if key == "factor_correlation" else False
            continue

        if fn == "score_vector":
            try:
                out[key] = all(
                    abs(float(reported[k]) - v) <= item["tol"] for k, v in target.items()
                )
            except (KeyError, TypeError, ValueError):
                out[key] = False
        elif fn == "score_scalar":
            try:
                out[key] = abs(float(reported) - float(target)) <= item["tol"]
            except (TypeError, ValueError):
                out[key] = False
        elif fn == "score_label_panel":

            def _flat(obj, prefix=""):
                flat = {}
                for k, v in (obj or {}).items():
                    if isinstance(v, dict):
                        flat.update(_flat(v, f"{prefix}{k}."))
                    else:
                        flat[f"{prefix}{k}"] = v
                return flat

            want, got = _flat(target), _flat(reported)
            out[key] = bool(want) and want == got
        elif fn == "score_label":
            out[key] = reported is not None and reported == target
        elif fn == "score_boolean_panel":
            try:
                out[key] = all(bool(reported[k]) == bool(v) for k, v in target.items())
            except (KeyError, TypeError):
                out[key] = False
        elif fn == "score_item_set":
            out[key] = set(reported) == set(target or [])
        elif fn == "score_pair_set":
            if item.get("match") == "composition":
                # Pairs of factors, named by the submission and matched to the
                # truth by which items each one covers.
                composition = factor_composition(model, items)
                want = {frozenset((frozenset(a), frozenset(b))) for a, b in (target or [])}
                try:
                    got = {frozenset((composition[a], composition[b])) for a, b in (reported or [])}
                except (KeyError, TypeError, ValueError):
                    got = None
            else:
                want = {frozenset(pair) for pair in (target or [])}
                try:
                    got = (
                        reported
                        if isinstance(reported, set)
                        else {frozenset(pair) for pair in (reported or [])}
                    )
                except TypeError:
                    got = None
            out[key] = got == want
        elif fn == "score_correlations_by_composition":
            out[key] = _match_correlations(reported, target, model, items, item.get("tol", 0.06))
        elif fn == "score_partition":
            out[key] = (
                _partition(reported) == _partition(target) if isinstance(reported, dict) else False
            )
        else:
            out[key] = None
    return out


def _dig(obj: dict, dotted: str):
    for part in filter(None, dotted.split(".")):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def score_model_criteria(submission: str | dict, params: dict, base_dir: str | Path = ".") -> dict:
    """Score one submission.

    submission  the agent's JSON answer, or the object already parsed
    params      the task's scoring rules
    base_dir    where the dataset and answers live

    Returns the full result. score_binary is the metric.
    """
    if params.get("task_type") == "population_classification":
        return score_population_classification(submission, params, base_dir)
    if params.get("task_type") == "gender_item_integrity":
        return score_gender_item_integrity(submission, params, base_dir)
    if params.get("task_type") == "behavioral_validity":
        return score_behavioral_validity(submission, params, base_dir)
    if params.get("task_type") == "misfit_replication":
        return score_misfit_replication(submission, params, base_dir)
    if params.get("task_type") == "adaptive_bank_choice":
        return score_adaptive_bank_choice(submission, params, base_dir)
    if params.get("task_type") == "model_identification":
        return score_model_identification(submission, params, base_dir)

    base = Path(base_dir)
    if isinstance(submission, str):
        try:
            submission = json.loads(submission.strip().strip("`").removeprefix("json"))
        except json.JSONDecodeError:
            return _zero("submission is not valid JSON")

    truth = json.loads((base / params["truth_path"]).read_text())
    ref = truth["scoring_reference"]
    floor = ref["reference_criteria"]
    pop = np.asarray(ref["population_correlation_matrix"], dtype=float)
    items = params["items"]

    known = params["syntax_whitelist"]["items"]
    data = pd.read_csv(base / params["truth_path"].rsplit("/", 1)[0] / params["dataset"], sep="\t")
    for col, val in params.get("subset", {}).items():
        data = data[data[col].isin(val)] if isinstance(val, list) else data[data[col] == val]
    X = data[[c for c in known if c in data.columns]]
    X = X[(X != 0).all(axis=1)].astype(float)
    # A dataset may carry a recording fault that the task is about. Repairing it
    # here keeps every submission judged on the same responses, whether or not
    # the submission noticed.
    for name, total in ref.get("repairs", {}).get("reverse_scored", {}).items():
        if name in X.columns:
            X[name] = total - X[name]

    try:
        spec = validate_syntax(
            submission.get("model_syntax", ""), known, params["syntax_whitelist"]["max_factors"]
        )
        # Which variables the model analyses is itself an answer when the task
        # leaves the choice open, and the later stages assume that set.
        if set(observed_in_spec(spec, known)) != set(items):
            return _result(
                0.0,
                {"instrument": "FAIL"},
                {},
                "TIER3 instrument: the model does not analyse the " "expected set of variables",
            )
        criteria, model = evaluate_model(spec, X, items, pop)
    except InvalidSubmission as exc:
        return _zero(f"invalid specification: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _zero(f"model failed to fit: {type(exc).__name__}: {exc}")

    checks: dict[str, str] = {}

    constraints, degenerate = check_constraints(
        model, criteria, _merge(params["tier_1_constraints"])
    )
    checks.update({k: _fmt(v) for k, v in constraints.items()})
    if not all(constraints.values()):
        return _result(
            0.0,
            checks,
            criteria,
            "TIER1 "
            + ",".join(k for k, v in constraints.items() if not v)
            + (f" {degenerate}" if degenerate else ""),
        )

    dominance = check_dominance(criteria, floor, params["tier_2_comparative"])
    checks.update({k: _fmt(v) for k, v in dominance.items()})
    if not all(v for v in dominance.values() if v is not None):
        return _result(
            0.0,
            checks,
            criteria,
            "TIER2 " + ",".join(k for k, v in dominance.items() if v is False),
        )

    derived = {}
    if params.get("holdout_datasets"):
        folder = base / params["truth_path"].rsplit("/", 1)[0]
        holdouts = {}
        for name, filename in params["holdout_datasets"].items():
            frame = pd.read_csv(folder / filename, sep="\t")
            frame = frame[[c for c in known if c in frame.columns]]
            holdouts[name] = frame[(frame != 0).all(axis=1)].astype(float)
        try:
            labels, detail = diagnose_holdouts(
                spec, items, model, holdouts, params["holdout_thresholds"]
            )
        except Exception as exc:  # noqa: BLE001
            return _zero(f"model failed to fit a holdout: {type(exc).__name__}: {exc}")
        derived["holdout_diagnosis"] = labels
        criteria["holdout_detail"] = detail

    if params.get("identity_columns"):
        retained, repeats, phi = deduplicate_and_refit(
            spec, items, data, submission.get("duplicate_key"), params["identity_columns"]
        )
        derived["retained_respondents"] = retained
        derived["repeats_remaining"] = repeats
        derived["corrected_factor_correlation"] = phi

    n_latents = len({ln.split("=~")[0].strip() for ln in spec.splitlines() if "=~" in ln})
    claims = check_claims(
        submission,
        truth,
        params["tier_3_claims"],
        n_latents,
        model=model,
        items=items,
        derived=derived,
    )
    checks.update({k: _fmt(v) for k, v in claims.items()})
    if not all(v for v in claims.values() if v is not None):
        return _result(
            0.0, checks, criteria, "TIER3 " + ",".join(k for k, v in claims.items() if v is False)
        )

    return _result(1.0, checks, criteria, "all tiers pass", _recorded(criteria))


def _merge(specs: list[dict]) -> dict:
    out: dict = {}
    for s in specs:
        out.update({k: v for k, v in s.items() if k != "key"})
    return out


def _fmt(v) -> str:
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def _recorded(criteria: dict) -> dict:
    """Extra numbers kept for later analysis. Never affects the score."""
    from scipy.stats import chi2 as chi2_dist

    record = {}
    if criteria.get("df", 0) > 0:
        record["chi_square_p"] = float(1 - chi2_dist.cdf(criteria["chi2"], criteria["df"]))
        record["chi_square_df"] = criteria["df"]
    return record


def _result(
    score: float, checks: dict, criteria: dict, reason: str, recorded: dict | None = None
) -> dict:
    applicable = [v for v in checks.values() if v != "n/a"]
    return {
        "score_binary": score,
        "score_partial": (
            sum(v == "PASS" for v in applicable) / len(applicable) if applicable else 0.0
        ),
        "checks_vector": checks,
        "criteria": criteria,
        "reason": reason,
        "recorded": recorded or {},
    }


def _zero(reason: str) -> dict:
    return {
        "score_binary": 0.0,
        "score_partial": 0.0,
        "checks_vector": {},
        "criteria": {},
        "reason": reason,
    }


def _outcome_coefficient(model, outcome: str, predictor: str) -> float | None:
    """Read a standardised behavioural regression from a fitted model."""
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.lval == outcome) & (ins.rval == predictor)]
    if not len(rows):
        return None
    value = pd.to_numeric(rows["Est. Std"], errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def _measurement_only(spec: str, outcome: str) -> str:
    """Keep the submitted measurement/residual structure for subgroup refits."""
    kept = []
    for line in spec.splitlines():
        stripped = line.strip()
        if not stripped or "=~" in stripped or "~~" in stripped:
            kept.append(stripped)
        elif "~" in stripped and stripped.split("~", 1)[0].strip() != outcome:
            # Structural paths for the outcome are replaced for each check;
            # other paths are not needed for the within-group measurement refit.
            continue
    return "\n".join(line for line in kept if line)


def _within_outcome_syntax(spec: str, outcome: str, group: str) -> str:
    """Keep outcome predictors but remove the between-group adjustment."""
    measurement = _measurement_only(spec, outcome)
    predictors = []
    for line in spec.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{outcome} ~"):
            predictors = [token.strip() for token in stripped.split("~", 1)[1].split("+")]
            predictors = [token for token in predictors if token and token != group]
    if not predictors:
        raise InvalidSubmission("outcome has no predictors for within-group refit")
    return f"{measurement}\n{outcome} ~ {' + '.join(predictors)}"


def _fit_outcome_model(spec: str, frame: pd.DataFrame, items: list[str], outcome: str, group: str):
    import semopy

    model = semopy.Model(spec)
    model.fit(frame[items + [group, outcome]])
    return model


def _item_outcome_effect(
    measurement: str,
    frame: pd.DataFrame,
    items: list[str],
    outcome: str,
    group: str,
    item: str,
) -> float | None:
    spec = f"{measurement}\n{outcome} ~ F + {group} + {item}"
    model = _fit_outcome_model(spec, frame, items, outcome, group)
    return _outcome_coefficient(model, outcome, item)


def score_behavioral_validity(
    submission: str | dict, params: dict, base_dir: str | Path = "."
) -> dict:
    """Score compositional evidence for a questionnaire's behavioural validity."""
    base = Path(base_dir)
    if isinstance(submission, str):
        try:
            submission = json.loads(submission.strip().strip("`").removeprefix("json"))
        except json.JSONDecodeError:
            return _zero("submission is not valid JSON")
    if not isinstance(submission, dict):
        return _zero("submission must be a JSON object")

    truth = json.loads((base / params["truth_path"]).read_text())
    target = truth["scored"]
    reported_association = submission.get("association")
    reported_items = submission.get("unstable_items")
    checks = {}
    if not isinstance(reported_association, dict):
        return _zero("missing association panel")
    if not isinstance(reported_items, list) or not all(
        isinstance(item, str) for item in reported_items
    ):
        return _zero("missing unstable_items")

    folder = base / params["truth_path"].rsplit("/", 1)[0]
    train = pd.read_csv(folder / params["dataset"], sep="\t")
    holdout = pd.read_csv(folder / params["holdout_dataset"], sep="\t")
    items = params["items"]
    outcome, group = params["outcome"], params["covariate"]
    known = items + [outcome, group]

    try:
        spec = validate_syntax(submission.get("model_syntax", ""), known, params["max_factors"])
        observed = observed_in_spec(spec, known)
        if set(known) - observed:
            return _result(0.0, {"model_coverage": "FAIL"}, {}, "model omits required variables")
        latent_names = {
            line.split("=~", 1)[0].strip() for line in spec.splitlines() if "=~" in line
        }
        if len(latent_names) != 1:
            return _result(
                0.0, {"one_latent_factor": "FAIL"}, {}, "task requires one latent factor"
            )
        latent = next(iter(latent_names))
        model = _fit_outcome_model(spec, train, items, outcome, group)
        if _outcome_coefficient(model, outcome, latent) is None:
            return _result(
                0.0, {"outcome_latent_path": "FAIL"}, {}, "model has no outcome-to-latent path"
            )
        if _outcome_coefficient(model, outcome, group) is None:
            return _result(0.0, {"group_adjustment": "FAIL"}, {}, "model has no group adjustment")

        complete_train = train[(train[items] != 0).all(axis=1)].copy()
        complete_holdout = holdout[(holdout[items] != 0).all(axis=1)].copy()
        train_model = _fit_outcome_model(spec, complete_train, items, outcome, group)
        holdout_model = _fit_outcome_model(spec, complete_holdout, items, outcome, group)
        train_effect = _outcome_coefficient(train_model, outcome, latent)
        holdout_effect = _outcome_coefficient(holdout_model, outcome, latent)
        if train_effect is None or holdout_effect is None:
            raise InvalidSubmission("could not estimate the latent outcome coefficient")

        measurement = _measurement_only(spec, outcome)
        within_syntax = _within_outcome_syntax(spec, outcome, group)
        within = {}
        for value in sorted(complete_train[group].dropna().unique()):
            subgroup = complete_train[complete_train[group] == value]
            within_model = _fit_outcome_model(within_syntax, subgroup, items, outcome, group)
            within[str(int(value))] = _outcome_coefficient(within_model, outcome, latent)
        if len(within) != 2 or any(value is None for value in within.values()):
            raise InvalidSubmission("could not estimate both within-group coefficients")

        threshold = params["min_effect"]
        tolerance = params["effect_tolerance"]
        adjusted_label = (
            "supported"
            if min(abs(train_effect), abs(holdout_effect)) >= threshold
            else "not_supported"
        )
        within_values = list(within.values())
        within_label = (
            "replicates"
            if min(abs(value) for value in within_values) >= threshold
            and np.sign(within_values[0]) == np.sign(within_values[1])
            and abs(within_values[0] - within_values[1]) <= tolerance
            else "does_not_replicate"
        )
        holdout_label = (
            "generalizes"
            if min(abs(train_effect), abs(holdout_effect)) >= threshold
            and abs(train_effect - holdout_effect) <= tolerance
            else "does_not_generalize"
        )
        derived_association = {
            "group_adjusted": adjusted_label,
            "within_group": within_label,
            "holdout": holdout_label,
        }
        checks["model_coverage"] = "PASS"
        checks["one_latent_factor"] = "PASS"
        checks["outcome_latent_path"] = "PASS"
        checks["group_adjustment"] = "PASS"
        checks["association"] = (
            "PASS"
            if reported_association == derived_association == target["association"]
            else "FAIL"
        )

        unstable = []
        for item in items:
            train_item = _item_outcome_effect(
                measurement, complete_train, items, outcome, group, item
            )
            holdout_item = _item_outcome_effect(
                measurement, complete_holdout, items, outcome, group, item
            )
            if (
                train_item is not None
                and holdout_item is not None
                and abs(train_item) >= params["item_train_min"]
                and abs(holdout_item) <= params["item_holdout_max"]
            ):
                unstable.append(item)
        checks["unstable_items"] = (
            "PASS"
            if set(reported_items) == set(unstable) == set(target["unstable_items"])
            else "FAIL"
        )
        checks["holdout_effect"] = (
            "PASS" if abs(train_effect - holdout_effect) <= tolerance else "FAIL"
        )
        recorded = {
            "train_group_adjusted": round(train_effect, 3),
            "holdout_group_adjusted": round(holdout_effect, 3),
            "within_group": {key: round(value, 3) for key, value in within.items()},
            "derived_unstable_items": unstable,
        }
    except (InvalidSubmission, Exception) as exc:  # noqa: BLE001
        checks.update(
            {
                "model_coverage": "FAIL",
                "model_fit": "FAIL",
                "association": "FAIL",
                "unstable_items": "FAIL",
            }
        )
        return _result(0.0, checks, {}, f"model failed: {type(exc).__name__}: {exc}")

    complete = all(value == "PASS" for value in checks.values())
    return _result(
        1.0 if complete else 0.0,
        checks,
        {
            "train_group_adjusted": round(train_effect, 3),
            "holdout_group_adjusted": round(holdout_effect, 3),
        },
        "all behavioural validity claims pass" if complete else "behavioural validity mismatch",
        recorded,
    )


def _measurement_item_map(spec: str) -> dict[str, set[str]]:
    """Map observed items to the latent factors named in a specification."""
    mapping: dict[str, set[str]] = {}
    for line in spec.splitlines():
        if "=~" not in line:
            continue
        factor, rhs = line.split("=~", 1)
        for item in rhs.replace("*", " ").split("+"):
            item = item.strip().split()[0] if item.strip() else ""
            if item:
                mapping.setdefault(item, set()).add(factor.strip())
    return mapping


def _model_fit_summary(spec: str, frame: pd.DataFrame, variables: list[str]) -> dict:
    import semopy

    model = semopy.Model(spec)
    model.fit(frame[variables])
    stats = semopy.calc_stats(model).iloc[0]
    return {
        "model": model,
        "CFI": float(stats["CFI"]),
        "RMSEA": float(stats["RMSEA"]),
    }


def _drop_line(spec: str, line: str) -> str:
    """The same model without one line."""
    return "\n".join(ln for ln in spec.splitlines() if ln.strip() != line.strip())


def _drop_residual_pair(spec: str, pair: list[str] | tuple[str, str]) -> str:
    """Remove an observed-item residual covariance regardless of order."""
    wanted = set(pair)
    kept = []
    for line in spec.splitlines():
        if "~~" in line:
            left, right = (part.strip() for part in line.split("~~", 1))
            if {left, right} == wanted:
                continue
        kept.append(line)
    return "\n".join(kept)


def _drop_cross_loading(spec: str, item: str, item_map: dict, model=None) -> str:
    """The same model with `item` left on only one factor."""
    factors = item_map.get(item, set())
    keep = sorted(factors)[:1]
    if model is not None:
        ins = model.inspect(std_est=True)
        rows = ins[(ins.op == "~") & (ins.lval == item) & ins.rval.isin(factors)].copy()
        if len(rows):
            rows["abs_std"] = pd.to_numeric(rows["Est. Std"], errors="coerce").abs()
            keep = [str(rows.loc[rows["abs_std"].idxmax(), "rval"])]
    out = []
    for line in spec.splitlines():
        if "=~" in line:
            factor, rhs = line.split("=~", 1)
            factor = factor.strip()
            members = [m.strip() for m in rhs.replace("+", " ").split()]
            if item in members and factor not in keep:
                members = [m for m in members if m != item]
                if not members:
                    continue
                line = f"{factor} =~ {'+'.join(members)}"
        out.append(line)
    return "\n".join(out)


def _modification_worth(spec, reduced, frame, variables):
    """CFI the modification buys in this sample, or None if either fit fails."""
    try:
        full = _model_fit_summary(spec, frame, variables)["CFI"]
        less = _model_fit_summary(reduced, frame, variables)["CFI"]
    except Exception:  # noqa: BLE001
        return None
    return round(full - less, 4)


def score_misfit_replication(
    submission: str | dict, params: dict, base_dir: str | Path = "."
) -> dict:
    """Score item-level modifications against an independent replication."""
    base = Path(base_dir)
    if isinstance(submission, str):
        try:
            submission = json.loads(submission.strip().strip("`").removeprefix("json"))
        except json.JSONDecodeError:
            return _zero("submission is not valid JSON")
    if not isinstance(submission, dict):
        return _zero("submission must be a JSON object")

    truth = json.loads((base / params["truth_path"]).read_text())
    target = truth["scored"]
    findings = submission.get("findings")
    replication = submission.get("replication")
    if not isinstance(findings, dict) or not isinstance(replication, dict):
        return _zero("missing findings or replication conclusions")

    folder = base / params["truth_path"].rsplit("/", 1)[0]
    development = pd.read_csv(folder / params["dataset"], sep="\t")
    replicate = pd.read_csv(folder / params["replication_dataset"], sep="\t")
    items, group = params["items"], params["covariate"]
    known = items + [group]
    checks = {}
    try:
        spec = validate_syntax(submission.get("model_syntax", ""), known, params["max_factors"])
        reported_poor = findings.get("poor_items", [])
        if not isinstance(reported_poor, list):
            raise InvalidSubmission("poor_items must be a list")
        expected_items = set(items) - set(reported_poor)
        observed = observed_in_spec(spec, known)
        model_items = observed & set(items)
        if model_items != expected_items:
            checks["item_coverage"] = "FAIL"
        else:
            checks["item_coverage"] = "PASS"

        dev = development[(development[list(model_items)] != 0).all(axis=1)]
        rep = replicate[(replicate[list(model_items)] != 0).all(axis=1)]
        variables = [item for item in items if item in model_items] + [group]
        fitted_dev = _model_fit_summary(spec, dev, variables)
        fitted_rep = _model_fit_summary(spec, rep, variables)
        factor_names = {
            line.split("=~", 1)[0].strip() for line in spec.splitlines() if "=~" in line
        }
        estimates = fitted_dev["model"].inspect(std_est=True)
        factor_group_paths = set(
            estimates.loc[
                (estimates.op == "~")
                & (estimates.rval == group)
                & estimates.lval.isin(factor_names),
                "lval",
            ]
        )
        checks["factor_group_paths"] = "PASS" if factor_group_paths == factor_names else "FAIL"

        item_map = _measurement_item_map(spec)
        cross = sorted(item for item in model_items if len(item_map.get(item, set())) > 1)
        residual = residual_pairs_from_fit(fitted_dev["model"], list(model_items))
        residual_list = sorted([sorted(pair) for pair in residual])
        dif = sorted(biased_items_from_fit(fitted_dev["model"], list(model_items), group))
        derived_findings = {
            "local_dependence": residual_list,
            "cross_loadings": cross,
            "dif_items": dif,
            "poor_items": sorted(reported_poor),
        }
        normalized_reported = dict(findings)
        normalized_target = dict(target["findings"])
        for panel in ("local_dependence",):
            normalized_reported[panel] = sorted(sorted(pair) for pair in findings.get(panel, []))
            normalized_target[panel] = sorted(
                sorted(pair) for pair in target["findings"].get(panel, [])
            )
        checks["findings"] = (
            "PASS" if derived_findings == normalized_target == normalized_reported else "FAIL"
        )
        checks["replication_fit"] = (
            "PASS" if fitted_rep["CFI"] >= 0.95 and fitted_rep["RMSEA"] <= 0.08 else "FAIL"
        )
        checks["model_fit"] = "PASS"

        # Each retained modification is taken back out and the model refitted in
        # the replication sample. What it is worth there is the verdict.
        margin = params.get("replication_gain", 0.010)
        ordered = [item for item in items if item in model_items]
        reduced_for = {
            "local_dependence": [_drop_residual_pair(spec, (a, b)) for a, b in residual_list],
            "cross_loadings": [
                _drop_cross_loading(spec, item, item_map, fitted_dev["model"]) for item in cross
            ],
            "dif_items": [_drop_line(spec, f"{item} ~ {group}") for item in dif],
        }
        derived_replication = {}
        worth = {}
        for panel, variants in reduced_for.items():
            values = [_modification_worth(spec, reduced, rep, variables) for reduced in variants]
            worth[panel] = values
            derived_replication[panel] = (
                "replicates"
                if values and all(v is not None and v >= margin for v in values)
                else "does_not_replicate"
            )
        # A poor item replicates as poor when it is still weak in the other sample.
        weak_cut = params.get("weak_loading", 0.30)
        if reported_poor:
            with_weak = spec.replace("F1 =~ ", "F1 =~ " + "+".join(reported_poor) + "+", 1)
            try:
                restored = _model_fit_summary(with_weak, rep, ordered + reported_poor + [group])
                estimated, _ = _standardised(restored["model"], ordered + reported_poor)
                weak = [abs(estimated[item]) for item in reported_poor if item in estimated]
                derived_replication["poor_items"] = (
                    "replicates"
                    if len(weak) == len(reported_poor) and max(weak) < weak_cut
                    else "does_not_replicate"
                )
            except Exception:  # noqa: BLE001
                derived_replication["poor_items"] = "does_not_replicate"
        else:
            derived_replication["poor_items"] = "does_not_replicate"

        target_replication = dict(target["replication"])
        rejected_target = sorted(
            sorted(pair)
            for pair in target_replication.pop("rejected_development_modifications", [])
        )
        rejected_reported = sorted(
            sorted(pair) for pair in replication.get("rejected_development_modifications", [])
        )
        reported_verdicts = {
            key: value
            for key, value in replication.items()
            if key != "rejected_development_modifications"
        }
        checks["replication_verdicts"] = (
            "PASS" if derived_replication == target_replication == reported_verdicts else "FAIL"
        )

        # A rejected modification has to be one the development sample really
        # supports and the replication sample really does not.
        justified = True
        for pair in rejected_reported:
            added = spec + f"\n{pair[0]} ~~ {pair[1]}"
            in_dev = _modification_worth(added, spec, dev, variables)
            in_rep = _modification_worth(added, spec, rep, variables)
            if in_dev is None or in_rep is None or in_dev < margin or in_rep >= margin:
                justified = False
        checks["rejected_modifications"] = (
            "PASS" if rejected_reported == rejected_target and justified else "FAIL"
        )

        recorded = {
            "development_CFI": round(fitted_dev["CFI"], 3),
            "development_RMSEA": round(fitted_dev["RMSEA"], 3),
            "replication_CFI": round(fitted_rep["CFI"], 3),
            "replication_RMSEA": round(fitted_rep["RMSEA"], 3),
            "derived_findings": derived_findings,
            "replication_worth": worth,
        }
    except (InvalidSubmission, Exception) as exc:  # noqa: BLE001
        checks.update(
            {
                "model_fit": "FAIL",
                "findings": "FAIL",
                "replication_verdicts": "FAIL",
                "rejected_modifications": "FAIL",
            }
        )
        return _result(0.0, checks, {}, f"model failed: {type(exc).__name__}: {exc}")

    complete = all(value == "PASS" for value in checks.values())
    return _result(
        1.0 if complete else 0.0,
        checks,
        {"n_items": len(model_items)},
        "all replicated modification claims pass" if complete else "misfit replication mismatch",
        recorded,
    )


def score_adaptive_bank_choice(
    submission: str | dict, params: dict, base_dir: str | Path = "."
) -> dict:
    """Score an item-bank decision for adaptive testing.

    The answer is what the responses say about the items: which pairs stay
    correlated once ability is accounted for, which vendor flags they do not
    support, and which items do not behave the same way in the holdout sample.
    No model or working is submitted.
    """
    base = Path(base_dir)
    if isinstance(submission, str):
        try:
            submission = json.loads(submission.strip().strip("`").removeprefix("json"))
        except json.JSONDecodeError:
            return _zero("submission is not valid JSON")
    if not isinstance(submission, dict):
        return _zero("submission must be a JSON object")

    truth = json.loads((base / params["truth_path"]).read_text())
    target = truth["scored"]
    checks = {}

    def _pairs(value):
        try:
            return sorted(sorted(str(x) for x in pair) for pair in value or [])
        except TypeError:
            return None

    def _names(value):
        try:
            return sorted(str(x) for x in value or [])
        except TypeError:
            return None

    checks["bank_choice"] = (
        "PASS" if submission.get("bank_choice") == target["bank_choice"] else "FAIL"
    )
    checks["dependent_pairs"] = (
        "PASS"
        if _pairs(submission.get("dependent_pairs")) == _pairs(target["dependent_pairs"])
        else "FAIL"
    )
    checks["unsupported_vendor_flags"] = (
        "PASS"
        if _names(submission.get("unsupported_vendor_flags"))
        == _names(target["unsupported_vendor_flags"])
        else "FAIL"
    )
    checks["unstable_items"] = (
        "PASS"
        if _names(submission.get("unstable_items")) == _names(target["unstable_items"])
        else "FAIL"
    )
    checks["recommendation"] = (
        "PASS" if submission.get("recommendation") == target["recommendation"] else "FAIL"
    )

    measured = truth.get("measured", {})
    complete = all(value == "PASS" for value in checks.values())
    return _result(
        1.0 if complete else 0.0,
        checks,
        {
            "vendor_bank": measured.get("vendor_bank", {}),
            "screened_bank": measured.get("screened_bank", {}),
        },
        "adaptive bank decision passes" if complete else "adaptive bank decision mismatch",
    )


def score_model_identification(
    submission: str | dict, params: dict, base_dir: str | Path = "."
) -> dict:
    """Score a complete mapping from anonymous datasets to candidate models."""
    base = Path(base_dir)
    if isinstance(submission, str):
        try:
            submission = json.loads(submission.strip().strip("`").removeprefix("json"))
        except json.JSONDecodeError:
            return _zero("submission is not valid JSON")
    if not isinstance(submission, dict) or not isinstance(submission.get("assignments"), dict):
        return _zero("submission must contain an assignments object")
    truth = json.loads((base / params["truth_path"]).read_text())["scored"]["assignments"]
    assignments = submission["assignments"]
    checks = {}
    for dataset in params["datasets"]:
        value = assignments.get(dataset)
        checks[dataset] = (
            "PASS"
            if isinstance(value, list) and sorted(value) == sorted(truth[dataset])
            else "FAIL"
        )
    checks["complete"] = "PASS" if set(assignments) == set(params["datasets"]) else "FAIL"
    complete = all(value == "PASS" for value in checks.values())
    return _result(
        1.0 if complete else 0.0,
        checks,
        {},
        "model assignments pass" if complete else "model assignments mismatch",
    )


def score_population_classification(
    submission: str | dict, params: dict, base_dir: str | Path = "."
) -> dict:
    """Score a process-agnostic classification of anonymised respondents.

    These tasks deliberately do not require the agent to submit its fitted
    models.  The observable answer is the set of populations compatible with
    each person; the hidden answer may contain one or several populations.
    Candidate order is irrelevant, but missing, extra, or unknown people are
    errors.
    """
    base = Path(base_dir)
    if isinstance(submission, str):
        try:
            submission = json.loads(submission.strip().strip("`").removeprefix("json"))
        except json.JSONDecodeError:
            return _zero("submission is not valid JSON")
    if not isinstance(submission, dict):
        return _zero("submission must be a JSON object")

    truth = json.loads((base / params["truth_path"]).read_text())
    target = truth["scored"]["classifications"]
    reported = submission.get("classifications")
    if not isinstance(reported, dict):
        return _zero("missing classifications")

    allowed = set(params["populations"])
    checks = {}
    for person, want in target.items():
        got = reported.get(person)
        valid = (
            isinstance(got, list)
            and len(got) == len(set(got))
            and all(isinstance(name, str) and name in allowed for name in got)
        )
        checks[person] = "PASS" if valid and set(got) == set(want) else "FAIL"
    extra = set(reported) - set(target)
    if extra:
        checks["unexpected_people"] = "FAIL"
    applicable = list(checks.values())
    passed = sum(value == "PASS" for value in applicable)
    complete = passed == len(target) and not extra
    return _result(
        1.0 if complete else 0.0,
        checks,
        {"n_people": len(target), "n_exact": passed},
        "all classifications pass" if complete else "classification mismatch",
    )


def score_gender_item_integrity(
    submission: str | dict, params: dict, base_dir: str | Path = "."
) -> dict:
    """Score item-integrity diagnoses made before a gender comparison.

    The item diagnoses and final comparison are the observable conclusions.
    The submitted model is additionally checked for coverage, a gender path,
    and successful fitting, but the scorer does not require a particular
    analysis workflow.
    """
    base = Path(base_dir)
    if isinstance(submission, str):
        try:
            submission = json.loads(submission.strip().strip("`").removeprefix("json"))
        except json.JSONDecodeError:
            return _zero("submission is not valid JSON")
    if not isinstance(submission, dict):
        return _zero("submission must be a JSON object")

    truth = json.loads((base / params["truth_path"]).read_text())
    target_items = truth["scored"]["item_diagnoses"]
    target_comparison = truth["scored"]["comparison"]
    reported_items = submission.get("item_diagnoses")
    reported_comparison = submission.get("comparison")
    checks = {}

    if isinstance(reported_items, dict):
        checks.update(
            {
                item: "PASS" if reported_items.get(item) == label else "FAIL"
                for item, label in target_items.items()
            }
        )
        checks["item_set"] = "PASS" if set(reported_items) == set(target_items) else "FAIL"
    else:
        checks.update({item: "FAIL" for item in target_items})
        checks["item_set"] = "FAIL"
    checks["comparison"] = "PASS" if reported_comparison == target_comparison else "FAIL"

    known = params["items"] + [params["covariate"]]
    try:
        spec = validate_syntax(submission.get("model_syntax", ""), known, params["max_factors"])
        observed = observed_in_spec(spec, known)
        if set(params["items"]) - observed:
            checks["model_coverage"] = "FAIL"
        else:
            checks["model_coverage"] = "PASS"
        has_covariate_path = any(
            "~" in line and params["covariate"] in line.split("~", 1)[1].split()
            for line in spec.splitlines()
        )
        checks["gender_path"] = "PASS" if has_covariate_path else "FAIL"
        folder = base / params["truth_path"].rsplit("/", 1)[0]
        items, covariate = params["items"], params["covariate"]
        frame = pd.read_csv(folder / params["dataset"], sep="\t")
        X = frame[(frame[items] != 0).all(axis=1)][items + [covariate]].astype(float).copy()

        # The submission's own diagnoses decide what gets repaired, so a wrong
        # diagnosis is carried into the model the comparison is read from.
        total = truth.get("repairs", {}).get("reverse_scored", {})
        for item, label in (reported_items or {}).items():
            if label == "mis_keyed" and item in X.columns:
                X[item] = total.get(item, 6) - X[item]

        _, model = evaluate_model(spec, X, items + [covariate])
        checks["model_fit"] = "PASS"

        # Which items the model lets differ between the groups, and what group
        # difference it leaves on the trait once they do.
        freed = biased_items_from_fit(model, items, covariate)
        checks["group_dependent_items"] = (
            "PASS" if freed == set(truth["scored"]["group_dependent_items"]) else "FAIL"
        )
        effect = latent_group_difference(model, items + [covariate], covariate)
        target_effect = truth["scored"]["repaired_gender_effect"]
        within = effect is not None and abs(effect - target_effect) <= params.get(
            "effect_tolerance", 0.05
        )
        checks["repaired_gender_effect"] = "PASS" if within else "FAIL"
        # The comparison label is only granted when the submitted model supports it.
        if reported_comparison == "reportable_after_repair" and not within:
            checks["comparison"] = "FAIL"
    except (InvalidSubmission, Exception) as exc:  # noqa: BLE001
        checks["model_coverage"] = "FAIL"
        checks["gender_path"] = "FAIL"
        checks["model_fit"] = "FAIL"
        checks["group_dependent_items"] = "FAIL"
        checks["repaired_gender_effect"] = "FAIL"
        return _result(0.0, checks, {}, f"model failed: {type(exc).__name__}: {exc}")

    applicable = list(checks.values())
    complete = all(value == "PASS" for value in applicable)
    return _result(
        1.0 if complete else 0.0,
        checks,
        {"n_items": len(target_items)},
        "all claims and model checks pass" if complete else "item-integrity comparison mismatch",
    )
