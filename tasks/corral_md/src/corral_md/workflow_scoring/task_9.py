"""Audit correlated Cu regression from immutable saved numerical evidence only."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from ase import units

from .common import (
    EvidenceError,
    UnsupportedEvidence,
    close,
    finite_array,
    is_teacher_model,
    optional_results_match,
    result_close,
    scientific_screen,
)
from .regression import (
    metrics,
    partition,
    predict,
    preprocessing_matches,
    ridge_stationarity,
)

RUNS = ("main", "independent_800", "independent_500", "independent_1100")
TARGETS = dict(zip(RUNS, (800, 800, 500, 1100), strict=False))
STRATEGIES = ("randomized", "time_aware")


class Unsupported(UnsupportedEvidence):
    """A valid alternative needs a different independent numerical adapter."""


def _supported(fn):
    def check():
        try:
            return fn()
        except Unsupported as exc:
            return None, str(exc)

    return check


def _number(value):
    return float(finite_array(value, shape=()))


def _indices(values, count, repeats=False):
    raw = finite_array(values, ndim=1)
    ix = raw.astype(int)
    if not np.array_equal(raw, ix) or np.any(ix < 0) or np.any(ix >= count):
        raise EvidenceError("Invalid index")
    if not repeats and len(np.unique(ix)) != len(ix):
        raise EvidenceError("Repeated index")
    return ix


def _time(a):
    return _number(
        a.info["time_fs"] if "time_fs" in a.info else 1000 * a.info["time_ps"]
    )


def _energy(a):
    cached = getattr(getattr(a, "calc", None), "results", {})
    for source, key in ((cached, "energy"), (a.info, "potential_energy_eV")):
        if key in source:
            return _number(source[key])
    raise EvidenceError("Stored energy absent; calculator calls are prohibited")


def _temperature(a, dof):
    if "momenta" not in a.arrays:
        raise EvidenceError("Saved momenta required")
    p = finite_array(a.arrays["momenta"], shape=(32, 3))
    return float(np.sum(p**2 / a.get_masses()[:, None]) / (dof * units.kB))


def _same(a, b):
    # Compare physical geometry modulo a global translation and periodic wrapping.
    if not np.array_equal(a.numbers, b.numbers) or not close(
        a.cell.array, b.cell.array
    ):
        return False
    delta = (a.positions - b.positions) @ np.linalg.inv(a.cell.array)
    delta -= delta[0]
    delta -= np.rint(delta)
    return close(delta, np.zeros_like(delta), atol=1e-7)


def _state_same(a, b):
    return (
        _same(a, b)
        and close(a.arrays["momenta"], b.arrays["momenta"], atol=1e-7)
        and close(_time(a), _time(b), atol=1e-7)
        and close(_energy(a), _energy(b), atol=1e-7)
    )


def _metric_match(reference, prediction, reported):
    calculated = metrics(reference, prediction)
    return all(result_close(calculated[k], reported[k], atol=1e-07) for k in calculated)


def _model_ok(x, y, model):
    if str(model.get("estimator", "")).lower() not in {
        "ridge",
        "sklearn.linear_model.ridge",
    }:
        raise EvidenceError("The task requires a declared Ridge estimator")
    if type(model.get("positive", False)) is not bool:
        raise EvidenceError("Ridge positive must be a boolean")
    return preprocessing_matches(x, model) and ridge_stationarity(x, y, model)


def evaluate(e, r):
    @lru_cache(None)
    def frames(run, stage="trajectory"):
        result = e.trajectory(f"runs.{run}.{stage}")
        n = 200 if stage == "trajectory" else None
        if (
            (n is not None and len(result) != n)
            or (stage == "equilibration" and len(result) < 2)
            or (stage == "boundary" and len(result) not in (1, 2))
        ):
            raise EvidenceError(
                "Expected 200 production frames, one shared or two boundary states, "
                "or >=2 equilibration states"
            )
        cell = finite_array(result[0].cell.array, shape=(3, 3))
        if abs(np.linalg.det(cell)) <= 1e-10:
            raise EvidenceError("Nondegenerate periodic cell required")
        for a in result:
            if (
                len(a) != 32
                or set(a.get_chemical_symbols()) != {"Cu"}
                or not np.all(a.pbc)
            ):
                raise EvidenceError("Expected periodic Cu32")
            finite_array(a.positions, shape=(32, 3))
            if not close(a.cell.array, cell, rtol=1e-6, atol=1e-7):
                raise EvidenceError("Cell changed within a fixed-cell run")
            if not np.allclose(a.get_masses(), 63.546, rtol=0, atol=1e-3):
                raise EvidenceError("Unexpected copper masses")
            finite_array(a.arrays["momenta"], shape=(32, 3))
            _energy(a)
            if stage == "trajectory":
                finite_array(getattr(a.calc, "results", {})["forces"], shape=(32, 3))
        times = finite_array([_time(a) for a in result], ndim=1)
        if stage != "boundary" and not np.all(np.diff(times) > 0):
            raise EvidenceError("Saved states must be chronological in physical time")
        return result

    @lru_cache(None)
    def xy(run):
        x = finite_array(e.array(f"runs.{run}.descriptors"), ndim=2)
        y = finite_array(e.array(f"runs.{run}.labels"), shape=(200,))
        if x.shape[0] != 200:
            raise EvidenceError("One global descriptor per retained frame is required")
        return x, y

    @lru_cache(None)
    def models():
        return e.json("models", "pipeline")

    def geometry():
        supplied = e.trajectory("input_structure")[0]
        if (
            len(supplied) != 32
            or set(supplied.get_chemical_symbols()) != {"Cu"}
            or not np.all(supplied.pbc)
        ):
            return False
        for run in RUNS:
            eq, prod = frames(run, "equilibration"), frames(run)
            if not _same(eq[0], supplied) or not close(
                prod[0].cell.array, supplied.cell.array
            ):
                return False
        return True

    r.check("fixed_cell_cu32_and_chronological_frames", 4, geometry)

    def initialization():
        seeds, momenta = [], []
        for run in RUNS:
            setting = e.settings["runs"][run]
            dof = setting["temperature_dof"]
            if (
                dof not in (93, 96)
                or setting["target_temperature_K"] != TARGETS[run]
                or type(setting["seed"]) is not int
                or setting["remove_com"] is not True
                or _number(setting["timestep_fs"]) <= 0
                or not setting["integrator"]
                or not setting["thermostat"]
            ):
                return False
            initial = frames(run, "equilibration")[0]
            p = initial.arrays["momenta"]
            target = TARGETS[run]
            if np.max(np.abs(p.sum(axis=0))) > 1e-7 or abs(
                _temperature(initial, dof) - target
            ) > 6 * target * np.sqrt(2 / dof):
                return False
            seeds.append(setting["seed"])
            momenta.append(p)
        return len(set(seeds)) == 4 and all(
            not np.array_equal(momenta[i], momenta[j])
            for i in range(4)
            for j in range(i)
        )

    r.check("distinct_initializations_and_recorded_md_settings", 3, initialization)

    def production_targets():
        compatible = []
        for run in RUNS:
            dof = e.settings["runs"][run]["temperature_dof"]
            if dof not in (93, 96):
                raise EvidenceError("Temperature degrees of freedom must be 93 or 96")
            target = TARGETS[run]
            mean = np.mean([_temperature(a, dof) for a in frames(run)])
            compatible.append(abs(mean - target) <= 6 * target * np.sqrt(2 / dof))
        return scientific_screen(
            all(compatible), "Production temperature target evidence is inconclusive"
        )

    r.check("production_temperature_targets", 1, production_targets)

    def boundaries():
        for run in RUNS:
            saved = frames(run, "boundary")
            end, start = (saved[0], saved[0]) if len(saved) == 1 else saved
            eq, prod = frames(run, "equilibration"), frames(run)
            if not _state_same(end, start) or not _state_same(eq[-1], end):
                return False
            if not close(end.cell.array, prod[0].cell.array) or _time(prod[0]) < _time(
                start
            ):
                return False
            if close(_time(prod[0]), _time(start)) and not _state_same(prod[0], start):
                return False
        return True

    r.check("equilibration_to_production_boundaries", 4, boundaries)

    def aligned_data():
        widths = []
        for run in RUNS:
            x, y = xy(run)
            if not close(y, [_energy(a) for a in frames(run)], atol=1e-7):
                return False
            widths.append(x.shape[1])
        soap = e.settings["soap"]
        return (
            len(set(widths)) == 1
            and soap["species"] == ["Cu"]
            and soap["periodic"] is True
            and soap["average"] in ("inner", "outer")
            and _number(soap["r_cut"]) > 0
            and int(soap["n_max"]) == soap["n_max"]
            and soap["n_max"] > 0
            and int(soap["l_max"]) == soap["l_max"]
            and soap["l_max"] >= 0
            and is_teacher_model(e.settings["teacher"])
        )

    r.check("energy_descriptor_alignment_and_teacher_configuration", 4, aligned_data)

    @lru_cache(None)
    def temperatures(run):
        dof = e.settings["runs"][run]["temperature_dof"]
        if dof not in (93, 96):
            raise EvidenceError("Temperature degrees of freedom must be 93 or 96")
        eq, prod = frames(run, "equilibration"), frames(run)
        eq_t = np.array([_temperature(a, dof) for a in eq])
        prod_t = np.array([_temperature(a, dof) for a in prod])
        k = len(eq_t) // 2
        return {
            "production_temperature_mean_K": float(prod_t.mean()),
            "production_temperature_std_K": float(prod_t.std()),
            "equilibration_first_half_temperature_K": float(eq_t[:k].mean()),
            "equilibration_second_half_temperature_K": float(eq_t[k:].mean()),
            "equilibration_temperature_drift_K_per_ps": float(
                np.polyfit(np.array([_time(a) for a in eq]) / 1000, eq_t, 1)[0]
            ),
        }

    def thermal_check():
        for run in RUNS:
            reported = e.results["runs"][run]
            expected = temperatures(run)
            if not result_close(
                expected["production_temperature_mean_K"],
                reported["production_temperature_mean_K"],
                atol=1e-7,
            ) or not optional_results_match(reported, expected, atol=1e-7):
                return False
            table = e.table(f"runs.{run}.log")
            times = finite_array(table["time_fs"], ndim=1)
            for a in frames(run, "equilibration") + frames(run):
                ix = np.flatnonzero(np.isclose(times, _time(a), rtol=0, atol=1e-7))
                if not len(ix):
                    return False
                row = table.iloc[ix[-1]]
                if not close(
                    row["temperature_K"],
                    _temperature(a, e.settings["runs"][run]["temperature_dof"]),
                    atol=1e-7,
                ) or not close(row["potential_energy_eV"], _energy(a), atol=1e-7):
                    return False
        return True

    r.check("kinetic_temperature_equilibration_and_logs", 4, thermal_check)

    @lru_cache(None)
    def correlation():
        doc = e.json("correlation")
        if doc["estimator"] != "acf":
            raise Unsupported(
                "Automatic temporal estimator currently supports biased/unbiased ACF"
            )
        t = np.array([_time(a) for a in frames("main")])
        dt = np.diff(t)
        convention = doc.get("lag_convention", "uniform_samples")
        if not isinstance(convention, str) or not convention.strip():
            raise EvidenceError("Record the ACF lag convention")
        if convention.strip().lower() != "uniform_samples":
            times = finite_array(doc["lag_times"], ndim=1)
            finite_array(doc["values"], shape=times.shape)
            if len(times) < 2 or np.any(np.diff(times) <= 0):
                raise EvidenceError("Correlation lag times must increase")
            if doc["time_unit"] not in {"fs", "ps"}:
                raise EvidenceError("Correlation time units must be fs or ps")
            raise Unsupported(
                "The documented ACF lag convention requires independent review"
            )
        if not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-7):
            raise EvidenceError(
                "ACF lag-to-time conversion requires uniform retained sampling"
            )
        lags = _indices(doc["lag_indices"], 200)
        if len(lags) < 2 or lags[0] != 0 or np.any(np.diff(lags) <= 0):
            raise EvidenceError("ACF lags must increase from zero")
        y = xy("main")[1]
        y = y - y.mean()
        variance_sum = np.dot(y, y)
        if variance_sum <= 0:
            raise EvidenceError("Constant energy provides no defined normalized ACF")
        acf = np.array([np.dot(y[: len(y) - k], y[k:]) / variance_sum for k in lags])
        if doc["normalization"] == "unbiased":
            acf *= len(y) / (len(y) - lags)
        elif doc["normalization"] != "biased":
            raise Unsupported("ACF normalization must be biased or unbiased")
        factor = {"fs": 1, "ps": 1000}.get(doc["time_unit"])
        if factor is None:
            raise EvidenceError("Correlation time units must be fs or ps")
        valid = result_close(doc["values"], acf, atol=1e-07) and result_close(
            doc["lag_times"], lags * dt[0] / factor, atol=1e-07
        )
        return valid, doc, lags, acf, dt[0], t[-1] - t[0]

    r.check(
        "temporal_dependence_curve_and_units", 8, _supported(lambda: correlation()[0])
    )

    def characteristic():
        valid, doc, lags, acf, dt, duration = correlation()
        if not valid:
            return False
        item = doc["characteristic"]
        factor = {"fs": 1, "ps": 1000}.get(item["time_unit"])
        if (
            factor is None
            or not result_close(item["sampling_interval"], dt / factor, atol=1e-07)
            or not result_close(
                item["observation_duration"], duration / factor, atol=1e-07
            )
        ):
            return False
        if item["method"] == "first_crossing":
            if not np.array_equal(lags, np.arange(len(lags))):
                raise EvidenceError("A first crossing requires all earlier lags")
            threshold = _number(item["threshold"])
            if not 0 < threshold < 1:
                return False
            indices = np.flatnonzero(acf <= threshold)
            value, bound = (
                (lags[indices[0]] * dt, "estimate")
                if len(indices)
                else (lags[-1] * dt, "lower")
            )
        elif item["method"] == "integrated":
            cutoff = int(item["cutoff_lag"])
            if (
                cutoff < 1
                or item["cutoff_lag"] != cutoff
                or not np.array_equal(lags[: cutoff + 1], np.arange(cutoff + 1))
            ):
                return False
            if item["convention"] == "half_plus_sum":
                value = dt * (0.5 + acf[1 : cutoff + 1].sum())
            elif item["convention"] == "one_plus_twice_sum":
                value = dt * (1 + 2 * acf[1 : cutoff + 1].sum())
            else:
                raise Unsupported("Unsupported integrated-correlation convention")
            bound = "estimate"
        else:
            raise Unsupported("Unsupported characteristic-time estimator")
        return item["bound"] == bound and result_close(
            item["value"], value / factor, atol=1e-07
        )

    r.check("characteristic_correlation_time_or_bound", 7, _supported(characteristic))

    @lru_cache(None)
    def outer(which):
        item = models()["validation"][which]
        train, test = partition(item["train_indices"], item["test_indices"], 200)
        if not len(train) or not len(test):
            raise EvidenceError("Validation partitions must be nonempty")
        return item, train, test

    def split_check():
        random, _, random_test = outer("randomized")
        if type(random["seed"]) is not int:
            return False
        scheme = random.get("sampling", "numpy_default_rng_choice")
        seed = random["seed"]
        if scheme == "numpy_default_rng_choice":
            sampled = np.random.default_rng(seed).choice(
                200, len(random_test), replace=False
            )
        elif scheme == "numpy_default_rng_permutation":
            sampled = np.random.default_rng(seed).permutation(200)[: len(random_test)]
        elif scheme == "numpy_randomstate_permutation":
            sampled = np.random.RandomState(seed).permutation(200)[: len(random_test)]
        else:
            raise Unsupported("Unsupported saved randomization engine")
        if not np.array_equal(np.sort(sampled), np.sort(random_test)):
            return False
        item, train, test = outer("time_aware")
        strategy = item["strategy"]
        if strategy not in {"blocked", "forward", "purged"}:
            raise Unsupported(
                "Supported time-aware layouts are blocked, forward and purged"
            )
        if strategy == "blocked" and not np.all(np.diff(np.sort(test)) == 1):
            return False
        if strategy == "forward" and not train.max() < test.min():
            return False
        distance = int(np.min(np.abs(train[:, None] - test[None, :]))) - 1
        times = np.array([_time(a) for a in frames("main")])
        gap_time = float(np.min(np.abs(times[train, None] - times[test][None, :])))
        return (
            ("minimum_gap_frames" not in item or item["minimum_gap_frames"] == distance)
            and (
                "minimum_gap_time_fs" not in item
                or close(item["minimum_gap_time_fs"], gap_time, atol=1e-7)
            )
            and (strategy != "purged" or distance > 0)
        )

    r.check("disjoint_randomized_and_time_aware_holdouts", 5, _supported(split_check))

    def outer_fit():
        x, y = xy("main")
        return all(
            _model_ok(x[train], y[train], item["model"])
            for item, train, _ in (outer(s) for s in STRATEGIES)
        )

    r.check(
        "training_only_outer_preprocessing_and_ridge_consistency",
        5,
        _supported(outer_fit),
    )

    def alternative_tuning(item, outer_train):
        method = item.get("tuning_method", "recorded_inner_splits")
        if not isinstance(method, str) or not method.strip():
            raise EvidenceError("Record the tuning method")
        if method == "recorded_inner_splits":
            return
        evidence = item["tuning"]
        rows = evidence["selection_indices"]
        indices = (
            np.empty(0, dtype=int)
            if method == "predeclared_fixed_alpha"
            and isinstance(rows, list)
            and not rows
            else _indices(rows, 200)
        )
        if not np.isin(indices, outer_train).all():
            raise EvidenceError(
                "Alternative tuning accessed an outer evaluation holdout"
            )
        if not len(indices) and method != "predeclared_fixed_alpha":
            raise EvidenceError("Record the training rows used for alternative tuning")
        if (
            not isinstance(evidence["calculations"], dict)
            or not evidence["calculations"]
        ):
            raise EvidenceError(
                "Preserve calculations supporting the alternative tuning"
            )
        if "dataset" in evidence and evidence["dataset"] != "main":
            raise EvidenceError("Tuning may use only the main trajectory")
        raise Unsupported(
            "The documented alternative tuning method requires independent review"
        )

    def tuning_check():
        x, y = xy("main")
        for which in STRATEGIES:
            item, outer_train, _ = outer(which)
            alternative_tuning(item, outer_train)
            records = item["tuning"]
            if not records:
                return False
            for fold in records:
                train, test = partition(
                    fold["train_indices"], fold["test_indices"], 200
                )
                if not np.isin(np.concatenate([train, test]), outer_train).all():
                    return False
                if not len(train) or not len(test):
                    return False
                model = fold["model"]
                if not _model_ok(x[train], y[train], model):
                    return False
                prediction = predict(x[test], model)
                if not close(
                    prediction, fold["predictions"], atol=1e-7
                ) or not _metric_match(y[test], prediction, fold["metrics"]):
                    return False
        return True

    r.check("nested_tuning_protects_both_outer_holdouts", 5, _supported(tuning_check))

    def selection_check():
        x, y = xy("main")
        for which in STRATEGIES:
            item, outer_train, _ = outer(which)
            alternative_tuning(item, outer_train)
            criterion = item["selection_criterion"]
            if not isinstance(criterion, str) or not criterion.strip():
                raise EvidenceError("Record the declared selection criterion")
            if criterion not in {
                "minimum_mean_rmse",
                "minimum_mean_mae",
                "maximum_mean_r2",
                "largest_alpha_within_rmse_tolerance",
            }:
                raise Unsupported(
                    "Declared selection criterion needs independent review: "
                    + criterion
                )
            metric = (
                "mae"
                if criterion == "minimum_mean_mae"
                else "r2"
                if criterion == "maximum_mean_r2"
                else "rmse"
            )
            scores = {}
            for fold in item["tuning"]:
                _, test = partition(fold["train_indices"], fold["test_indices"], 200)
                alpha = _number(fold["model"]["alpha"])
                value = metrics(y[test], predict(x[test], fold["model"]))[metric]
                scores.setdefault(alpha, []).append(value)
            if not scores:
                return False
            means = {alpha: float(np.mean(values)) for alpha, values in scores.items()}
            selected = _number(item["model"]["alpha"])
            if selected not in means:
                return False
            if criterion == "largest_alpha_within_rmse_tolerance":
                tolerance = _number(item["selection_tolerance_eV"])
                if tolerance < 0:
                    return False
                eligible = [
                    a
                    for a, value in means.items()
                    if value <= min(means.values()) + tolerance + 1e-10
                ]
                if selected != max(eligible):
                    return False
            elif criterion == "maximum_mean_r2":
                if means[selected] < max(means.values()) - 1e-10:
                    return False
            elif means[selected] > min(means.values()) + 1e-10:
                return False
        return True

    r.check("selection_follows_declared_criterion", 2, _supported(selection_check))

    @lru_cache(None)
    def evaluation(name):
        if name in STRATEGIES:
            item, _, test = outer(name)
            x, y = xy("main")
            prediction = predict(x[test], item["model"])
            if not result_close(prediction, item["predictions"], atol=1e-07):
                raise EvidenceError(
                    "Validation predictions do not follow exported coefficients"
                )
            return y[test], prediction
        x, y = xy(name)
        prediction = predict(x, models()["final"]["model"])
        if not result_close(
            prediction, e.array(f"runs.{name}.predictions"), atol=1e-07
        ):
            raise EvidenceError(
                "Independent predictions do not follow frozen exported coefficients"
            )
        return y, prediction

    def metric_units():
        return e.results["metric_units"] == {
            "mae": "eV",
            "rmse": "eV",
            "r2": "dimensionless",
        }

    def validation_metrics():
        return metric_units() and all(
            _metric_match(*evaluation(s), e.results["validation"][s])
            for s in STRATEGIES
        )

    r.check("reproduced_validation_predictions_and_metrics", 4, validation_metrics)

    def budgets():
        pairs = [(len(outer(s)[1]), len(outer(s)[2])) for s in STRATEGIES]
        claims = e.results["validation"].get("comparison", {})
        if not isinstance(claims, dict):
            raise EvidenceError("Validation budget summaries must be an object")
        for key, counts in (
            ("training_counts", [p[0] for p in pairs]),
            ("evaluation_counts", [p[1] for p in pairs]),
        ):
            if key in claims and not np.array_equal(claims[key], counts):
                return False
        return optional_results_match(
            claims,
            {
                "training_count_ratio": pairs[0][0] / pairs[1][0],
                "evaluation_count_ratio": pairs[0][1] / pairs[1][1],
            },
        )

    r.check("comparable_or_quantified_validation_budgets", 4, budgets)

    def final_fit():
        if (
            e.artifact("pipeline_checkpoint", "trained_pipeline", "estimator")
            .stat()
            .st_size
            <= 0
            or e.artifact("inference_entrypoint", "inference_script").stat().st_size
            <= 0
        ):
            return False
        item = models()["final"]
        train = _indices(item["train_indices"], 200)
        x, y = xy("main")
        if not len(train):
            return False
        source = item["selection_source"]
        if not isinstance(source, str) or not source.strip():
            raise EvidenceError(
                "Record final model selection using the main trajectory"
            )
        source = source.strip().lower()
        if source in RUNS[1:]:
            raise EvidenceError(
                "Independent trajectories cannot select the final model"
            )
        if source not in STRATEGIES and "selection_indices" not in item:
            raise EvidenceError("Record main-trajectory rows used for final selection")
        if source not in STRATEGIES and "selection_dataset" not in item:
            raise EvidenceError(
                "Record the dataset namespace for final model selection"
            )
        for field in ("selection_dataset", "training_dataset"):
            if field in item:
                dataset_name = item[field]
                if not isinstance(
                    dataset_name, str
                ) or dataset_name.strip().lower() not in {
                    "main",
                    "main_trajectory",
                    "train",
                    "training",
                }:
                    raise EvidenceError(
                        "Final selection and fitting must use the main trajectory dataset"
                    )
        if "selection_indices" in item and not len(
            _indices(item["selection_indices"], 200)
        ):
            return False
        if len(train) < 200 and not item.get("training_selection"):
            raise EvidenceError("Record the final training-subset selection method")
        return _model_ok(x[train], y[train], item["model"])

    r.check("final_main_only_affine_pipeline", 8, _supported(final_fit))
    for name in RUNS[1:]:
        r.check(
            f"{name}_coefficient_predictions",
            3,
            lambda name=name: bool(evaluation(name)[0].size),
        )
        r.check(
            f"{name}_mae_rmse_r2",
            2,
            lambda name=name: metric_units()
            and _metric_match(*evaluation(name), e.results["tests"][name]),
        )

    def comparisons():
        computed = {
            name: metrics(*evaluation(name))["rmse"]
            for name in (*STRATEGIES, *RUNS[1:])
        }
        expected = {
            "randomized_minus_time_aware_rmse_eV": computed["randomized"]
            - computed["time_aware"],
            "independent_500_minus_800_rmse_eV": computed["independent_500"]
            - computed["independent_800"],
            "independent_1100_minus_800_rmse_eV": computed["independent_1100"]
            - computed["independent_800"],
            "shifted_temperature_differences_K": [
                temperatures(run)["production_temperature_mean_K"]
                - temperatures("independent_800")["production_temperature_mean_K"]
                for run in RUNS[2:]
            ],
        }
        return optional_results_match(
            e.results.get("comparisons", {}), expected, atol=1e-7
        )

    r.check("quantified_temporal_and_temperature_transfer_comparisons", 7, comparisons)
    r.unverified(
        "execution_and_information_access_provenance",
        "Artifacts establish numerical consistency, not actual supplied-input/teacher/SOAP identity, hidden data access, pipeline freeze chronology, or absence of unrecorded resets.",
    )
    r.unverified(
        "scientific_sampling_adequacy",
        "Numeric diagnostics alone do not establish scientific adequacy of the chosen gap, equilibration, or correlation truncation.",
    )
