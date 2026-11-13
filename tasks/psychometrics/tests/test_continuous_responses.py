"""Regression checks for continuous responses and equivalent method models."""

import json

import numpy as np
import pandas as pd
import pytest
import semopy

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C
from corral_psychometrics.generators.level_1 import gen_l1_t02_dd_structure as dd
from corral_psychometrics.generators.level_1 import gen_l1_t03_local_dependence as local
from corral_psychometrics.generators.level_1 import gen_l1_t09_careless_responding as careless
from corral_psychometrics.score import score_model_criteria


@pytest.mark.parametrize(
    "level,number", [(level, n) for level in (1, 2) for n in range(1, 11) if (level, n) != (2, 9)]
)
def test_questionnaire_artifacts_contain_continuous_responses(level, number):
    task = paths.find(level, number)
    found = False
    for path in task.artifacts.glob("*.csv"):
        data = pd.read_csv(path, sep="\t", nrows=100)
        columns = [c for c in data if c in C.ALL_ITEMS or (c.startswith("Q") and c[1:].isdigit())]
        if not columns:
            continue
        values = data[columns].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        assert np.mean(values != np.round(values)) > 0.8, path
        found = True
    assert found, task


def test_zero_is_observed_and_nan_is_missing():
    data = pd.DataFrame({"country": ["US"] * 3, "x": [0.0, np.nan, -1.2]})
    assert C.analysis_sample(data, ["x"]).x.tolist() == [0.0, -1.2]


def test_straightlining_screen_handles_floating_point_constants():
    data = pd.DataFrame(
        [np.full(len(careless.ITEMS), 0.1), np.arange(len(careless.ITEMS))], columns=careless.ITEMS
    )
    data["country"] = "US"
    kept = careless.analysis_sample(data, screen="spread")
    assert kept.index.tolist() == [1]


def test_dirty_dozen_facet_reliability():
    assert all(0.60 <= value <= 0.70 for value in dd.facet_reliability().values())


def test_equivalent_method_scaling_receives_full_credit():
    task = paths.find(1, 3)
    params = json.loads(task.definition.read_text())[0]["scoring_params"]
    data = pd.read_csv(task.artifacts / "data.csv", sep="\t")
    X = C.analysis_sample(data, local.ITEMS)
    # Doubling method loadings and estimating one quarter of the variance
    # preserves the implied covariance, parameter count and substantive answer.
    syntax = local.reference_syntax().replace("1*HSNS", "2*HSNS")
    fitted = semopy.Model(syntax)
    fitted.fit(X)
    rows = fitted.inspect(std_est=True)
    rows = rows[(rows.op == "~") & (rows.rval == "G")]
    submission = {
        "model_syntax": syntax,
        "loadings": {row.lval: abs(float(row["Est. Std"])) for _, row in rows.iterrows()},
        "factor_correlation": None,
        "measurement_explanation": "one_trait_with_method_effect",
        "trait_variance_share": local.variance_shares()["trait"],
    }
    result = score_model_criteria(submission, params, base_dir=paths.task_root())
    assert result["score_binary"] == 1.0, result

    a, b = local.DECOY_PAIR
    submission["model_syntax"] += f"\nExtra =~ 1*{a}+1*{b}\nExtra ~~ 0*G+0*Format"
    assert (
        score_model_criteria(submission, params, base_dir=paths.task_root())["score_binary"] == 0.0
    )


def test_method_variance_is_not_counted_as_trait_reliability():
    loadings = np.array([local.LOADINGS[item][1] for item in local.ITEMS])
    method = np.array([item in local.METHOD_ITEMS for item in local.ITEMS], dtype=float)
    unique = 1 - loadings**2 - method * local.METHOD_VARIANCE
    covariance = (
        np.outer(loadings, loadings)
        + local.METHOD_VARIANCE * np.outer(method, method)
        + np.diag(unique)
    )
    shares = local.variance_shares()
    assert sum(shares.values()) == pytest.approx(1)
    assert shares["trait"] == pytest.approx(loadings.sum() ** 2 / covariance.sum())
    assert shares["method"] > 0.08
    sample = local.simulate_hsns(50_000, np.random.default_rng(843)).to_numpy()
    np.testing.assert_allclose(np.cov(sample, rowvar=False), covariance, atol=0.025)


def test_pair_method_and_residual_models_are_equivalent():
    from corral_psychometrics.score import local_dependence_pairs_from_fit

    rng = np.random.default_rng(17)
    trait, method = rng.normal(size=(2, 6000))
    items = [f"X{i}" for i in range(6)]
    X = pd.DataFrame(
        {
            item: 0.65 * trait
            + (0.4 * method if i < 2 else 0)
            + rng.normal(0, np.sqrt(1 - 0.65**2 - (0.4**2 if i < 2 else 0)), 6000)
            for i, item in enumerate(items)
        }
    )
    base = "G =~ " + "+".join(items)
    residual = C.fit(base + "\nX0 ~~ X1", X)
    method_model = C.fit(base + "\n" + C.method_syntax("Page", items[:2], ["G"]), X)
    np.testing.assert_allclose(residual.calc_sigma()[0], method_model.calc_sigma()[0], atol=0.003)
    assert (
        local_dependence_pairs_from_fit(residual, items)
        == local_dependence_pairs_from_fit(method_model, items)
        == {frozenset(items[:2])}
    )


def test_replication_distinguishes_method_from_trait_cross_loadings():
    from corral_psychometrics.generators.level_2 import gen_l2_t08_misfit_replication as replication

    task = paths.find(2, 8)
    params = json.loads(task.definition.read_text())[0]["scoring_params"]
    good = replication.candidate_submissions()["correct"]
    result = score_model_criteria(good, params, base_dir=paths.task_root())
    assert result["score_binary"] == 1, result
    assert result["recorded"]["derived_findings"]["cross_loadings"] == [
        replication.CROSS_LOADING[0]
    ]
    wrong = {**good, "model_syntax": good["model_syntax"] + "\nFormat ~ gender"}
    assert score_model_criteria(wrong, params, base_dir=paths.task_root())["score_binary"] == 0


def test_replication_accepts_equivalent_residual_representation():
    from corral_psychometrics.generators.level_2 import gen_l2_t08_misfit_replication as replication
    from corral_psychometrics.score import _drop_method_factor

    task = paths.find(2, 8)
    params = json.loads(task.definition.read_text())[0]["scoring_params"]
    good = replication.candidate_submissions()["correct"]
    syntax = _drop_method_factor(good["model_syntax"], "Format")
    items = replication.METHOD_ITEMS
    # Multiple residual terms on one line exercise equivalent compact syntax.
    syntax += "\n" + "\n".join(
        f"{a} ~~ " + "+".join(items[index + 1 :]) for index, a in enumerate(items[:-1])
    )
    result = score_model_criteria(
        {**good, "model_syntax": syntax}, params, base_dir=paths.task_root()
    )
    assert result["score_binary"] == 1, result
