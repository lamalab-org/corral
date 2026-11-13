"""Common model-quality standards across the continuous SEM tasks."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

from corral_psychometrics import paths
from corral_psychometrics.score import model_quality_checks, score_model_criteria
from corral_psychometrics.scoring_policy import POLICY_NAME, comparative_rules, validity_rules


def test_continuous_tasks_declare_the_shared_policy():
    for level in (1, 2):
        for number in range(1, 11):
            if (level, number) == (2, 9):
                continue
            params = json.loads(paths.find(level, number).definition.read_text())[0][
                "scoring_params"
            ]
            assert params["model_quality_policy"] == POLICY_NAME
            if "tier_1_constraints" in params:
                assert params["tier_1_constraints"] == validity_rules()
            if "tier_2_comparative" in params:
                assert params["tier_2_comparative"][:4] == comparative_rules()


@pytest.mark.parametrize("number", [6, 7, 8])
def test_custom_scorers_check_validity_fit_and_parsimony(number):
    task = paths.find(2, number)
    gen = importlib.import_module(task.generator)
    params = json.loads(task.definition.read_text())[0]["scoring_params"]
    result = score_model_criteria(gen.candidate_submissions()["correct"], params)
    assert result["score_binary"] == 1, result
    checks = result["checks_vector"]
    for suffix in (
        "converged",
        "positive_df",
        "no_negative_variance",
        "CFI",
        "RMSEA",
        "SRMR",
        "BIC",
    ):
        values = [value for name, value in checks.items() if name.endswith("_quality_" + suffix)]
        assert values and all(value == "PASS" for value in values), (suffix, checks)
    quality = result["recorded"]["model_quality"]
    if number == 8:
        assert (
            quality["replication"]["submitted"]["n_free_parameters"]
            == quality["replication"]["reference"]["n_free_parameters"]
        )


def sample():
    rng = np.random.default_rng(72)
    trait = rng.normal(size=6000)
    return pd.DataFrame(
        {item: 0.7 * trait + np.sqrt(0.51) * rng.normal(size=len(trait)) for item in "abcdef"}
    )


def test_valid_factor_sign_changes_do_not_change_model_quality():
    X = sample()
    base = "F =~ a+b+c+d+e+f"
    base = "F =~ " + "+".join(f"1*{item}" for item in X)
    reversed_factor = "F =~ " + "+".join(f"-1*{item}" for item in X)
    checks, _ = model_quality_checks(reversed_factor, base, X, list(X), "quality")
    assert all(value == "PASS" for value in checks.values()), checks


def test_extra_parameters_cannot_win_on_fit_alone():
    X = sample()
    base = "F =~ a+b+c+d+e+f"
    bloated = base + "\na ~~ b+c\nb ~~ c"
    checks, evidence = model_quality_checks(bloated, base, X, list(X), "quality")
    assert checks["quality_BIC"] == "FAIL", evidence
    assert evidence["submitted"]["n_free_parameters"] > evidence["reference"]["n_free_parameters"]


def test_saturated_model_is_not_accepted_as_a_measurement_explanation():
    X = sample()[list("abc")]
    checks, evidence = model_quality_checks("F =~ a+b+c", "F =~ a+b+c", X, list(X), "quality")
    assert checks["quality_positive_df"] == "FAIL", evidence
