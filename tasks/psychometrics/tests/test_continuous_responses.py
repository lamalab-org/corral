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


def test_equivalent_identified_method_factors_receive_full_credit():
    task = paths.find(1, 3)
    params = json.loads(task.definition.read_text())[0]["scoring_params"]
    data = pd.read_csv(task.artifacts / "data.csv", sep="\t")
    X = C.analysis_sample(data, local.ITEMS)
    lines = ["G =~ " + "+".join(local.ITEMS)]
    for index, ((a, b), _) in enumerate(local.RESIDUAL_PAIRS):
        # Fixed equal loadings identify each pair; its variance is estimated.
        lines.extend([f"M{index} =~ 1*{a}+1*{b}", f"G ~~ 0*M{index}"])
    lines.append("M0 ~~ 0*M1")
    syntax = "\n".join(lines)
    fitted = semopy.Model(syntax)
    fitted.fit(X)
    rows = fitted.inspect(std_est=True)
    rows = rows[(rows.op == "~") & (rows.rval == "G")]
    submission = {
        "model_syntax": syntax,
        "loadings": {row.lval: abs(float(row["Est. Std"])) for _, row in rows.iterrows()},
        "factor_correlation": None,
    }
    result = score_model_criteria(submission, params, base_dir=paths.task_root())
    assert result["score_binary"] == 1.0, result

    # An extra unexplained pair is still wrong even when written as a method.
    a, b = local.DECOY_PAIR
    submission["model_syntax"] += f"\nExtra =~ 1*{a}+1*{b}\nExtra ~~ 0*G+0*M0+0*M1"
    assert (
        score_model_criteria(submission, params, base_dir=paths.task_root())["score_binary"] == 0.0
    )
