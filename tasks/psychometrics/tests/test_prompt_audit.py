"""Answer-leak and malformed-submission regressions from the prompt audit."""

import importlib
import json
from types import SimpleNamespace

import pytest
import semopy

from corral_psychometrics import paths
from corral_psychometrics.env import _task_prompt
from corral_psychometrics.generators.level_1 import gen_l1_t01_hsns_structure as structure
from corral_psychometrics.score import check_claims, evaluate_model, score_model_criteria


def definition(level, number):
    return json.loads(paths.find(level, number).definition.read_text())[0]


@pytest.mark.parametrize("value", [None, "omitted"])
def test_required_factor_correlation_cannot_be_skipped(value):
    task = paths.find(1, 1)
    truth = json.loads(task.truth.read_text())["scored"]
    answer = {"model_syntax": structure.reference_syntax(), "loadings": truth["loadings"]}
    if value is None:
        answer["factor_correlation"] = None
    result = score_model_criteria(answer, definition(1, 1)["scoring_params"])
    assert result["score_binary"] == 0, result
    assert result["checks_vector"]["factor_correlation"] == "FAIL"


@pytest.mark.parametrize("value", ["false", "true", 1, [False], {"value": True}])
def test_boolean_claims_require_json_booleans(value):
    truth = {"answers": {"claim": True}}
    claim = [{"key": "answers", "fn": "score_boolean_panel", "truth_key": "answers"}]
    assert check_claims({"answers": {"claim": value}}, truth, claim, 1)["answers"] is False
    assert check_claims({"answers": {"claim": True}}, truth, claim, 1)["answers"] is True


@pytest.mark.parametrize("value", [[], None, True, "text"])
def test_generic_scorer_rejects_non_object_json(value):
    assert (
        score_model_criteria(json.dumps(value), definition(1, 1)["scoring_params"])["score_binary"]
        == 0
    )


def test_failed_optimizer_cannot_receive_model_credit(monkeypatch):
    import numpy as np
    import pandas as pd

    original = semopy.Model.fit

    def failed_fit(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        result.success = False
        return result

    monkeypatch.setattr(semopy.Model, "fit", failed_fit)
    X = pd.DataFrame(np.random.default_rng(9).normal(size=(200, 4)), columns=list("abcd"))
    with pytest.raises(ValueError, match="did not converge"):
        evaluate_model("F =~ a+b+c+d", X, list(X))


@pytest.mark.parametrize("level,number", [(level, n) for level in (1, 2) for n in range(1, 11)])
def test_agent_prompts_do_not_render_private_task_fields(level, number):
    task = definition(level, number)
    public = SimpleNamespace(
        description=task["description"],
        submission_format=task["submission_format"],
        initial_input=task["initial_input"],
    )
    env = SimpleNamespace(current_task=public)
    original = _task_prompt(env, None)
    marker = "PRIVATE_ANSWER_MARKER_47"
    public.initial_input = {
        **public.initial_input,
        "scoring_params": marker,
        "truth_path": marker,
        "data_sha256": marker,
        "seed": marker,
    }
    assert _task_prompt(env, None) == original
    assert marker not in original
    assert "scoring_params" not in original
    assert "truth/" not in original


def test_structure_discovery_schema_does_not_assume_its_answer():
    task = definition(1, 3)
    assert "broad trait factor" not in task["submission_format"]
    assert "multiple_traits_with_method_effect" in task["submission_format"]
    assert "multiple_traits" in task["submission_format"]
    task = definition(1, 7)
    assert '"<country>"' in task["submission_format"]
    assert "<M|P|N>" in task["submission_format"]
    assert '"CA"' not in task["submission_format"]


@pytest.mark.parametrize("level,number", [(1, 3), (2, 3), (2, 4)])
def test_presentation_logs_include_unaffected_candidates(level, number):
    gen = importlib.import_module(paths.find(level, number).generator)
    book = (paths.find(level, number).artifacts / "codebook.md").read_text()
    assert set(gen.METHOD_ITEMS).isdisjoint(gen.CONTROL_PAGE_ITEMS)
    assert len(gen.CONTROL_PAGE_ITEMS) == len(gen.METHOD_ITEMS)
    for item in (*gen.METHOD_ITEMS, *gen.CONTROL_PAGE_ITEMS):
        assert (
            item in book.split("Presentation log:")[-1]
            if level == 1
            else item in book.split("Items ")[-1]
        )
    assert "variance 0." not in book


def test_task_wording_matches_generators():
    for level in (1, 2):
        for number in range(1, 11):
            gen = importlib.import_module(paths.find(level, number).generator)
            task = definition(level, number)
            assert task["description"] == gen.PROMPT
            assert task["submission_format"] == gen.SUBMISSION_FORMAT


def test_internal_derived_item_sets_remain_valid(monkeypatch):
    import corral_psychometrics.score as scorer

    monkeypatch.setattr(scorer, "biased_items_from_fit", lambda *args: {"x", "y"})
    claim = [
        {
            "key": "affected",
            "fn": "score_item_set",
            "truth_key": "affected",
            "derive_from": "refit_covariate_paths",
        }
    ]
    assert (
        check_claims({}, {"affected": ["x", "y"]}, claim, 1, model=None, items=["x", "y"])[
            "affected"
        ]
        is True
    )
    public_claim = [{key: value for key, value in claim[0].items() if key != "derive_from"}]
    assert (
        check_claims(
            {"affected": {"x": "anything", "y": "anything"}},
            {"affected": ["x", "y"]},
            public_claim,
            1,
        )["affected"]
        is False
    )
