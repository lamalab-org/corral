"""Tests for Inspect scorer selection."""

from inference_opt.scoring_specs import spec_for


def test_chembench_uses_official_scorer():
    spec = spec_for("chembench", "numeric")
    assert spec.answer_format == "numeric"
    assert spec.scorer_name == "chembench"


def test_other_numeric_answers_use_inspect_numeric_match():
    spec = spec_for("gsm8k", "numeric")
    assert spec.answer_format == "numeric"
    assert spec.scorer_name == "match_numeric"


def test_multiple_choice_uses_inspect_choice_scorer():
    assert spec_for("mmlu_pro", "mcq_single").scorer_name == "choice"
    assert spec_for("mmlu_pro", "mcq_multi").multiple_correct


def test_free_text_uses_inspect_match_scorer():
    assert spec_for("bbh", "text").scorer_name == "match"
