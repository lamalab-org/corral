"""Malformed answers are model failures, not simulator/evaluation failures."""

import json

import pytest

from wetlab.score import score_salt


@pytest.mark.parametrize(
    "answer",
    ["[]", "null", "42", '"NaCl"', "true", "{}", '{"cation": [], "anion": "Cl-"}'],
)
def test_non_object_and_non_string_ions_score_zero(answer):
    assert score_salt(answer, {"cation": "Na+", "anion": "Cl-"}) == 0.0


def test_correct_salt_still_passes():
    salt = {"cation": "Na+", "anion": "Cl-"}
    assert score_salt(json.dumps(salt), salt) == 1.0


def test_incorrect_salt_still_fails():
    assert (
        score_salt(
            json.dumps({"cation": "K+", "anion": "Cl-"}),
            {"cation": "Na+", "anion": "Cl-"},
        )
        == 0.0
    )
