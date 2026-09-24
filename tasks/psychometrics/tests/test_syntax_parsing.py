"""Check how submitted model syntax is read.

Every scorer reads a submitted model through these helpers, so what they
tolerate is what a submitter may write.
"""

import pytest

from corral_psychometrics.score import (
    InvalidSubmission,
    latent_names,
    parse_statement,
    statements,
    validate_syntax,
)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("F1 =~ HSNS1+HSNS2", ("=~", "F1", ["HSNS1", "HSNS2"])),
        ("F1=~HSNS1+HSNS2", ("=~", "F1", ["HSNS1", "HSNS2"])),
        ("  F1  =~  HSNS1 + HSNS2  ", ("=~", "F1", ["HSNS1", "HSNS2"])),
        ("F1 =~ 0.5*HSNS1+HSNS2", ("=~", "F1", ["HSNS1", "HSNS2"])),
        ("HSNS5 ~~ HSNS7", ("~~", "HSNS5", ["HSNS7"])),
        ("F1 ~~ 0*F2", ("~~", "F1", ["F2"])),
        ("behavior ~ F + gender", ("~", "behavior", ["F", "gender"])),
        ("behavior~F+gender", ("~", "behavior", ["F", "gender"])),
    ],
)
def test_statements_are_read_however_they_are_spaced(line, expected):
    assert tuple(parse_statement(line)) == expected


@pytest.mark.parametrize("line", ["", "   ", "\t"])
def test_blank_lines_state_nothing(line):
    assert parse_statement(line) is None


def test_measurement_is_distinguished_from_covariance_and_regression():
    # `=~` and `~~` both contain `~`, so operator order decides these.
    assert parse_statement("F1 =~ HSNS1").op == "=~"
    assert parse_statement("F1 ~~ F2").op == "~~"
    assert parse_statement("F1 ~ gender").op == "~"


def test_latent_names_are_whatever_the_submitter_chose():
    spec = "Alpha =~ HSNS1+HSNS2\nBeta =~ HSNS3\nAlpha ~~ Beta"
    assert latent_names(spec) == {"Alpha", "Beta"}
    assert len(statements(spec)) == 3


def test_validator_rejects_a_line_that_states_nothing():
    with pytest.raises(InvalidSubmission, match="no permitted operator"):
        validate_syntax("F1 =~ HSNS1\nHSNS1 HSNS2", ["HSNS1", "HSNS2"])
