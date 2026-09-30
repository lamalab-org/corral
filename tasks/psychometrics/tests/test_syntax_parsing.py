"""Check how submitted model syntax is read.

Every scorer reads a submitted model through these helpers, so what they
tolerate is what a submitter may write.
"""

import pytest

from corral_psychometrics.score import (
    InvalidSubmission,
    latent_names,
    names_in,
    observed_in_spec,
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


# --------------------------------------------------------------------------
# The validator against semopy, which fits whatever the validator accepts
# --------------------------------------------------------------------------
ITEMS = [f"HSNS{i}" for i in range(1, 7)]
SECOND = "F2 =~ HSNS4 + HSNS5 + HSNS6"


@pytest.fixture(scope="module")
def two_factor_data():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    factors = rng.normal(size=(500, 2))
    return pd.DataFrame(
        {
            item: 0.7 * factors[:, index // 3] + 0.7 * rng.normal(size=500)
            for index, item in enumerate(ITEMS)
        }
    )


def _fits(spec, data):
    import semopy

    model = semopy.Model(spec)
    model.fit(data[ITEMS])
    return model


@pytest.mark.parametrize(
    "spec",
    [
        f"# two factors\nF1 =~ HSNS1 + HSNS2 + HSNS3\n{SECOND}",
        f"F1 =~ HSNS1 + HSNS2 + HSNS3  # first factor\n{SECOND}",
        f"F1 =~ HSNS1 + HSNS2 + HSNS3\n\n   # spare note\n{SECOND}",
        f"F1 =~ HSNS1 + -0.5*HSNS2 + HSNS3\n{SECOND}",
        f"F1 =~ HSNS1 + 1e-3*HSNS2 + HSNS3\n{SECOND}",
        f"F1 =~ HSNS1 + 2.5E-1*HSNS2 + 1.*HSNS3\n{SECOND}",
        f"Retrieval =~ HSNS1 + HSNS2 + HSNS3\n{SECOND}",
        f"Executive_system =~ HSNS1 + HSNS2 + HSNS3\n{SECOND}",
        f"F__1 =~ HSNS1 + HSNS2 + HSNS3\n{SECOND}",
        f"F1\t=~\tHSNS1+HSNS2+HSNS3\r\n{SECOND}",
    ],
)
def test_accepted_models_fit(spec, two_factor_data):
    cleaned = validate_syntax(spec, ITEMS)
    assert "#" not in cleaned
    _fits(cleaned, two_factor_data)


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (f"F1 =~\n{SECOND}", "missing a side"),
        (f"=~ HSNS1 + HSNS2\n{SECOND}", "missing a side"),
        (f"F1 =~ HSNS1 + HSNS2 ~~ HSNS3\n{SECOND}", "one statement per line"),
        (f"F1 =~ HSNS1 + HSNS2 + HSNS3; {SECOND}", "one statement per line"),
        (f"F1 =~ HSNS1 + HSNS2 +\n    HSNS3\n{SECOND}", "empty term"),
        (f"F1 =~ HSNS1 HSNS2 + HSNS3\n{SECOND}", "not a variable name"),
        (f"F1 =~ HSNS1 + HSNS2 + HSNS7\n{SECOND}", "unknown variable"),
        (f"F1 =~ HSNS1 + a*HSNS2 + HSNS3\n{SECOND}", "numeric fixings"),
        (f"F1 =~ HSNS1 + start(0.5)*HSNS2 + HSNS3\n{SECOND}", "numeric fixings"),
        # semopy cannot read these spellings of a number.
        (f"F1 =~ HSNS1 + 2.5E+0*HSNS2 + HSNS3\n{SECOND}", "not a variable name"),
        (f"F1 =~ HSNS1 + .8*HSNS2 + HSNS3\n{SECOND}", "numeric fixings"),
        (f"F1 =~ HSNS1 + +0.8*HSNS2 + HSNS3\n{SECOND}", "empty term"),
        ("# only a comment", "no statements"),
    ],
)
def test_rejected_models_say_why(spec, reason, two_factor_data):
    with pytest.raises(InvalidSubmission, match=reason):
        validate_syntax(spec, ITEMS)
    # Labels, start values and unknown names are refused by choice; everything
    # else refused here is a model semopy could not have fitted either.
    if reason not in {"numeric fixings", "unknown variable", "no statements"} or ".8*" in spec:
        with pytest.raises(Exception):  # noqa: B017 - semopy raises several types
            _fits(spec, two_factor_data)


def test_comments_never_count_as_variables():
    spec = f"F1 =~ HSNS1 + HSNS2 + HSNS3  # not HSNS6\n{SECOND}"
    assert observed_in_spec(spec, ITEMS) == set(ITEMS)
    assert observed_in_spec("F1 =~ HSNS1 + HSNS2  # HSNS3", ITEMS) == {"HSNS1", "HSNS2"}
    assert parse_statement("F1 =~ HSNS1 + HSNS2  # first") == ("=~", "F1", ["HSNS1", "HSNS2"])


def test_numbers_are_not_names():
    assert names_in("F1 =~ 1e-3*HSNS1 + -0.5*HSNS2 + 1.*HSNS13") == [
        "F1",
        "HSNS1",
        "HSNS2",
        "HSNS13",
    ]
