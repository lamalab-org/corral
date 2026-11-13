"""Shared quality criteria for continuous SEM task submissions.

Absolute fit cutoffs cannot replace task-specific evidence: some exercises
intentionally contain model misfit. Compare competing models on the same rows
and observed variables, alongside admissibility and scientific interpretation.
"""

from copy import deepcopy

POLICY_NAME = "continuous_sem_v1"
_VALIDITY = [
    {"key": "converged"},
    {"key": "no_negative_variance"},
    {"key": "positive_df"},
    {"key": "finite_standard_errors", "max_standard_error": 10.0},
    {"key": "no_redundant_factor", "phi_max": 0.90},
    {
        "key": "no_collapsed_factor",
        "min_salient_loading": 0.30,
        "min_salient_per_factor": 2,
        "sign_reversal_at": -0.10,
    },
]
_COMPARISONS = [
    {"key": "CFI", "direction": "higher", "eps": 0.005, "criterion": "global_fit"},
    {"key": "RMSEA", "direction": "lower", "eps": 0.005, "criterion": "global_fit"},
    {"key": "SRMR", "direction": "lower", "eps": 0.005, "criterion": "global_fit"},
    {"key": "BIC", "direction": "lower", "eps": 10.0, "criterion": "parsimony"},
]


def validity_rules():
    return deepcopy(_VALIDITY)


def comparative_rules(*, population=False):
    rules = deepcopy(_COMPARISONS)
    if population:
        rules.append(
            {
                "key": "sigma_max_abs_deviation",
                "direction": "lower",
                "eps": 0.010,
                "criterion": "accuracy",
            }
        )
    return rules


def validity_options():
    return {key: value for rule in _VALIDITY for key, value in rule.items() if key != "key"}
