"""Stargazer benchmark; importing public numerics does not load private code."""

from importlib import import_module

__all__ = [
    "CandidatePlanet",
    "CandidateSubmission",
    "EvaluationCriteria",
    "PlanetParams",
    "StargazerTask",
    "evaluate_submission",
    "load_task",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = (
        "public_rv"
        if name in {"CandidatePlanet", "CandidateSubmission", "PlanetParams"}
        else "score"
        if name in {"EvaluationCriteria", "evaluate_submission"}
        else "models"
    )
    value = getattr(import_module(f"stargazer.{module}"), name)
    globals()[name] = value
    return value
