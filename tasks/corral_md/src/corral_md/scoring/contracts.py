"""Validate the documented answer interface and normalize older result names."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from .common import EvidenceError

RESULT_ALIASES = {
    1: {"diffusion": "diffusion_m2_s"},
    2: {"tg": "cooling_tg_K"},
    4: {"zpe": "zpe_eV_per_atom"},
    6: {
        "mean_frequency": "mean_frequency_THz",
        "std_frequency": "std_frequency_THz",
    },
    7: {"beta": "beta_volume"},
}
FIT_ALIASES = {
    "isotropic": "isotropic_coefficient_eV_per_strain",
    "uniaxial": "uniaxial_coefficient_eV_per_strain",
}


def normalize_results(results: dict, task_number: int, level: int) -> dict:
    """Return canonical inputs without changing the submitted evidence or units."""
    if level != 2:
        return results

    def rename(values, aliases):
        result = dict(values)
        for name, old in aliases.items():
            if old not in result:
                continue
            previous = result.pop(old)
            if name in result:
                actual = result[name]
                same = actual is None and previous is None
                if type(actual) in (int, float) and type(previous) in (int, float):
                    same = math.isfinite(actual) and actual == previous
                if not same:
                    raise EvidenceError(f"Conflicting result fields: {name}, {old}")
            else:
                result[name] = previous
        return result

    result = rename(results, RESULT_ALIASES.get(task_number, {}))
    if task_number == 5 and isinstance(result.get("fit"), dict):
        result["fit"] = rename(result["fit"], FIT_ALIASES)
    if task_number == 10 and isinstance(result.get("heat_capacity"), dict):
        result["heat_capacity"] = result["heat_capacity"].get("eV_per_atom_K")
    return result


@lru_cache(None)
def submission_example(task_number: int, level: int) -> dict:
    if task_number not in range(1, 11) or level not in (1, 2):
        raise ValueError("Expected task 1..10 and level 1 or 2")
    package = Path(__file__).resolve().parent.parent
    roots = (package / "environments", package.parent.parent / "environments")
    relative = Path(f"level_{level}/tasks_json/task_{task_number}.json")
    path = next(
        (root / relative for root in roots if (root / relative).is_file()), None
    )
    if path is None:
        raise RuntimeError("Task definitions unavailable to the scoring infrastructure")
    task = json.loads(path.read_text())
    if isinstance(task, list):
        (task,) = task
    match = re.search(r"```json\s*(.*?)\s*```", task["submission_format"], re.S)
    if match is None:
        raise RuntimeError("Task definition has no JSON submission example")
    return json.loads(match.group(1))


def validate_manifest(evidence, task_number: int, level: int) -> None:
    """Require the example's keys, not its illustrative filenames or zero values.

    Additional fields remain permitted. Task 4 explicitly allows a null ZPE;
    optional diagnostics described in the prose are checked by individual scorers.
    """

    def check(actual, example, path):
        location = ".".join(path)
        if isinstance(example, dict):
            if not isinstance(actual, dict):
                raise EvidenceError(f"{location or 'Submission'} must be an object")
            for key, child in example.items():
                if key not in actual:
                    raise EvidenceError(
                        f"Missing requested submission field: {location + '.' if location else ''}{key}"
                    )
                check(actual[key], child, (*path, key))
        elif isinstance(example, list):
            if not isinstance(actual, list) or not actual:
                raise EvidenceError(f"{location} must be a nonempty list")
            for value in actual:
                check(value, example[0], path)
        elif path[0] == "artifacts":
            evidence._path(actual)
        elif actual is None and task_number == 4 and path == ("results", "zpe"):
            return
        elif type(actual) not in (int, float) or not math.isfinite(actual):
            raise EvidenceError(f"{location} must be a finite numerical result")

    manifest = {**evidence.manifest, "results": evidence.results}
    check(manifest, submission_example(task_number, level), ())
