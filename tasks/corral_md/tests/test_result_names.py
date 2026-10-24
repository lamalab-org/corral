"""Public result names preserve numerical scoring and older saved submissions."""

import copy
import json

import pytest
from corral_md.scoring.common import Evidence, EvidenceError
from corral_md.scoring.contracts import normalize_results

from . import test_score_task_2 as glass
from . import test_score_task_5 as silicon
from . import test_scoring_level2_quantities as quantities
from . import test_workflow_task_4 as palladium

CASES = [
    (1, {"diffusion_m2_s": 1.7e-8}, {"diffusion": 1.7e-8}),
    (2, {"cooling_tg_K": 1576.88}, {"tg": 1576.88}),
    (4, {"zpe_eV_per_atom": 0.03}, {"zpe": 0.03}),
    (4, {"zpe_eV_per_atom": None}, {"zpe": None}),
    (
        5,
        {
            "fit": {
                "isotropic_coefficient_eV_per_strain": 0.4,
                "uniaxial_coefficient_eV_per_strain": 0.2,
            }
        },
        {"fit": {"isotropic": 0.4, "uniaxial": 0.2}},
    ),
    (
        6,
        {"mean_frequency_THz": 2.0, "std_frequency_THz": 0.5},
        {"mean_frequency": 2.0, "std_frequency": 0.5},
    ),
    (7, {"beta_volume": 1e-4}, {"beta": 1e-4}),
    (10, {"heat_capacity": {"eV_per_atom_K": 0.0003}}, {"heat_capacity": 0.0003}),
]


@pytest.mark.parametrize(("task", "legacy", "canonical"), CASES)
def test_result_aliases_preserve_values_and_source(task, legacy, canonical):
    original = copy.deepcopy(legacy)
    assert normalize_results(legacy, task, 2) == canonical
    assert legacy == original
    assert normalize_results(canonical, task, 2) == canonical


@pytest.mark.parametrize(("task", "legacy", "canonical"), CASES[:-1])
def test_conflicting_aliases_are_rejected(task, legacy, canonical):
    combined = copy.deepcopy(legacy)
    if task == 5:
        combined["fit"].update(canonical["fit"])
        target = combined["fit"]
        key = "isotropic"
    else:
        combined.update(canonical)
        target = combined
        key = next(iter(canonical))
    assert normalize_results(combined, task, 2) == canonical
    target[key] = 999
    with pytest.raises(EvidenceError, match="Conflicting result fields"):
        normalize_results(combined, task, 2)


@pytest.mark.parametrize("task", [2, 4, 5, 6, 7, 10])
def test_short_names_use_the_same_numerical_checks(tmp_path, task):
    if task == 2:
        path, _, _ = glass.calculation(tmp_path)
        evaluate = glass.score
    elif task == 4:
        path = palladium._submission(tmp_path)
        evaluate = palladium.score
    elif task == 5:
        path = silicon.make_submission(tmp_path)
        evaluate = silicon.scorer().evaluate
    else:
        factory = {
            6: quantities.phonons,
            7: quantities.expansion,
            10: quantities.heat_capacity,
        }[task]
        path, scorer = factory(tmp_path)
        evaluate = scorer.evaluate
    original = evaluate(path)
    assert original["score"] == 1, original
    data = json.loads(path.read_text())
    results = data["results"]
    if task == 2:
        results["tg"] = results.pop("cooling_tg_K")
        target, key = results, "tg"
    elif task == 4:
        results["zpe"] = results.pop("zpe_eV_per_atom")
        target, key = results, "zpe"
    elif task == 5:
        target = results["fit"]
        target["isotropic"] = target.pop("isotropic_coefficient_eV_per_strain")
        target["uniaxial"] = target.pop("uniaxial_coefficient_eV_per_strain")
        key = "isotropic"
    elif task == 6:
        results["mean_frequency"] = results.pop("mean_frequency_THz")
        results["std_frequency"] = results.pop("std_frequency_THz")
        target, key = results, "mean_frequency"
    elif task == 7:
        results["beta"] = results.pop("beta_volume")
        target, key = results, "beta"
    else:
        results["heat_capacity"] = results["heat_capacity"]["eV_per_atom_K"]
        target, key = results, "heat_capacity"
    path.write_text(json.dumps(data))
    current = evaluate(path)
    assert current["score"] == 1, current
    computed = "calculated_results" if task in (4, 5) else "computed"
    assert current[computed] == original[computed]
    target[key] *= 2
    path.write_text(json.dumps(data))
    assert evaluate(path)["score"] == 0


def test_normalization_does_not_change_evidence_identity(tmp_path):
    path, _, _ = glass.calculation(tmp_path)
    before = path.read_bytes()
    raw = Evidence(path)
    normalized = Evidence(path, task_number=2, level=2)
    assert normalized.results["tg"] == raw.results["cooling_tg_K"]
    assert normalized.manifest == raw.manifest
    assert normalized.fingerprint() == raw.fingerprint()
    assert path.read_bytes() == before


def test_short_zpe_allows_null_for_unstable_modes(tmp_path):
    path = palladium._submission(tmp_path, sign=-1)
    data = json.loads(path.read_text())
    data["results"]["zpe"] = data["results"].pop("zpe_eV_per_atom")
    assert data["results"]["zpe"] is None
    path.write_text(json.dumps(data))
    assert palladium.score(path)["score"] == 1


@pytest.mark.parametrize("value", [None, True, "1576.88", float("nan"), float("inf")])
def test_short_names_still_require_finite_numbers(tmp_path, value):
    path, answer, _ = glass.calculation(tmp_path)
    answer["results"] = {"tg": value}
    path.write_text(json.dumps(answer))
    assert glass.score(path)["score"] == 0
