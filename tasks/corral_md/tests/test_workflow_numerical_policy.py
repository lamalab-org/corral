"""Shared reporting tolerance and removal of statistical deliverables."""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
from corral_md.scoring import WorkflowScorer, check_level2_workflow
from corral_md.scoring.common import EvidenceError, close, result_close


def _read(path):
    return json.loads(path.read_text())


def _write(path, data):
    path.write_text(json.dumps(data))


def _obsolete(key):
    return any(
        part in key for part in ("uncert", "sensitiv", "_sem", "standard_error")
    ) or key in {
        "stability",
        "n_blocks",
        "block_size",
        "correlation_time_samples",
        "confidence_multiplier",
        "energy_agrees",
        "structure_agrees",
        "joint_agreement",
    }


def _strip(value):
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if not _obsolete(key)}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def _submission(number, root):
    spec = importlib.util.spec_from_file_location(
        f"workflow_fixture_{number}",
        Path(__file__).with_name(f"test_workflow_task_{number}.py"),
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    fixtures.submission.__wrapped__(root)
    path = root / "manifest.json"
    manifest = _read(path)
    if "settings" not in manifest:
        _write(root / "settings.json", {"units": "real"})
        manifest["settings"] = str(root / "settings.json")
    if "scripts" not in manifest:
        script = root / "analysis.py"
        script.write_text("raise RuntimeError('scoring must not execute code')\n")
        manifest["scripts"] = [str(script)]
    _write(path, manifest)
    # Remove the files as well as their manifest links; a grader must not rely
    # on implicit conventional paths to a retired artifact.
    for role, filename in manifest["artifacts"].items():
        if _obsolete(role) and isinstance(filename, str):
            artifact = Path(filename)
            (artifact if artifact.is_absolute() else root / artifact).unlink()
    for file in root.glob("*.json"):
        _write(file, _strip(_read(file)))
    return path


# One independently calculated output from every task, with a nonzero scale.
CASES = [
    (4, ("zpe_eV_per_atom",), "dos_zpe_and_imaginary_modes"),
]


@pytest.mark.parametrize(("number", "keys", "check_name"), CASES)
def test_full_credit_without_statistics_and_numerical_allowance(
    tmp_path, number, keys, check_name
):
    path = _submission(number, tmp_path)
    grader = check_level2_workflow(number)
    if number == 4:
        from corral_md.scoring.verification import ModalVerifier

        def response(plan, fingerprint):
            return {
                "evidence_sha256": fingerprint,
                "challenge": plan.challenge,
                "jobs": [
                    {
                        "id": job["id"],
                        "status": "complete",
                        "value": plan.expected[job["id"]]["values"],
                    }
                    for job in plan.jobs
                ],
            }

        grader = WorkflowScorer(4, verifier=ModalVerifier(transport=response))
    baseline = grader.evaluate(path)
    assert baseline["score"] == pytest.approx(1), baseline
    assert sum(c["points"] for c in baseline["checks"]) == 100
    assert not any(_obsolete(c["name"]) for c in baseline["checks"] if c["points"])
    manifest = _read(path)
    target = manifest["results"]
    for key in keys[:-1]:
        target = target[key]
    original = target[keys[-1]]
    assert original != 0
    for factor, expected in [(1.005, "passed"), (1.05, "failed")]:
        target[keys[-1]] = original * factor
        _write(path, manifest)
        result = grader.evaluate(path)
        check = next(c for c in result["checks"] if c["name"] == check_name)
        assert check["status"] == expected, check
    target[keys[-1]] = original
    manifest["results"]["uncertainty"] = {"method": "unsupported legacy method"}
    manifest["results"]["sensitivity"] = {"unused": 123}
    _write(path, manifest)
    assert float(grader(path)) == pytest.approx(1)


@pytest.mark.parametrize("number", range(1, 11))
def test_prompts_omit_statistical_deliverables(number):
    root = Path(__file__).parents[1]
    task = _read(root / f"environments/level_2/tasks_json/task_{number}.json")[0]
    prompt = task["description"] + task["submission_format"]
    # Physical reference acceptance intervals are part of the task contract.
    assert not re.search(r"uncertaint|sensitiv|rtol|atol", prompt, re.I)


def test_numerical_policy_handles_sign_zero_scale_and_invalid_arrays():
    assert result_close(-100, -100.5)
    assert not result_close(-100, -105)
    assert result_close(0, 5e-16, atol=1e-15)
    assert not result_close(0, 1e-8, atol=1e-15)
    assert not result_close(1e-8, 0, atol=1e-15)
    assert not result_close(-1, 1)
    assert not result_close([1, 2], [[1, 2]])
    with pytest.raises(EvidenceError):
        result_close(np.nan, 1)
    with pytest.raises(EvidenceError):
        result_close(1, np.inf)
    # Reporting tolerance must not silently weaken state/identity comparisons.
    assert result_close(100, 100.5)
    assert not close(100, 100.5)
