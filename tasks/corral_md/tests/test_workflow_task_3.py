"""Task 3 grades raw calculation records and independently recomputed answers."""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from corral_md.scoring.common import Evidence
from corral_md.scoring.evaluator import (
    PendingReviewError,
    WorkflowScorer,
    check_level1_workflow,
    check_level2_workflow,
)
from corral_md.scoring.task_3 import build_plan
from corral_md.scoring.verification import file_ref


def write(path, value):
    Path(path).write_text(json.dumps(value))
    return str(path)


class SyntheticVerifier:
    """Controlled model responses, independent of submitted numerical claims."""

    def _calculate(self, plan, fingerprint):
        job = plan.jobs[0]
        name = job["id"]
        rows = []
        if name == "bulk":
            frames = plan.evidence.trajectory("md_trajectory")
            # Positions identify time, so altered saved energies/forces cannot pass.
            for i in plan.expected["bulk"]["indices"]:
                fraction = frames[i].info["time_fs"] / 5000
                rows.append(
                    {
                        "energy": -400 + fraction,
                        "forces": np.full((108, 3), fraction).tolist(),
                    }
                )
        else:
            for frame in job["frames"]:
                distance = np.linalg.norm(np.diff(frame["positions"], axis=0))
                error = {"teacher": 0, "student_before": 2, "student_after": 0.1}[name]
                forces = np.zeros((2, 3))
                forces[:, 0] = [1 / distance**2, -1 / distance**2]
                rows.append(
                    {
                        "energy": -1 / distance + error,
                        "forces": (forces + error).tolist(),
                    }
                )
        value = {"frames": rows}
        if name.startswith("student_"):
            value["parameter_sha256"] = ("a" if name.endswith("before") else "b") * 64
        return {
            "evidence_sha256": fingerprint,
            "challenge": plan.challenge,
            "jobs": [{"id": name, "status": "complete", "value": value}],
        }


@pytest.fixture
def submission(tmp_path):
    reference = []
    for d in np.linspace(0.6, 5, 45):
        forces = np.zeros((2, 3))
        forces[:, 0] = [1 / d**2, -1 / d**2]
        reference.append(
            {
                "symbols": ["Ag", "Ag"],
                "positions": [[0, 0, 0], [d, 0, 0]],
                "pbc": False,
                "energy": -1 / d,
                "forces": forces.tolist(),
            }
        )
    write(tmp_path / "dataset.json", reference)
    checkpoint = tmp_path / "trained.model"
    checkpoint.write_bytes(b"checkpoint bytes are only read by the isolated backend")
    digest = file_ref(checkpoint)["sha256"]
    log = {
        "teacher_model": "teacher.model",
        "student_model": "student.model",
        "default_dtype": "float64",
        "dispersion": True,
        "energy_unit": "eV",
        "force_unit": "eV/Angstrom",
        "training": {
            "steps": 20,
            "initial_parameters_sha256": "a" * 64,
            "final_parameters_sha256": "b" * 64,
            "checkpoint_sha256": digest,
        },
        "md": {
            "timestep_fs": 1,
            "temperature_dof": 321,
            "target_temperature_K": 300,
            "initial_temperature_K": 300,
            "random_seed": 1337,
            "velocity_initializations": 1,
            "remove_com_once": True,
            "thermostat": "Langevin",
            "checkpoint_sha256": digest,
        },
    }
    write(tmp_path / "run.json", log)
    initial = bulk("Ag", "fcc", a=4.09, cubic=True).repeat((3, 3, 3))
    random = np.random.default_rng(1337)
    momenta = random.normal(size=(108, 3))
    momenta -= momenta.mean(axis=0)
    momenta *= np.sqrt(
        321 * units.kB * 300 / np.sum(momenta**2 / initial.get_masses()[:, None])
    )
    displacement = random.normal(size=(108, 3)) * 0.03
    displacement -= displacement.mean(axis=0)
    frames = []
    fractions = np.linspace(0, 1, 101)
    minimum = []
    for f in fractions:
        atoms = initial.copy()
        atoms.positions += f * displacement
        delta = atoms.positions[:, None, :] - atoms.positions[None, :, :]
        distances = np.linalg.norm(delta - np.rint(delta / 12.27) * 12.27, axis=2)
        np.fill_diagonal(distances, np.inf)
        minimum.append(distances.min())
        frames.append(
            {
                "symbols": list(atoms.symbols),
                "positions": atoms.positions.tolist(),
                "cell": atoms.cell.array.tolist(),
                "pbc": True,
                "time_fs": 5000 * f,
                "momenta": (momenta * np.sqrt((300 + 100 * f) / 300)).tolist(),
                "energy": -400 + f,
                "forces": np.full((108, 3), f).tolist(),
            }
        )
    write(tmp_path / "bulk.json", frames)
    result = {
        "fitting": {
            phase: {"energy_rmse_per_atom_eV": error / 2, "force_rmse_eV_A": error}
            for phase, error in [("before", 2), ("after", 0.1)]
        },
        "bulk": {
            "temperature_mean_K": 350,
            "temperature_std_K": np.std(300 + 100 * fractions),
            "potential_energy_range_eV": 1,
            "force_max_eV_A": np.sqrt(3),
            "final_displacement_rms_A": np.sqrt(
                np.mean(np.sum(displacement**2, axis=1))
            ),
            "min_pair_distance_A": min(minimum),
        },
    }
    manifest = {
        "artifacts": {
            "dataset": "dataset.json",
            "run_log": "run.json",
            "trained_checkpoint": "trained.model",
            "md_trajectory": "bulk.json",
        },
        "results": result,
    }
    path = tmp_path / "manifest.json"
    write(path, manifest)
    return path


def score(path, level=2, verifier=None):
    return WorkflowScorer(
        3, level=level, verifier=verifier or SyntheticVerifier()
    ).evaluate(path)


def test_compact_submission_recomputes_every_answer(submission):
    report = score(submission)
    assert report["score"] == 1, report
    assert report["calculated_results"]["fitting"]["after"][
        "energy_rmse_per_atom_eV"
    ] == pytest.approx(0.05)
    assert report["calculated_results"]["bulk"]["temperature_mean_K"] == pytest.approx(
        350
    )
    assert sum(c["points"] for c in report["checks"]) == 100


def test_artifact_only_answer_keeps_fitting_and_bulk_checks(submission):
    manifest = json.loads(submission.read_text())
    manifest.pop("results")
    write(submission, manifest)
    report = score(submission)
    assert report["score"] == 1, report
    assert len(report["calculated_results"]["bulk"]) == 6
    assert set(report["calculated_results"]["fitting"]) == {"before", "after"}

    log_path = submission.parent / "run.json"
    log = json.loads(log_path.read_text())
    log["md"]["checkpoint_sha256"] = "wrong checkpoint"
    write(log_path, log)
    assert score(submission)["score"] == 0


def test_level1_needs_only_reference_calculations(submission):
    manifest = json.loads(submission.read_text())
    manifest["artifacts"] = {
        k: v for k, v in manifest["artifacts"].items() if k in {"dataset", "run_log"}
    }
    manifest.pop("results")
    log = json.loads((submission.parent / "run.json").read_text())
    for key in ("md", "training", "student_model"):
        log.pop(key)
    write(submission.parent / "run.json", log)
    write(submission, manifest)
    assert score(submission, level=1)["score"] == 1
    assert score(submission)["score"] == 0
    assert [j["id"] for j in build_plan(Evidence(submission), 1, "challenge").jobs] == [
        "teacher"
    ]


@pytest.mark.parametrize(
    ("section", "phase", "key"),
    [
        ("fitting", phase, key)
        for phase in ("before", "after")
        for key in ("energy_rmse_per_atom_eV", "force_rmse_eV_A")
    ]
    + [
        ("bulk", None, key)
        for key in (
            "temperature_mean_K",
            "temperature_std_K",
            "potential_energy_range_eV",
            "force_max_eV_A",
            "final_displacement_rms_A",
            "min_pair_distance_A",
        )
    ],
)
def test_altered_answer_fails(submission, section, phase, key):
    manifest = json.loads(submission.read_text())
    target = manifest["results"][section]
    if phase:
        target = target[phase]
    target[key] += 100
    write(submission, manifest)
    assert score(submission)["score"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "grid",
        "labels",
        "missing_forces",
        "checkpoint",
        "no_steps",
        "unchanged_parameters",
        "wrong_teacher",
        "no_dispersion",
        "units",
        "wrong_bulk_checkpoint",
        "truncated",
        "velocity_reset",
        "mass",
        "missing_momenta",
        "nan",
        "wrong_bulk_energy",
    ],
)
def test_bad_calculation_records_fail(submission, change):
    root = submission.parent
    log = json.loads((root / "run.json").read_text())
    frames = json.loads((root / "bulk.json").read_text())
    reference = json.loads((root / "dataset.json").read_text())
    if change == "grid":
        reference[0]["positions"][1][0] += 0.1
    elif change == "labels":
        reference[0]["energy"] += 100
    elif change == "missing_forces":
        del reference[0]["forces"]
    elif change == "checkpoint":
        (root / "trained.model").write_bytes(b"changed")
    elif change == "no_steps":
        log["training"]["steps"] = 0
    elif change == "unchanged_parameters":
        log["training"]["final_parameters_sha256"] = "a" * 64
    elif change == "wrong_teacher":
        log["teacher_model"] = "student.model"
    elif change == "no_dispersion":
        log["dispersion"] = False
    elif change == "units":
        log["energy_unit"] = "kcal/mol"
    elif change == "wrong_bulk_checkpoint":
        log["md"]["checkpoint_sha256"] = "0" * 64
    elif change == "truncated":
        frames.pop()
    elif change == "velocity_reset":
        log["md"]["velocity_initializations"] = 2
    elif change == "mass":
        frames[1]["masses"] = [1] * 108
    elif change == "missing_momenta":
        del frames[1]["momenta"]
    elif change == "nan":
        frames[1]["positions"][0][0] = float("nan")
    elif change == "wrong_bulk_energy":
        frames[-1]["energy"] += 1
    write(root / "run.json", log)
    write(root / "bulk.json", frames)
    write(root / "dataset.json", reference)
    assert score(submission)["score"] == 0


def test_unavailable_backend_stays_pending(submission):
    scorer = check_level2_workflow(3, verification_backend="offline")
    assert scorer.evaluate(submission)["score"] is None
    with pytest.raises(PendingReviewError):
        scorer(submission)
    assert (
        check_level1_workflow(3, verification_backend="offline").evaluate(submission)[
            "score"
        ]
        is None
    )


def test_unloadable_model_fails(submission):
    class Broken(SyntheticVerifier):
        def _calculate(self, plan, fingerprint):
            response = super()._calculate(plan, fingerprint)
            if plan.jobs[0]["id"] == "student_after":
                response["jobs"][0] = {
                    "id": "student_after",
                    "status": "error",
                    "detail": "Invalid checkpoint",
                }
            return response

    assert score(submission, verifier=Broken())["score"] == 0


def test_no_required_improvement_or_bulk_temperature_success(submission):
    class PoorFit(SyntheticVerifier):
        def _calculate(self, plan, fingerprint):
            response = super()._calculate(plan, fingerprint)
            if plan.jobs[0]["id"] == "student_after":
                for row in response["jobs"][0]["value"]["frames"]:
                    row["energy"] += 10
                    row["forces"] = (np.asarray(row["forces"]) + 10).tolist()
            return response

    manifest = json.loads(submission.read_text())
    manifest["results"]["fitting"]["after"] = {
        "energy_rmse_per_atom_eV": 5.05,
        "force_rmse_eV_A": 10.1,
    }
    write(submission, manifest)
    assert score(submission, verifier=PoorFit())["score"] == 1


def test_frozen_evidence_is_checked(submission):
    class Mutating(SyntheticVerifier):
        def _calculate(self, plan, fingerprint):
            response = super()._calculate(plan, fingerprint)
            (submission.parent / "trained.model").write_bytes(b"mutated during grading")
            return response

    with pytest.raises(RuntimeError, match="changed during grading"):
        score(submission, verifier=Mutating())


def test_reporting_rounding_is_allowed(submission):
    manifest = json.loads(submission.read_text())
    manifest["results"]["fitting"]["after"]["force_rmse_eV_A"] *= 1.005
    write(submission, manifest)
    assert score(submission)["score"] == 1


def test_dimer_order_does_not_change_errors(submission):
    path = submission.parent / "dataset.json"
    write(path, json.loads(path.read_text())[::-1])
    assert score(submission)["score"] == 1


@pytest.mark.parametrize(("maximum", "expected"), [(1200, 1), (6000, None)])
def test_bulk_heating_is_distinguished_from_unresolved_integration(
    submission, maximum, expected
):
    path = submission.parent / "bulk.json"
    frames = json.loads(path.read_text())
    temperatures = np.linspace(300, maximum, len(frames))
    for frame, temperature, old_temperature in zip(
        frames, temperatures, np.linspace(300, 400, len(frames)), strict=True
    ):
        frame["momenta"] = (
            np.array(frame["momenta"]) * np.sqrt(temperature / old_temperature)
        ).tolist()
    write(path, frames)
    manifest = json.loads(submission.read_text())
    manifest["results"]["bulk"].update(
        temperature_mean_K=temperatures.mean(), temperature_std_K=temperatures.std()
    )
    write(submission, manifest)
    assert score(submission)["score"] == expected
    if expected is None:
        manifest["results"]["bulk"]["temperature_mean_K"] = 300
        write(submission, manifest)
        assert score(submission)["score"] == 0


def test_independent_calculation_cannot_reuse_another_submission(submission):
    class WrongEvidence(SyntheticVerifier):
        def _calculate(self, plan, fingerprint):
            result = super()._calculate(plan, fingerprint)
            result["evidence_sha256"] = "another-submission"
            return result

    assert score(submission, verifier=WrongEvidence())["score"] == 0


def test_nonfinite_derived_errors_do_not_escape_into_the_report(submission):
    class Overflow(SyntheticVerifier):
        def _calculate(self, plan, fingerprint):
            result = super()._calculate(plan, fingerprint)
            if plan.jobs[0]["id"] == "student_after":
                result["jobs"][0]["value"]["frames"][0]["energy"] = 1e308
            return result

    report = score(submission, verifier=Overflow())
    assert report["score"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("level", [1, 2])
def test_reference_settings_use_aliases_and_explicit_calculator_description(
    submission, level
):
    log_path = submission.parent / "run.json"
    log = json.loads(log_path.read_text())
    log["teacher_model"] = "/workspace/resources/models/teacher.model"
    log["energy_units"] = log.pop("energy_unit")
    log["force_units"] = "eV/A"
    log.pop("force_unit")
    log["calculator"] = (
        f"MACECalculator(dispersion=True, default_dtype={log.pop('default_dtype')})"
    )
    log.pop("dispersion")
    write(log_path, log)
    assert score(submission, level=level)["score"] == 1


def test_checkpoint_identity_is_derived_without_unrequested_hash_fields(submission):
    path = submission.parent / "run.json"
    log = json.loads(path.read_text())
    log["training"] = {}
    log.pop("student_model")
    log["md"].pop("checkpoint_sha256")
    write(path, log)
    assert score(submission)["score"] == 1
