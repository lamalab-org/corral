"""Level 1 thermodynamic logs need no unpublished atom-wise trajectory schema."""

import json

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from corral_md.scoring.common import Evidence, EvidenceError
from corral_md.scoring.formats import thermodynamic_stage
from corral_md.scoring.task_6 import Task6Scorer
from corral_md.scoring.task_10 import evaluate_submission


class RestartVerifier:
    def __init__(self):
        self.plans = []

    def _evaluate_plan(self, plan, _fingerprint):
        self.plans.append(plan)
        assert len(plan.jobs) == 1
        assert len(plan.jobs[0]["frames"]) == 1
        return ([{"id": job["id"], "status": "passed"} for job in plan.jobs], {}, {})


def _scalar_submission(tmp_path, task, *, dof_correction=False):
    """Use the observed scalar/summary layout with a small deterministic signal."""
    atoms = bulk("Al", "fcc", a=4.05, cubic=task == 10).repeat(
        (4, 4, 4) if task == 6 else (3, 3, 3)
    )
    n = len(atoms)
    dof = 3 * n - (3 if dof_correction else 0)
    momentum = np.random.default_rng(4).normal(size=(n, 3))
    momentum -= momentum.mean(axis=0)
    momentum *= np.sqrt(
        (dof / 3)
        * units.kB
        * 300
        / (momentum**2 / atoms.get_masses()[:, None]).sum(axis=0)
    )
    atoms.set_momenta(momentum)
    potential = -3.7 * n
    atoms.calc = SinglePointCalculator(atoms, energy=potential)
    restart = tmp_path / "final.extxyz"
    write(restart, atoms)
    header = {
        "type": "metadata",
        "cell": atoms.cell.tolist(),
        "natoms": n,
        "ensemble": "NVT",
        "thermostat": "Bussi (CSVR)" if task == 10 else "Langevin",
        "T_target": 300,
        "seed_velocities": 7,
        "com_removed_once": True,
        "timestep_fs": 1,
        "sample_interval_steps": 50,
        "dof": dof,
    }
    if task == 6:
        header["nsteps"] = 3000
        total = 3000
    else:
        header.update(equil_steps=2000, prod_steps=4000)
        total = 6000
    rows = [header]
    for step in range(0, total + 1, 50):
        temperature = 300 + 4 * np.sin(2 * np.pi * step / 1000)
        energy = potential + 0.04 * np.sin(2 * np.pi * step / 1000)
        row = {
            "type": "sample",
            "step": step,
            "time_ps": step / 1000,
            "temperature_K": temperature,
            "epot_eV": energy,
            "ekin_eV": temperature * dof * units.kB / 2,
        }
        if task == 10:
            row.update(
                stage="equilibration" if step <= 2000 else "production",
                Tx_K=temperature,
                Ty_K=temperature,
                Tz_K=temperature,
                etot_per_atom_eV=(energy + row["ekin_eV"]) / n,
            )
        rows.append(row)
    rows.append({"type": "summary", "restart_files": [restart.name]})
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps(row) for row in rows))
    return {"artifacts": {"run_log": str(log)}}, rows, restart


def _evaluate(manifest, task, verifier):
    if task == 6:
        return Task6Scorer(level=1, verifier=verifier).evaluate(manifest)
    return evaluate_submission(manifest, level=1, verifier=verifier)


@pytest.mark.parametrize("task", [6, 10])
@pytest.mark.parametrize("dof_correction", [False, True])
def test_scalar_log_and_restart_are_scored_without_atomwise_samples(
    tmp_path, task, dof_correction
):
    manifest, rows, _ = _scalar_submission(
        tmp_path, task, dof_correction=dof_correction
    )
    verifier = RestartVerifier()
    report = _evaluate(manifest, task, verifier)
    assert report["score"] == 1, report
    assert len(verifier.plans) == 1
    job = verifier.plans[0].jobs[0]
    assert job["parameters"]["model"] == "teacher.model"
    assert job["parameters"]["dispersion"] is False
    expected = verifier.plans[0].expected[job["id"]]["values"]
    assert expected["frames"][0]["energy"] == pytest.approx(rows[-2]["epot_eV"])
    assert "positions_A" not in rows[1]


@pytest.mark.parametrize("task", [6, 10])
@pytest.mark.parametrize(
    "corruption",
    [
        "temperature",
        "energy",
        "species",
        "restart",
        "truncated",
        "summary_count",
        "summary_mean",
    ],
)
def test_scalar_evidence_inconsistency_scores_zero(tmp_path, task, corruption):
    manifest, rows, restart = _scalar_submission(tmp_path, task)
    if corruption == "temperature":
        rows[4]["temperature_K"] += 100
    elif corruption == "energy":
        rows[-2]["epot_eV"] += 10
    elif corruption == "species":
        atoms = read(restart)
        atoms.numbers[0] = 29
        write(restart, atoms)
    elif corruption == "restart":
        rows[-1]["restart_files"] = []
    elif corruption == "truncated":
        rows.pop(-2)
    elif corruption == "summary_count":
        rows[-1]["n_samples"] = len(rows)
    elif corruption == "summary_mean":
        rows[-1]["final_half_mean_T_K" if task == 6 else "mean_T"] = 400
    (tmp_path / "run.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    verifier = RestartVerifier()
    report = _evaluate(manifest, task, verifier)
    assert report["score"] == 0
    assert any(check["status"] == "failed" for check in report["checks"])
    assert not verifier.plans


def test_scalar_logs_cannot_claim_arbitrary_temperature_degrees_of_freedom(tmp_path):
    manifest, rows, _ = _scalar_submission(tmp_path, 10)
    rows[0]["dof"] = 100
    (tmp_path / "run.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(EvidenceError, match="degrees of freedom"):
        thermodynamic_stage(Evidence(manifest), 10)


def test_scalar_adapter_does_not_guess_unknown_record_types(tmp_path):
    manifest, rows, _ = _scalar_submission(tmp_path, 6)
    rows[3]["type"] = "unknown_event"
    (tmp_path / "run.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    report = _evaluate(manifest, 6, RestartVerifier())
    assert report["score"] is None
    assert report["status"] == "pending_review"


def test_langevin_log_does_not_require_com_removal_after_initialization(tmp_path):
    manifest, rows, restart = _scalar_submission(tmp_path, 6)
    atoms = read(restart)
    atoms.set_momenta(atoms.get_momenta() + np.array([0.01, 0, 0]))
    write(restart, atoms)
    rows[1]["com_momentum"] = [0, 0, 0]
    rows[-2]["com_momentum"] = atoms.get_momenta().sum(axis=0).tolist()
    rows[-2]["ekin_eV"] = atoms.get_kinetic_energy()
    rows[-2]["temperature_K"] = atoms.get_temperature()
    (tmp_path / "run.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    assert _evaluate(manifest, 6, RestartVerifier())["score"] == 1


def test_restart_com_must_match_explicit_final_sample(tmp_path):
    manifest, rows, _ = _scalar_submission(tmp_path, 6)
    rows[-2]["com_momentum"] = [10, 0, 0]
    (tmp_path / "run.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    report = _evaluate(manifest, 6, RestartVerifier())
    assert report["score"] == 0
    assert "center-of-mass" in report["checks"][0]["detail"]
