"""Level 1 thermodynamic logs need no unpublished atom-wise trajectory schema."""

import json

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from corral_md.scoring import check_level1_workflow


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
    assert check_level1_workflow(6).evaluate(manifest)["score"] == 1


@pytest.mark.parametrize("task", [6, 10])
@pytest.mark.parametrize("dof_correction", [False, True])
def test_scalar_samples_do_not_need_a_restart(tmp_path, task, dof_correction):
    manifest, rows, restart = _scalar_submission(
        tmp_path, task, dof_correction=dof_correction
    )
    restart.unlink()
    report = check_level1_workflow(task).evaluate(manifest)
    assert report["score"] == 1, report
    assert "positions_A" not in rows[1]


@pytest.mark.parametrize("task", [6, 10])
@pytest.mark.parametrize("timestep_fs", [0.5, 2.0])
def test_scalar_samples_are_scored_using_their_physical_times(
    tmp_path, task, timestep_fs
):
    manifest, rows, _ = _scalar_submission(tmp_path, task)
    header = rows[0]
    header["timestep_fs"] = timestep_fs
    for key in ("sample_interval_steps", "nsteps", "equil_steps", "prod_steps"):
        if key in header:
            header[key] = int(header[key] / timestep_fs)
    for row in rows[1:-1]:
        row["step"] = int(row["step"] / timestep_fs)
    (tmp_path / "run.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    assert check_level1_workflow(task)(manifest) == 1
