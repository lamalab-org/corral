"""Existing temperature-cycle logs remain readable by the quantity scorers."""

import json

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from corral_md.scoring import check_level1_workflow, check_level2_workflow
from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256
from scipy.spatial import cKDTree


def write_log(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def submission(tmp_path):
    """Analytic energy/temperature relation with nonzero high-T and return shifts."""
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    seed, rng = 17, np.random.default_rng(20)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.default_rng(seed)
    )
    Stationary(atoms, preserve_temperature=True)
    rows = [
        {
            "event": "start",
            "model_sha256": PINNED_TEACHER_SHA256,
            "random_seed": seed,
            "velocity_rng": "numpy.default_rng",
            "numbers": atoms.numbers.tolist(),
            "masses_amu": atoms.get_masses().tolist(),
            "units": {
                "time": "fs",
                "position": "Angstrom",
                "energy": "eV",
                "momenta": "ASE",
            },
        }
    ]
    answers, structural = [], []
    slope, intercept = 3 * units.kB, -3.4
    previous = None
    for index, target in enumerate((300, 400, 500, 600, 700, 800, 900, 300)):
        rows.append(
            {
                "event": "stage",
                "stage": index,
                "target_temperature_K": target,
                "integrator": "Langevin",
                "timestep_fs": 1.0,
                "equilibration_steps": 2000,
                "production_steps": 3200,
                "interval": 100,
            }
        )
        means, distances = [], []
        for step in range(0, 5201, 100):
            if step == 0 and previous is not None:
                sample = {**previous, "step": 0}
            else:
                p = rng.normal(size=(108, 3))
                p -= p.mean(axis=0)
                measured = target * 1.02 + 2 * np.sin(step / 100 * np.pi / 2)
                p *= np.sqrt(
                    108
                    * units.kB
                    * measured
                    / (p**2 / atoms.get_masses()[:, None]).sum(axis=0)
                )
                position = atoms.positions + rng.normal(
                    0, 0.02 if index < 7 else 0.06, (108, 3)
                )
                if index == step == 0:
                    position, p = atoms.positions, atoms.get_momenta()
                kinetic = (p**2 / atoms.get_masses()[:, None]).sum() / 2
                measured = 2 * kinetic / (324 * units.kB)
                shift = {5: 0.01, 6: -0.005, 7: 0.02}.get(index, 0)
                sample = {
                    "event": "sample",
                    "step": step,
                    "time_fs": index * 5200 + step,
                    "cell_A": atoms.cell.array.tolist(),
                    "positions_A": position.tolist(),
                    "momenta": p.tolist(),
                    "potential_energy_eV": 108 * (intercept + slope * measured + shift)
                    - kinetic,
                }
            rows.append(sample)
            if step > 2000:
                means.append([measured, intercept + slope * measured + shift])
                if index in (0, 7):
                    wrapped = position % (3 * 4.05)
                    distances.append(
                        cKDTree(wrapped, boxsize=3 * 4.05)
                        .query(wrapped, k=2)[0][:, 1]
                        .mean()
                    )
        previous = sample
        rows.append(
            {
                "event": "end",
                "stage": index,
                "step": 5200,
                "time_fs": (index + 1) * 5200,
            }
        )
        t, energy = np.mean(means, axis=0)
        answers.append(
            {
                "temperature_K": t,
                "directional_temperature_K": [t] * 3,
                "energy_eV_per_atom": energy,
            }
        )
        if distances:
            structural.append(float(np.mean(distances)))
    manifest = {
        "artifacts": {"run_log": "cycle.jsonl"},
        "results": {
            "stages": answers,
            "heat_capacity": {
                "eV_per_atom_K": slope,
                "kB_per_atom": 3,
                "intercept_eV_per_atom": intercept,
            },
            "high_temperature_residuals_eV_per_atom": [0.01, -0.005],
            "return_state": {
                "energy_difference_eV_per_atom": 0.02,
                "initial_nearest_neighbor_A": structural[0],
                "returning_nearest_neighbor_A": structural[1],
                "nearest_neighbor_difference_A": structural[1] - structural[0],
            },
        },
    }
    write_log(tmp_path / "cycle.jsonl", rows)
    path = tmp_path / "answer.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, rows


def test_missing_log_fails(tmp_path):
    path = tmp_path / "answer.json"
    path.write_text('{"artifacts": {}, "results": {}}')
    assert check_level2_workflow(10).evaluate(path)["score"] == 0


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("representation", ["nested", "aliased"])
def test_scientific_score_is_independent_of_md_log_layout(
    submission, rewrite_md_recording, level, representation
):
    path = submission[0] if isinstance(submission, tuple) else submission
    answer = json.loads(path.read_text())
    log = path.parent / answer["artifacts"]["run_log"]
    if level == 1:
        answer.pop("results", None)
        path.write_text(json.dumps(answer))
    rewrite_md_recording(log, representation, first_stage=level == 1)
    factory = check_level1_workflow if level == 1 else check_level2_workflow
    report = factory(10).evaluate(path)
    assert report["score"] == 1, report


def test_default_level2_final_quantity_policy_accepts_numerical_legacy_log(submission):
    from corral_md.scoring import check_level2_workflow

    report = check_level2_workflow(10, verification_backend="offline").evaluate(
        submission[0]
    )
    assert report["score"] == 1, report
    assert report["scoring_policy"] == "level2-final-quantities-v1"
