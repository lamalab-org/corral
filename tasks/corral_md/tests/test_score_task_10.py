"""Log-only heat-capacity scoring, independent checks, and state continuity."""

import copy
import json

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from corral_md.scoring import (
    WorkflowScorer,
)
from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256
from corral_md.scoring.task_10 import TARGETS
from corral_md.scoring.verification import ModalVerifier
from scipy.spatial import cKDTree


def write_log(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def verified(plan, fingerprint):
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


def scorer(level=2, transport=verified):
    return WorkflowScorer(10, level=level, verifier=ModalVerifier(transport=transport))


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
    for index, target in enumerate(TARGETS):
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


def test_complete_cycle_recomputes_measured_fit_and_nonzero_return(submission):
    path, _, _ = submission
    report = scorer().evaluate(path)
    assert report["score"] == 1, report
    values = report["computed"]
    assert values["heat_capacity"]["kB_per_atom"] == pytest.approx(3)
    assert values["stages"][0]["temperature_K"] == pytest.approx(306)
    assert values["high_temperature_residuals_eV_per_atom"] == pytest.approx(
        [0.01, -0.005]
    )
    assert values["return_state"]["energy_difference_eV_per_atom"] == pytest.approx(
        0.02
    )
    assert values["return_state"]["nearest_neighbor_difference_A"] < 0


def test_compact_cycle_answer_keeps_stage_and_structure_diagnostics(submission):
    path, manifest, rows = submission
    results = manifest["results"]
    manifest["results"] = {
        "heat_capacity": {"eV_per_atom_K": results["heat_capacity"]["eV_per_atom_K"]}
    }
    path.write_text(json.dumps(manifest))
    report = scorer().evaluate(path)
    assert report["score"] == 1, report
    assert len(report["computed"]["stages"]) == 8
    assert "intercept_eV_per_atom" in report["computed"]["heat_capacity"]
    assert len(report["computed"]["return_state"]) == 4
    assert "kB_per_atom" in report["computed"]["heat_capacity"]
    write_log(path.parent / "cycle.jsonl", rows[:-1])
    assert scorer().evaluate(path)["score"] == 0


def test_level1_needs_only_initial_stage_and_restartable_log(submission):
    path, manifest, rows = submission
    end = next(i for i, row in enumerate(rows) if row["event"] == "end")
    write_log(path.parent / "cycle.jsonl", rows[: end + 1])
    manifest.pop("results")
    path.write_text(json.dumps(manifest))
    assert scorer(1).evaluate(path)["score"] == 1
    assert scorer().evaluate(path)["score"] == 0


def test_level2_still_requires_heat_capacity(submission):
    path, manifest, _ = submission
    manifest.pop("results")
    path.write_text(json.dumps(manifest))
    assert scorer().evaluate(path)["score"] == 0


@pytest.mark.parametrize("method", ["ols", "wls", "theil_sen"])
def test_fit_strategies(submission, method):
    path, manifest, _ = submission
    manifest["results"]["fit"] = {"method": method, "weights": [1, 2, 3, 4, 5]}
    path.write_text(json.dumps(manifest))
    assert scorer().evaluate(path)["score"] == 1
    if method == "wls":
        manifest["results"]["fit"]["weights"] = [v * 1e307 for v in range(1, 6)]
        path.write_text(json.dumps(manifest))
        assert scorer().evaluate(path)["score"] == 1
        manifest["results"]["fit"]["weights"][0] = 0
        path.write_text(json.dumps(manifest))
        assert scorer().evaluate(path)["score"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "truncated",
        "missing_sample",
        "order",
        "cell",
        "reset",
        "time",
        "units",
        "model",
        "seed",
        "mass",
        "anisotropy",
        "energy_drift",
        "too_short",
        "static",
    ],
)
def test_bad_records_fail(submission, change):
    path, _, rows = submission
    stages = [i for i, r in enumerate(rows) if r["event"] == "stage"]
    sample = rows[stages[1] + 1]
    if change == "truncated":
        rows.pop()
    elif change == "missing_sample":
        rows.pop(20)
    elif change == "order":
        rows[stages[2]]["target_temperature_K"] = 600
    elif change == "cell":
        sample["cell_A"] = (np.array(sample["cell_A"]) * 1.001).tolist()
    elif change == "reset":
        sample["momenta"] = (-np.array(sample["momenta"])).tolist()
    elif change == "time":
        sample["time_fs"] = 0
    elif change == "units":
        rows[0]["units"]["energy"] = "J"
    elif change == "model":
        rows[0]["model_sha256"] = "0" * 64
    elif change == "seed":
        rows[0]["random_seed"] += 1
    elif change == "mass":
        rows[0]["masses_amu"][0] *= 2
    elif change == "too_short":
        rows[stages[0]]["production_steps"] = 100
    else:
        for row in rows[stages[0] + 1 : stages[1] - 2]:
            if row["step"] <= 2000:
                continue
            if change == "anisotropy":
                row["momenta"] = (
                    np.array(row["momenta"]) * [np.sqrt(2), np.sqrt(0.5), np.sqrt(0.5)]
                ).tolist()
            elif change == "energy_drift":
                row["potential_energy_eV"] += row["step"] / 20
            elif change == "static":
                row["positions_A"] = rows[2]["positions_A"]
    write_log(path.parent / "cycle.jsonl", rows)
    assert scorer().evaluate(path)["score"] == 0


@pytest.mark.parametrize(
    "keys",
    [
        ("stages", 0, "temperature_K"),
        ("stages", 3, "directional_temperature_K", 0),
        ("stages", 4, "energy_eV_per_atom"),
        ("heat_capacity", "eV_per_atom_K"),
        ("heat_capacity", "kB_per_atom"),
        ("heat_capacity", "intercept_eV_per_atom"),
        ("high_temperature_residuals_eV_per_atom", 0),
        ("return_state", "energy_difference_eV_per_atom"),
        ("return_state", "nearest_neighbor_difference_A"),
    ],
)
def test_altered_answers_fail(submission, keys):
    path, manifest, _ = submission
    target = manifest["results"]
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] *= 1.1
    path.write_text(json.dumps(manifest))
    assert scorer().evaluate(path)["score"] == 0


def test_offline_and_disagreeing_model_cannot_pass(submission):
    path, _, _ = submission
    offline = WorkflowScorer(10).evaluate(path)
    assert offline["status"] == "pending_review"
    assert offline["computed"]["heat_capacity"]["kB_per_atom"] == pytest.approx(3)

    def wrong(plan, fingerprint):
        response = copy.deepcopy(verified(plan, fingerprint))
        response["jobs"][0]["value"]["frames"][0]["energy"] += 1
        return response

    assert scorer(transport=wrong).evaluate(path)["score"] == 0


def test_missing_log_fails(tmp_path):
    path = tmp_path / "answer.json"
    path.write_text('{"artifacts": {}, "results": {}}')
    assert scorer().evaluate(path)["score"] == 0
