"""Known equations of state exercise Task 7 without rerunning MACE."""

import json

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from corral_md.scoring import (
    check_level1_workflow,
    check_level2_workflow,
)
from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256
from corral_md.scoring.physical_reference import ALUMINUM_EXPANSION_PER_K


def write_log(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def submission(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    v0 = atoms.get_volume()
    rng = np.random.default_rng(42)
    p = rng.normal(size=(108, 3))
    p -= p.mean(axis=0)
    p *= np.sqrt(324 * units.kB * 300 / (p**2 / atoms.get_masses()[:, None]).sum())

    def state(volume, temperature, step):
        scale = (volume / v0) ** (1 / 3)
        displacement = 0.01 * np.sin(step + np.arange(324).reshape(108, 3))
        return {
            "cell_A": (atoms.cell.array * scale).tolist(),
            "positions_A": (
                atoms.positions * scale + (displacement if step else 0)
            ).tolist(),
            "momenta": (p * np.sqrt(temperature / 300)).tolist(),
        }

    def volume(temperature):
        # The first heating stage measures 301 K; its volume is the denominator
        # of the expansion coefficient. Keep the analytic EOS at the true target.
        return v0 * (1 + ALUMINUM_EXPANSION_PER_K * (temperature - 301))

    tc, vc = 401.0, volume(401)
    modulus, pressure_beta = 70, ALUMINUM_EXPANSION_PER_K
    specs = [("heating", t, volume(t + 1)) for t in (300, 380, 400, 420, 500)]
    specs += [("cooling", t, volume(t + 1) + 0.5) for t in (400, 300)]
    specs += [
        ("pressure", t, v)
        for t, v in (
            (380, vc),
            (400, vc),
            (420, vc),
            (400, vc * 0.99),
            (400, vc * 1.01),
        )
    ]
    rows = [
        {
            "event": "start",
            "model_sha256": PINNED_TEACHER_SHA256,
            "random_seed": 42,
            "numbers": atoms.numbers.tolist(),
            "masses_amu": atoms.get_masses().tolist(),
            "units": {
                "time": "fs",
                "cell": "Angstrom",
                "energy": "eV",
                "pressure": "GPa",
                "momenta": "ASE",
            },
        }
    ]
    previous, offset = state(v0, 300, 0), 0
    for index, (branch, target, mean_volume) in enumerate(specs):
        npt = branch != "pressure"
        start = previous if npt else state(mean_volume, target + 1, index)
        rows.append(
            {
                "event": "stage",
                "ensemble": "NPT" if npt else "NVT",
                "branch": branch,
                "integrator": "NPTBerendsen" if npt else "Langevin",
                "target_temperature_K": target,
                "target_pressure_GPa": 1.01325e-4 if npt else None,
                "timestep_fs": 1,
                "equilibration_steps": 2000,
                "production_steps": 4000,
                "interval": 50,
                "state": start,
            }
        )
        # Independent, bounded fluctuations; each production mean is exact.
        noise = rng.normal(size=(80, 3))
        noise -= noise.mean(axis=0)
        for step in range(0, 6001, 50):
            fluctuation = noise[(step // 50 - 41) % 80] if step > 2000 else np.zeros(3)
            temperature = target + 1 + fluctuation[0] * 0.2
            vol = mean_volume + (fluctuation[1] * 0.02 if npt else 0)
            pressure = (
                1.01325e-4
                if npt
                else modulus * pressure_beta * (target + 1 - tc)
                - modulus * (vol / vc - 1)
            )
            pressure += fluctuation[2] * 0.0001
            kinetic = 324 * units.kB * temperature / 2
            cell = state(vol, temperature, step)["cell_A"]
            if step == 0:
                cell = start["cell_A"]
                kinetic = (
                    np.array(start["momenta"]) ** 2 / atoms.get_masses()[:, None]
                ).sum() / 2
                temperature = 2 * kinetic / (324 * units.kB)
                vol = np.linalg.det(cell)
            rows.append(
                {
                    "event": "sample",
                    "step": step,
                    "time_fs": offset + step,
                    "cell_A": cell,
                    "kinetic_energy_eV": kinetic,
                    "temperature_K": temperature,
                    "potential_pressure_GPa": pressure
                    - 2 * kinetic / (3 * vol * units.GPa),
                    "pressure_GPa": pressure,
                }
            )
        previous = state(vol, temperature, index + 1)
        offset += 6000
        rows.append(
            {"event": "end", "step": 6000, "time_fs": offset, "state": previous}
        )
    write_log(tmp_path / "run.jsonl", rows)
    # Independent expected answers: polynomial mean volumes and analytic EOS.
    heating_t = np.array([301, 381, 401, 421, 501])
    slope = np.polyfit(heating_t, volume(heating_t), 1)[0]
    local = v0 * ALUMINUM_EXPANSION_PER_K / vc
    answer = {
        "artifacts": {"run_log": "run.jsonl"},
        "results": {
            "beta_volume": slope / volume(301),
            "beta_npt_local": local,
            "bulk_modulus_GPa": modulus,
            "beta_pressure": pressure_beta,
            "local_difference_K_inv": pressure_beta - local,
            "volume_differences_A3": {"300": 0.5, "400": 0.5},
        },
    }
    path = tmp_path / "answer.json"
    path.write_text(json.dumps(answer))
    return path


def test_cli_scores_log_submission(submission, tmp_path, monkeypatch):
    import sys

    from corral_md.scoring.evaluator import main

    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys, "argv", ["score", "7", str(submission), "--output", str(output)]
    )
    assert main() == 0
    assert json.loads(output.read_text())["score"] == 1


@pytest.mark.parametrize(
    "value", [None, {}, [], "", 123, {"results": {"beta_volume": float("nan")}}]
)
def test_malformed_submissions_score_zero(value):
    assert check_level2_workflow(7)(value) == 0


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("representation", ["nested", "aliased"])
def test_legacy_md_layouts_are_supported_only_at_level2(
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
    report = factory(7).evaluate(path)
    # Level 1 requires the documented scalar sample fields. This legacy log
    # instead supplies cell geometry and relies on inherited stage settings.
    assert report["score"] == (1 if level == 2 else 0), report
    assert report["status"] == "complete", report


def test_default_level2_final_quantity_policy_accepts_numerical_legacy_log(submission):
    from corral_md.scoring import check_level2_workflow

    report = check_level2_workflow(7, verification_backend="offline").evaluate(
        submission
    )
    assert report["score"] == 1, report
    assert report["scoring_policy"] == "level2-final-quantities-v2"
