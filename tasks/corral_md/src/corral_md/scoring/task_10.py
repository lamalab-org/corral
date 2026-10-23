"""Heat capacity and return state from one recorded temperature-cycle log.

No submitted code is executed and dynamics are never rerun. Sampled independent
energy checks establish model consistency, not execution attestation. Sampling
screens are evaluator policy requiring calibration on validated production runs.
"""

from __future__ import annotations

import json
import secrets

import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from .common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    close,
    finite_array,
    result_close,
)
from .level1_trusted import PINNED_TEACHER_SHA256
from .verification import Plan

TARGETS = (300, 400, 500, 600, 700, 800, 900, 300)


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read_log(evidence, level):
    """Read complete, uniformly sampled stages with continuous boundary states."""
    stages, current = [], None
    with evidence.artifact("run_log").open() as stream:
        header = json.loads(next(stream))
        _require(
            header["event"] == "start"
            and header["model_sha256"] == PINNED_TEACHER_SHA256,
            "Use the supplied MACE-MP-0 model",
        )
        _require(header["numbers"] == [13] * 108, "Expected Al108")
        masses = finite_array(header["masses_amu"], shape=(108,))
        _require(
            close(masses, np.full(108, 26.9815385), atol=1e-3, rtol=0),
            "Incorrect aluminum masses",
        )
        _require(
            type(header["random_seed"]) is int, "Record an integer initialization seed"
        )
        _require(
            header["units"]
            == {"time": "fs", "position": "Angstrom", "energy": "eV", "momenta": "ASE"},
            "Incorrect log units",
        )
        for line in stream:
            row = json.loads(line)
            if row["event"] == "stage":
                _require(current is None, "Unfinished stage")
                current = {**row, "samples": []}
            elif row["event"] == "sample":
                _require(current is not None, "Sample outside a stage")
                current["samples"].append(row)
            elif row["event"] == "end":
                _require(current is not None, "End outside a stage")
                current["end"] = row
                stages.append(current)
                current = None
            else:
                raise EvidenceError("Unexpected log event")
    _require(
        current is None and len(stages) == (1 if level == 1 else 8),
        "Incomplete temperature cycle or extra stages",
    )
    offset, cell, previous = 0.0, None, None
    for index, stage in enumerate(stages):
        _require(
            type(stage["stage"]) is int
            and stage["stage"] == index
            and close(stage["target_temperature_K"], TARGETS[index], rtol=0, atol=1e-8),
            "Incorrect stage order or temperature target",
        )
        _require(
            stage["integrator"]
            in {"Langevin", "NVTBerendsen", "NoseHooverChainNVT", "Bussi", "Andersen"},
            "Use fixed-cell NVT dynamics",
        )
        eq, prod, interval = [
            stage[k] for k in ("equilibration_steps", "production_steps", "interval")
        ]
        _require(
            all(type(v) is int and v > 0 for v in (eq, prod, interval))
            and eq % interval == 0
            and prod % interval == 0,
            "Invalid sampling schedule",
        )
        dt = float(finite_array(stage["timestep_fs"], shape=()))
        _require(
            0 < dt <= 2
            and eq * dt >= 2000
            and prod * dt >= 3000
            and interval * dt <= 100
            and prod // interval >= 32,
            "Use <=2 fs steps, >=2 ps equilibration, >=3 ps production and >=32 production samples at <=100 fs",
        )
        rows = stage.pop("samples")
        steps = np.arange((eq + prod) // interval + 1) * interval
        _require(
            len(rows) == len(steps)
            and all(type(r["step"]) is int for r in rows)
            and np.array_equal([r["step"] for r in rows], steps),
            "Missing, duplicate, or unordered samples",
        )
        time = finite_array([r["time_fs"] for r in rows], shape=(len(rows),))
        _require(
            close(time, offset + steps * dt, rtol=0, atol=1e-6)
            and stage["end"]["stage"] == index
            and stage["end"]["step"] == eq + prod
            and close(stage["end"]["time_fs"], time[-1], rtol=0, atol=1e-6),
            "Truncated or inconsistent cumulative times",
        )
        cells = finite_array([r["cell_A"] for r in rows], shape=(len(rows), 3, 3))
        if cell is None:
            cell = cells[0]
            _require(
                close(cell @ cell.T, np.eye(3) * (3 * 4.05) ** 2, rtol=0, atol=1e-5),
                "Incorrect initial FCC cell",
            )
        _require(
            np.allclose(cells, cell, rtol=0, atol=1e-7), "Cell changed during the cycle"
        )
        positions = finite_array(
            [r["positions_A"] for r in rows], shape=(len(rows), 108, 3)
        )
        momenta = finite_array([r["momenta"] for r in rows], shape=(len(rows), 108, 3))
        potential = finite_array(
            [r["potential_energy_eV"] for r in rows], shape=(len(rows),)
        )
        if previous is not None:
            delta = (positions[0] - previous["positions"][-1]) @ np.linalg.inv(cell)
            _require(
                close(delta - np.rint(delta), np.zeros((108, 3)), rtol=0, atol=1e-7)
                and close(momenta[0], previous["momenta"][-1], rtol=0, atol=1e-7)
                and close(potential[0], previous["potential"][-1], rtol=0, atol=1e-6),
                "Positions, momenta, or energy reset at a stage boundary",
            )
        directional = (momenta**2 / masses[None, :, None]).sum(axis=1) / (
            108 * units.kB
        )
        kinetic = directional.sum(axis=1) * (108 * units.kB / 2)
        stage.update(
            time=time,
            positions=positions,
            momenta=momenta,
            potential=potential,
            cell=cell,
            masses=masses,
            production=steps > eq,
            temperature=directional.mean(axis=1),
            directional=directional,
            energy=(potential + kinetic) / 108,
        )
        previous, offset = stage, time[-1]
    first = stages[0]
    initial = Atoms(
        "Al108", positions=first["positions"][0], cell=cell, pbc=True, masses=masses
    )
    shells = np.sort(initial.get_all_distances(mic=True), axis=1)
    _require(
        np.allclose(shells[:, 1:13], 4.05 / np.sqrt(2), rtol=0, atol=1e-5)
        and np.allclose(shells[:, 13:19], 4.05, rtol=0, atol=1e-5),
        "Initialize from the prescribed conventional FCC crystal",
    )
    MaxwellBoltzmannDistribution(
        initial, temperature_K=300, rng=np.random.default_rng(header["random_seed"])
    )
    Stationary(initial, preserve_temperature=True)
    _require(
        close(initial.get_momenta(), first["momenta"][0], rtol=0, atol=1e-7),
        "Initial momenta do not match the recorded seed and COM removal",
    )
    return stages


def stage_statistics(stage):
    production = stage["production"]
    target = stage["target_temperature_K"]
    temperature = stage["directional"][production]
    tail = stage["directional"][~production]
    tail = tail[len(tail) // 2 :]
    blocks = np.array([part.mean(axis=0) for part in np.array_split(temperature, 4)])
    _require(
        np.max(np.abs(tail.mean(axis=0) - target)) <= 0.15 * target
        and np.max(np.abs(temperature.mean(axis=0) - target)) <= 0.15 * target
        and abs(temperature.mean() - target) <= 0.1 * target
        and np.max(np.ptp(blocks, axis=0)) <= 0.2 * target
        and np.max(blocks.std(axis=0, ddof=1) / 2) <= 0.03 * target,
        f"Stage {stage['stage']}: overall/directional temperatures are not settled or sufficiently sampled",
    )
    energy = stage["energy"][production]
    energy_blocks = np.array([part.mean() for part in np.array_split(energy, 4)])
    energy_se = float(energy_blocks.std(ddof=1) / 2)
    _require(
        energy_se <= 0.15 * units.kB * target
        and np.ptp(energy_blocks) <= 0.6 * units.kB * target
        and abs(energy[: len(energy) // 2].mean() - energy[len(energy) // 2 :].mean())
        <= 0.3 * units.kB * target,
        f"Stage {stage['stage']}: production energy is drifting or insufficiently resolved",
    )
    delta = np.diff(stage["positions"][production], axis=0) @ np.linalg.inv(
        stage["cell"]
    )
    motion = np.linalg.norm((delta - np.rint(delta)) @ stage["cell"], axis=(1, 2))
    _require(
        np.mean(motion > 1e-5) >= 0.9, "Production samples do not show resolved motion"
    )
    return {
        "temperature_K": float(temperature.mean()),
        "directional_temperature_K": temperature.mean(axis=0).tolist(),
        "energy_eV_per_atom": float(energy.mean()),
        "energy_se_eV_per_atom": energy_se,
    }


def nearest_neighbor_mean(stage):
    """Mean per-atom nearest-neighbor distance, using the same MIC metric twice."""
    values = []
    cell = stage["cell"]
    inverse = np.linalg.inv(cell)
    for positions in stage["positions"][stage["production"]]:
        delta = (positions[:, None, :] - positions[None, :, :]) @ inverse
        distances = np.linalg.norm((delta - np.rint(delta)) @ cell, axis=2)
        np.fill_diagonal(distances, np.inf)
        values.append(distances.min(axis=1).mean())
    return float(np.mean(values))


def cycle_statistics(stages, summaries, fit):
    x = np.array([s["temperature_K"] for s in summaries[:5]])
    y = np.array([s["energy_eV_per_atom"] for s in summaries[:5]])
    method = fit.get("method", "ols")
    _require(method in {"ols", "wls", "theil_sen"}, "Use ols, wls, or theil_sen")
    if method == "theil_sen":
        slope = np.median(
            [
                (y[j] - y[i]) / (x[j] - x[i])
                for i in range(5)
                for j in range(i + 1, 5)
                if x[j] != x[i]
            ]
        )
        intercept = np.median(y - slope * x)
    else:
        weights = (
            finite_array(fit["weights"], shape=(5,)) if method == "wls" else np.ones(5)
        )
        _require(np.all(weights > 0), "Fit weights must be positive")
        weights = weights / weights.max()
        xm, ym = np.average(x, weights=weights), np.average(y, weights=weights)
        variance = np.sum(weights * (x - xm) ** 2)
        _require(variance > 1e-12, "Fit weights do not resolve a temperature slope")
        slope = np.sum(weights * (x - xm) * (y - ym)) / variance
        intercept = ym - slope * xm
    initial, returning = (
        nearest_neighbor_mean(stages[0]),
        nearest_neighbor_mean(stages[-1]),
    )
    return {
        "heat_capacity": {
            "eV_per_atom_K": float(slope),
            "kB_per_atom": float(slope / units.kB),
            "intercept_eV_per_atom": float(intercept),
        },
        "high_temperature_residuals_eV_per_atom": [
            float(s["energy_eV_per_atom"] - (slope * s["temperature_K"] + intercept))
            for s in summaries[5:7]
        ],
        "return_state": {
            "energy_difference_eV_per_atom": summaries[-1]["energy_eV_per_atom"]
            - summaries[0]["energy_eV_per_atom"],
            "initial_nearest_neighbor_A": initial,
            "returning_nearest_neighbor_A": returning,
            "nearest_neighbor_difference_A": returning - initial,
        },
    }


def evaluate_submission(submission, *, level, verifier=None, review=None):
    rubric = Rubric(10, binary=True, fail_fast=True)
    computed, stages, verification = {}, [], {}
    try:
        evidence = Evidence(submission)
        fingerprint = evidence.fingerprint()
    except Exception as exc:
        rubric.check("calculation_log", 100, False, str(exc))
        return {**rubric.as_dict(), "level": level, "computed": computed}

    def recorded():
        stages.extend(read_log(evidence, level))
        computed["stages"] = [stage_statistics(s) for s in stages]
        return True

    rubric.check("recorded_temperature_cycle", 40, recorded)

    def answers():
        def compare_claim(claim, expected, key, optional=()):
            if isinstance(expected, dict):
                _require(isinstance(claim, dict), f"{key} must be an object")
                for name, value in expected.items():
                    if name == "energy_se_eV_per_atom":
                        continue
                    if name in optional and name not in claim:
                        continue
                    compare_claim(claim[name], value, name)
            else:
                atol = (
                    0.1
                    if "temperature_K" in key
                    else 1e-8
                    if key == "eV_per_atom_K"
                    else 1e-5
                )
                _require(
                    result_close(claim, expected, atol=atol),
                    f"Incorrect {key}; recomputed {expected}",
                )

        if "stages" in evidence.results:
            claimed = evidence.results["stages"]
            _require(
                isinstance(claimed, list) and len(claimed) == len(stages),
                "Report each stage in cycle order",
            )
            for claim, expected in zip(claimed, computed["stages"], strict=True):
                compare_claim(claim, expected, "stages")
        if level == 2:
            computed.update(
                cycle_statistics(
                    stages, computed["stages"], evidence.results.get("fit", {})
                )
            )
            for key, optional in (
                ("heat_capacity", ("intercept_eV_per_atom", "kB_per_atom")),
                ("high_temperature_residuals_eV_per_atom", ()),
                (
                    "return_state",
                    ("initial_nearest_neighbor_A", "returning_nearest_neighbor_A"),
                ),
            ):
                if key != "heat_capacity" and key not in evidence.results:
                    continue
                compare_claim(evidence.results[key], computed[key], key, optional)
        return True

    rubric.check("recomputed_answers", 40, answers)

    def independent():
        if verifier is None:
            raise UnsupportedEvidence(
                "Small independent energy checks require a local or Modal backend"
            )
        plan = Plan(evidence, 10, secrets.token_hex(32))
        for index, stage in enumerate(stages):
            selected = np.flatnonzero(stage["production"])
            indices = selected[plan.sample(len(selected), 3, f"energy_{index}")]
            frames = [
                Atoms(
                    "Al108",
                    positions=stage["positions"][i],
                    cell=stage["cell"],
                    pbc=True,
                    masses=stage["masses"],
                )
                for i in indices
            ]
            plan.frames(
                f"energy_{index}",
                frames,
                count=3,
                values=[{"energy": float(stage["potential"][i])} for i in indices],
                parameters={
                    "model": "teacher.model",
                    "default_dtype": "float64",
                    "dispersion": False,
                },
            )
        checks, _, _ = verifier._evaluate_plan(plan, fingerprint)
        verification["checks"] = checks
        _require(
            all(c["status"] != "failed" for c in checks),
            "Independent model energies disagree with the log",
        )
        if len(checks) != len(plan.jobs) or any(
            c["status"] != "passed" for c in checks
        ):
            raise UnsupportedEvidence("Independent energy verification is incomplete")
        return True

    rubric.check("independent_model_calculation", 20, independent)
    if evidence.fingerprint() != fingerprint:
        raise RuntimeError("Submission changed during grading; retry with frozen logs")
    if review is not None:
        rubric.apply_review(review, fingerprint)
    return {
        **rubric.as_dict(),
        "level": level,
        "mode": "saved_logs",
        "computed": computed,
        "evidence_sha256": fingerprint,
        "verification": verification,
    }
