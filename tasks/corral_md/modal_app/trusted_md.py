"""Controlled task-10 dynamics: JSON parameters in, observed states out.

The process never executes supplied code or loads a submitted checkpoint. Its
record is collected by the controller before any agent can edit the outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

# -I excludes script directories. This directory belongs to the pinned image.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

TARGETS = [300, 400, 500, 600, 700, 800, 900, 300]


def validate_config(value):
    allowed = {"random_seed", "timestep_fs", "friction_fs", "default_dtype", "stages"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Only declarative MD parameters are accepted")
    seed = value.get("random_seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random_seed must be an integer in [0, 2**32)")
    result = {"random_seed": seed}
    for name, default, maximum in (
        ("timestep_fs", 1.0, 10.0),
        ("friction_fs", 0.01, 10.0),
    ):
        v = value.get(name, default)
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= maximum:
            raise ValueError(f"Invalid {name}")
        result[name] = float(v)
    dtype = value.get("default_dtype", "float64")
    if dtype not in {"float64", "float32"}:
        raise ValueError("Invalid default_dtype")
    result["default_dtype"] = dtype
    stages = value.get(
        "stages",
        [
            {
                "id": f"stage_{i}",
                "target_temperature_K": t,
                "equilibration_steps": 4000,
                "production_steps": 1000,
                "sample_interval": 1,
                "equilibration_interval": 10,
            }
            for i, t in enumerate(TARGETS)
        ],
    )
    keys = {
        "id",
        "target_temperature_K",
        "equilibration_steps",
        "production_steps",
        "sample_interval",
        "equilibration_interval",
    }
    if not isinstance(stages, list) or len(stages) != 8:
        raise ValueError("The complete eight-stage temperature cycle is required")
    ids, count = set(), 0
    for stage, target in zip(stages, TARGETS, strict=True):
        if not isinstance(stage, dict) or set(stage) != keys:
            raise ValueError("Invalid stage parameters")
        name = stage["id"]
        if not isinstance(name, str) or not name or len(name) > 64 or name in ids:
            raise ValueError("Stage identifiers must be distinct nonempty strings")
        ids.add(name)
        if stage["target_temperature_K"] != target:
            raise ValueError("Temperature sequence must be 300..900 K then 300 K")
        for key in keys - {"id", "target_temperature_K"}:
            if type(stage[key]) is not int or not 1 <= stage[key] <= 1000000:
                raise ValueError(f"Invalid {key}")
        if (
            stage["equilibration_steps"] % stage["equilibration_interval"]
            or stage["production_steps"] % stage["sample_interval"]
        ):
            raise ValueError("Sampling intervals must divide their phase lengths")
        if stage["equilibration_steps"] // stage["equilibration_interval"] < 2:
            raise ValueError("Retain at least two equilibration samples")
        count += stage["equilibration_steps"] + stage["production_steps"]
    if count > 1000000:
        raise ValueError("Controlled run exceeds one million steps")
    result["stages"] = stages
    return result


def state_digest(atoms):
    digest = hashlib.sha256()
    for values, dtype in (
        (atoms.numbers, "<i8"),
        (atoms.pbc, "u1"),
        (atoms.cell.array, "<f8"),
        (atoms.positions, "<f8"),
        (atoms.get_masses(), "<f8"),
        (atoms.get_momenta(), "<f8"),
    ):
        digest.update(np.asarray(values, dtype=dtype).tobytes())
    return digest.hexdigest()


def run(config, destination, calculator=None):
    from ase import units
    from ase.build import bulk
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    config = validate_config(config)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if calculator is None:
        from verification_worker import mace_calculator

        assets = json.loads(Path("/opt/corral-md/assets.json").read_text())
        calculator = mace_calculator(
            {"model": "teacher.model", "default_dtype": config["default_dtype"]},
            assets,
            Path("/evidence"),
        )
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    atoms.calc = calculator
    rng = np.random.RandomState(config["random_seed"])
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=rng, force_temp=True)
    Stationary(atoms)
    dynamics = Langevin(
        atoms,
        config["timestep_fs"] * units.fs,
        temperature_K=300,
        friction=config["friction_fs"] / units.fs,
        fixcm=False,
        rng=rng,
    )
    record = {
        "schema": 1,
        "runner": "corral.task10.langevin.v1",
        "config": config,
        "velocity_initializations": 1,
        "manual_velocity_resets": 0,
        "stages": [],
        "initial_state_sha256": state_digest(atoms),
    }
    settings = {
        "model": "MACE-MP-0",
        "model_settings": {
            "default_dtype": config["default_dtype"],
            "dispersion": False,
        },
        "md": {
            "thermostat": "Langevin",
            "ensemble": "NVT",
            "timestep_fs": config["timestep_fs"],
            "friction_fs": config["friction_fs"],
            "random_seed": config["random_seed"],
            "initial_temperature_K": 300,
            "velocity_initializations": 1,
            "manual_velocity_resets": 0,
            "temperature_dof": 324,
        },
        "stages": [],
        "fit": {"method": "ols"},
        "structure": {"method": "coordination", "cutoff_A": 3.3},
    }
    columns = [
        "stage",
        "time_fs",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "total_energy_eV",
        "temperature_K",
        "temperature_x_K",
        "temperature_y_K",
        "temperature_z_K",
    ]
    with (
        (destination / "thermal_trace.csv").open("w") as trace_file,
        (destination / "equilibration_trace.csv").open("w") as eq_file,
        Trajectory(str(destination / "initial.traj"), "w") as initial,
        Trajectory(str(destination / "boundaries.traj"), "w") as boundaries,
        Trajectory(str(destination / "production.traj"), "w") as production,
    ):
        trace = csv.DictWriter(trace_file, fieldnames=columns)
        eq_trace = csv.DictWriter(
            eq_file,
            fieldnames=["stage", "target_K", "global_step", "kinetic_temperature_K"],
        )
        trace.writeheader()
        eq_trace.writeheader()

        def snapshot(stage):
            energy, forces = atoms.get_potential_energy(), atoms.get_forces()
            saved = atoms.copy()
            saved.calc = SinglePointCalculator(saved, energy=energy, forces=forces)
            saved.info.update(
                stage=stage["id"],
                step=dynamics.nsteps,
                time_fs=dynamics.nsteps * config["timestep_fs"],
                target_temperature_K=stage["target_temperature_K"],
                thermostat="Langevin",
                model="MACE-MP-0",
                ensemble="NVT",
                friction_fs=config["friction_fs"],
            )
            kinetic_axes = (
                np.sum(atoms.get_momenta() ** 2 / atoms.get_masses()[:, None], axis=0)
                / 2
            )
            temps = 2 * kinetic_axes / (len(atoms) * units.kB)
            row = dict(
                zip(
                    columns,
                    [
                        stage["id"],
                        saved.info["time_fs"],
                        energy,
                        float(kinetic_axes.sum()),
                        energy + float(kinetic_axes.sum()),
                        float(temps.mean()),
                        *temps.tolist(),
                    ],
                    strict=True,
                )
            )
            trace.writerow(row)
            return saved, float(temps.mean())

        for i, stage in enumerate(config["stages"]):
            dynamics.set_temperature(temperature_K=stage["target_temperature_K"])
            start_step = dynamics.nsteps
            start_hash = state_digest(atoms)
            saved, _ = snapshot(stage)
            boundaries.write(saved)
            if i == 0:
                initial.write(saved)
            for _ in range(
                stage["equilibration_steps"] // stage["equilibration_interval"]
            ):
                dynamics.run(stage["equilibration_interval"])
                saved, temperature = snapshot(stage)
                eq_trace.writerow(
                    {
                        "stage": stage["id"],
                        "target_K": stage["target_temperature_K"],
                        "global_step": dynamics.nsteps,
                        "kinetic_temperature_K": temperature,
                    }
                )
            eq_step, eq_hash = dynamics.nsteps, state_digest(atoms)
            with Trajectory(
                str(destination / f"stage_{i}.traj"), "w"
            ) as stage_trajectory:
                # Both the current artifact adapter and the proposal accept this boundary frame.
                production.write(saved)
                stage_trajectory.write(saved)
                for _ in range(stage["production_steps"] // stage["sample_interval"]):
                    dynamics.run(stage["sample_interval"])
                    saved, _ = snapshot(stage)
                    production.write(saved)
                    stage_trajectory.write(saved)
            boundaries.write(saved)
            record["stages"].append(
                {
                    "id": stage["id"],
                    "start_step": start_step,
                    "production_start_step": eq_step,
                    "end_step": dynamics.nsteps,
                    "start_state_sha256": start_hash,
                    "equilibration_end_state_sha256": eq_hash,
                    "production_start_state_sha256": eq_hash,
                    "end_state_sha256": state_digest(atoms),
                }
            )
            settings["stages"].append(
                {
                    "id": stage["id"],
                    "target_temperature_K": stage["target_temperature_K"],
                    "start_time_fs": start_step * config["timestep_fs"],
                    "end_time_fs": dynamics.nsteps * config["timestep_fs"],
                    "production_start_time_fs": eq_step * config["timestep_fs"],
                    "production_end_time_fs": dynamics.nsteps * config["timestep_fs"],
                }
            )
    record["total_steps"] = dynamics.nsteps
    (destination / "settings.json").write_text(
        json.dumps(settings, allow_nan=False, indent=2)
    )
    (destination / "observed.json").write_text(
        json.dumps(record, allow_nan=False, indent=2)
    )
    return record


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
