"""Generate Task 2's 2600 K checkpoint using the pinned LAMMPS executable.

Historical task2-2600K-v2 audit helper; not the active Task 2 protocol.

This is a maintainer operation, not work repeated by benchmark agents. The
reference cools continuously from 4000 to 300 K. The resumed calculation uses
the same fix ID and global ramp endpoints to restore the Nose-Hoover state.
Outputs are retained for comparison before the checkpoint is bundled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

TASK_ROOT = Path(__file__).resolve().parents[1]
COMMON = """include pot.mod
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
timestep 1.0
thermo 1000
thermo_style custom step time temp press density pe ke etotal vol
thermo_modify format float %.15g
"""
COOLING = "fix cooling all npt temp 4000 300 100 iso 0 0 10000\n"
TAIL = """write_data cooling_end.dat nocoeff
write_restart cooling_end.restart
unfix cooling
fix hold all npt temp 300 300 100 iso 0 0 5000
run 100000
write_data hold_end.dat nocoeff
write_restart hold_end.restart
unfix hold
fix heating all npt temp 300 2600 100 iso 0 0 10000
run 230000
write_data reheating_end.dat nocoeff
write_restart reheating_end.restart
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(lammps, directory, name, contents):
    script = directory / name
    script.write_text(contents)
    with script.with_suffix(".stdout").open("w") as stream:
        subprocess.run(
            [str(lammps), "-in", name, "-log", name + ".log"],
            cwd=directory,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    executable = args.lammps.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    reference, resumed = root / "reference", root / "resumed"
    for directory in (reference, resumed):
        directory.mkdir(exist_ok=True)
        with ZipFile(TASK_ROOT / "potentials.zip") as archive:
            (directory / "pot.mod").write_bytes(archive.read("potentials/BKS/pot.mod"))
    with ZipFile(TASK_ROOT / "structures.zip") as archive:
        (reference / "liq4000.dat").write_bytes(
            archive.read("structures/melt/liq4000.dat")
        )
    version = subprocess.check_output([str(executable), "-h"], text=True)
    if "22 Jul 2025" not in version:
        raise ValueError("Use the benchmark's pinned stable_22Jul2025 LAMMPS build")
    (root / "lammps-version.txt").write_text(version)
    run(
        executable,
        reference,
        "reference.in",
        "units real\natom_style charge\nboundary p p p\nread_data liq4000.dat\n"
        + COMMON
        + COOLING
        + "restart 140000 checkpoint.*.restart\nrun 370000\nrestart 0\n"
        + "write_data cooling_end.dat nocoeff\n",
    )
    checkpoint = reference / "checkpoint.140000.restart"
    (resumed / "liq2600.restart").write_bytes(checkpoint.read_bytes())
    run(
        executable,
        resumed,
        "export.in",
        "units real\natom_style charge\nread_restart liq2600.restart\n"
        + COMMON
        + COOLING
        + "write_data liq2600.dat nocoeff\n",
    )
    run(
        executable,
        resumed,
        "resume.in",
        "units real\natom_style charge\nread_restart liq2600.restart\n"
        + COMMON
        + COOLING
        + "run 230000 start 0 stop 370000\n"
        + TAIL,
    )
    metadata = {
        "protocol_version": "task2-2600K-v2",
        "lammps_commit": "c7ae612a9497437412cb787b78769570f48653dd",
        "lammps_executable_sha256": sha256(executable),
        "source_sha256": sha256(reference / "liq4000.dat"),
        "potential_sha256": sha256(reference / "pot.mod"),
        "restart_sha256": sha256(resumed / "liq2600.restart"),
        "data_sha256": sha256(resumed / "liq2600.dat"),
        "checkpoint_step": 140000,
        "checkpoint_elapsed_time_ps": 140,
        "checkpoint_target_temperature_K": 2600,
        "original_ramp_K": [4000, 300],
        "original_ramp_steps": [0, 370000],
        "ramp_rate_K_ps": 10,
        "timestep_fs": 1,
        "thermostat_damping_fs": 100,
        "cooling_fix_id": "cooling",
        "ramp_pressure_damping_fs": 10000,
        "hold_pressure_damping_fs": 5000,
        "agent_stage_durations_ps": {"cooling": 230, "hold": 100, "reheating": 230},
        "preparation": "A checkpoint along the original cooling ramp, not an independently equilibrated 2600 K liquid.",
        "validation_status": "awaiting comparison of reference and resumed output",
    }
    (root / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(root)  # noqa: T201


if __name__ == "__main__":
    main()
