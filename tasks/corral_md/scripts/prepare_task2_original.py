"""Run Task 2's full original cooling/hold protocol, optionally with reheating.

This retains maintainer validation evidence; it does not establish a resolved Tg.
Use the LAMMPS build pinned in modal_app/assets.json.
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
PREPARATION = """units real
atom_style charge
boundary p p p
read_data liq4000.dat
include pot.mod
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
timestep 1.0
thermo 1000
thermo_style custom step time temp press density pe ke etotal vol
thermo_modify format float %.15g
fix cooling all npt temp 4000 300 100 iso 0 0 10000
run 370000
write_data cooling_end.dat nocoeff
write_restart cooling_end.restart
unfix cooling
fix hold all npt temp 300 300 100 iso 0 0 5000
run 100000
write_data hold_end.dat nocoeff
write_restart hold_end.restart
"""
REHEATING = """unfix hold
fix heating all npt temp 300 4000 100 iso 0 0 10000
run 370000
write_data reheating_end.dat nocoeff
write_restart reheating_end.restart
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    executable, output = args.lammps.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve prior evidence")
    version = subprocess.check_output([str(executable), "-h"], text=True)
    if "22 Jul 2025" not in version:
        raise ValueError("Use the benchmark's pinned stable_22Jul2025 LAMMPS build")
    (output / "lammps-version.txt").write_text(version)
    for archive_name, member, filename in (
        ("structures.zip", "structures/melt/liq4000.dat", "liq4000.dat"),
        ("potentials.zip", "potentials/BKS/pot.mod", "pot.mod"),
    ):
        with ZipFile(TASK_ROOT / archive_name) as archive:
            (output / filename).write_bytes(archive.read(member))
    (output / "run.in").write_text(PREPARATION + (REHEATING if args.level == 2 else ""))
    with (output / "console.log").open("w") as stream:
        subprocess.run(
            [str(executable), "-in", "run.in", "-log", "run.log"],
            cwd=output,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir()
        if p.is_file()
    }
    metadata = {
        "protocol_version": "task2-4000K-v3",
        "level": args.level,
        "lammps_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "files_sha256": files,
        "scientific_status": "Protocol execution only; a resolved Tg is not established.",
    }
    (output / "validation.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
