"""Scientific calculations in a disposable, network-restricted verifier sandbox.

Requests contain geometry and calculator inputs, never expected answers. Only
this installed entry point runs; agent scripts are not imported. A submitted
checkpoint gets its own sandbox, separate from trusted model calculations.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array(value, shape=None):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all() or (shape is not None and result.shape != shape):
        raise ValueError("Invalid numerical array")
    return result


def atoms_from_record(record):
    from ase import Atoms

    numbers = record["numbers"]
    if not isinstance(numbers, list) or not 1 <= len(numbers) <= 10000:
        raise ValueError("Invalid atom count")
    if any(type(z) is not int or not 1 <= z <= 118 for z in numbers):
        raise ValueError("Invalid atomic numbers")
    pbc = record["pbc"]
    if (
        not isinstance(pbc, list)
        or len(pbc) != 3
        or any(type(v) is not bool for v in pbc)
    ):
        raise ValueError("Explicit periodic boundary flags required")
    atoms = Atoms(
        numbers=numbers,
        positions=array(record["positions"], (len(numbers), 3)),
        cell=array(record["cell"], (3, 3)),
        pbc=pbc,
    )
    if any(pbc) and abs(np.linalg.det(atoms.cell)) < 1e-10:
        raise ValueError("Periodic cell is singular")
    if "masses" in record:
        masses = array(record["masses"], (len(atoms),))
        if np.any(masses <= 0):
            raise ValueError("Masses must be positive")
        atoms.set_masses(masses)
    return atoms


def mace_calculator(parameters, assets, evidence_root, *, cache=None):
    from mace.calculators import MACECalculator, mace_mp

    allowed = {
        "model",
        "default_dtype",
        "dispersion",
        "checkpoint",
        "checkpoint_sha256",
    }
    if set(parameters) - allowed:
        raise NotImplementedError("Unsupported MACE calculator settings")
    dtype = parameters.get("default_dtype", "float64")
    dispersion = parameters.get("dispersion", False)
    if dtype not in {"float32", "float64"} or type(dispersion) is not bool:
        raise ValueError("Invalid MACE precision or dispersion")
    model = parameters.get("model", "teacher.model")
    if model == "submitted":
        path = Path(evidence_root) / parameters["checkpoint"]
        if path.parent != Path(evidence_root) or path.is_symlink():
            raise ValueError("Checkpoint must be an isolated evidence file")
        expected = parameters["checkpoint_sha256"]
    else:
        if model not in assets["models"]:
            raise ValueError("Unknown pinned model")
        path = Path("/workspace/models") / model
        expected = assets["models"][model]["sha256"]
    if file_hash(path) != expected:
        raise ValueError("Model checkpoint digest differs")
    key = (expected, dtype, dispersion)
    if cache is not None and key in cache:
        return cache[key]
    if model == "submitted" and not dispersion:
        calculator = MACECalculator(
            model_paths=str(path), device="cuda", default_dtype=dtype
        )
    else:
        calculator = mace_mp(
            model=str(path), device="cuda", default_dtype=dtype, dispersion=dispersion
        )
    if cache is not None:
        cache[key] = calculator
    return calculator


def lammps_restart_state(parameters, evidence_root):
    """Read one submitted restart with the release-pinned LAMMPS executable.

    The submitted bytes are input data only. The command and conversion output
    path are fixed by this worker; no submitted LAMMPS input is executed.
    """
    if set(parameters) != {"checkpoint", "checkpoint_sha256"}:
        raise ValueError("Restart conversion needs only a checkpoint and its digest")
    name = parameters["checkpoint"]
    expected = parameters["checkpoint_sha256"]
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".restart")
        or not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None
    ):
        raise ValueError("Invalid restart filename or digest")
    root = Path(evidence_root)
    path = root / name
    if path.parent != root or path.is_symlink() or not path.is_file():
        raise ValueError("Restart must be a flat, regular evidence file")
    if path.stat().st_size > 64 * 1024 * 1024 or file_hash(path) != expected.lower():
        raise ValueError("Restart size or SHA-256 differs")

    from ase.io import read

    with tempfile.TemporaryDirectory(prefix="corral-restart-") as temporary:
        data = Path(temporary) / "converted.data"
        dump = Path(temporary) / "converted.dump"
        command = [
            "/usr/local/bin/lmp",
            "-log",
            "none",
            "-screen",
            "none",
            "-restart2data",
            str(path),
            str(data),
            "nocoeff",
        ]
        completed = subprocess.run(
            command,
            cwd=temporary,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not data.is_file():
            raise ValueError(
                "LAMMPS restart conversion failed: " + completed.stderr[-500:]
            )
        if data.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Converted state is too large")
        dumped = subprocess.run(
            [
                "/usr/local/bin/lmp",
                "-log",
                "none",
                "-screen",
                "none",
                "-restart2dump",
                str(path),
                "all",
                "custom",
                str(dump),
                "id",
                "type",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
            ],
            cwd=temporary,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if dumped.returncode != 0 or not dump.is_file():
            raise ValueError("LAMMPS restart dump failed: " + dumped.stderr[-500:])
        if dump.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Converted dump is too large")
        with dump.open() as snapshot:
            markers = [snapshot.readline().strip() for _ in range(5)]
        if (
            markers[0] != "ITEM: TIMESTEP"
            or markers[2] != "ITEM: NUMBER OF ATOMS"
            or not markers[4].startswith("ITEM: BOX BOUNDS ")
        ):
            raise ValueError("Converted dump lacks timestep, count, or boundaries")
        try:
            dump_step = int(markers[1])
            dump_count = int(markers[3])
        except ValueError as exc:
            raise ValueError("Converted dump has invalid timestep or count") from exc
        pbc = [value == "pp" for value in markers[4].split()[-3:]]
        if len(pbc) != 3 or not all(pbc):
            raise ValueError("Restart is not periodic in all directions")
        with data.open() as converted:
            header = converted.readline()
        match = re.search(
            r"\btimestep\s*=\s*(\d+)\b.*\bunits\s*=\s*metal\b", header
        )
        if match is None:
            raise ValueError("Converted state lacks a metal-unit timestep")
        if int(match.group(1)) != dump_step:
            raise ValueError("Data and dump conversions disagree on timestep")
        atoms = read(
            data,
            format="lammps-data",
            atom_style="full",
            units="metal",
            sort_by_id=True,
        )

    ids = atoms.arrays.get("id")
    velocities = atoms.get_velocities()
    masses = atoms.get_masses()
    if (
        len(atoms) != 216
        or dump_count != len(atoms)
        or set(atoms.get_chemical_symbols()) != {"Si"}
        or ids is None
        or len(np.unique(ids)) != 216
        or velocities is None
        or not np.all(np.isfinite(atoms.positions))
        or not np.all(np.isfinite(atoms.cell.array))
        or np.linalg.det(atoms.cell.array) <= 0
        or not np.all(np.isfinite(velocities))
        or not np.all(np.isfinite(masses))
        or not np.all(masses > 0)
    ):
        raise ValueError("Converted restart is not a finite 216-atom Si state")
    return {
        "checkpoint_sha256": expected.lower(),
        "state": {
            "numbers": atoms.numbers.tolist(),
            "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist(),
            "pbc": pbc,
            "atom_ids": ids.tolist(),
            "momenta": atoms.get_momenta().tolist(),
            "masses": masses.tolist(),
            "timestep": int(match.group(1)),
        },
    }


def calculate(job, assets, evidence_root, cache=None):
    operation = job["operation"]
    parameters = job.get("parameters", {})
    if operation == "lammps_restart":
        return lammps_restart_state(parameters, evidence_root)
    if operation == "pipeline":
        # This process and its entire sandbox are discarded after this one job.
        # There are no labels, expected predictions, scripts or credentials here.
        import pickle
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge

        path = Path(evidence_root) / parameters["checkpoint"]
        if (
            path.parent != Path(evidence_root)
            or file_hash(path) != parameters["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint digest/path differs")
        with path.open("rb") as stream:
            pipeline = pickle.load(stream)
        if not isinstance(pipeline, (Pipeline, Ridge)):
            raise NotImplementedError("Checkpoint is not a sklearn prediction pipeline")
        prediction = array(pipeline.predict(array(job["features"])))
        return {
            "predictions": prediction.tolist(),
            "checkpoint_sha256": file_hash(path),
        }

    records = job["frames"]
    if not isinstance(records, list) or not 1 <= len(records) <= 2000:
        raise ValueError("Invalid verification batch size")
    frames = [atoms_from_record(record) for record in records]
    if operation == "soap":
        from dscribe.descriptors import SOAP

        allowed = {
            "species",
            "r_cut",
            "n_max",
            "l_max",
            "sigma",
            "rbf",
            "weighting",
            "average",
            "compression",
            "periodic",
            "dtype",
            "sparse",
        }
        if set(parameters) - allowed:
            raise NotImplementedError("Unsupported SOAP settings")
        if parameters.get("sparse", False) or parameters.get("average") not in {
            "inner",
            "outer",
        }:
            raise NotImplementedError("Verifier expects dense global SOAP descriptors")
        descriptor = SOAP(**parameters)
        features = array(descriptor.create(frames, n_jobs=1))
        if len(frames) == 1 and features.ndim == 1:
            features = features[None, :]
        if features.ndim != 2 or features.shape[0] != len(frames):
            raise ValueError("SOAP output does not contain one row per frame")
        return {"features": features.tolist()}

    properties = job.get("properties", ["energy", "forces"])
    if not properties or set(properties) - {"energy", "forces", "stress", "hessian"}:
        raise ValueError("Invalid requested properties")
    postprocess = job.get("postprocess")
    if postprocess is not None and (
        properties != ["hessian"]
        or not isinstance(postprocess, list)
        or len(postprocess) != len(frames)
    ):
        raise ValueError("Hessian postprocessing must align with every input frame")
    if operation == "mace":
        calculator = mace_calculator(parameters, assets, evidence_root, cache=cache)
    elif operation == "sw":
        from ase.calculators.lammpsrun import LAMMPS

        if parameters or any(set(a.get_chemical_symbols()) != {"Si"} for a in frames):
            raise ValueError("SW verification uses only the supplied silicon potential")
        potential = Path("/workspace/potentials/SW/Si.sw")
        if not potential.is_file():
            raise FileNotFoundError("Pinned silicon SW potential is missing")
        calculator = LAMMPS(
            command="/usr/local/bin/lmp",
            files=[str(potential)],
            specorder=["Si"],
            pair_style="sw",
            pair_coeff=[f"* * {potential} Si"],
            units="metal",
            atom_style="atomic",
            keep_alive=False,
        )
    else:
        raise ValueError("Unknown verification operation")
    output = []
    for index, atoms in enumerate(frames):
        atoms.calc = calculator
        result = {}
        for name in properties:
            if name == "hessian":
                if operation != "mace" or not hasattr(calculator, "get_hessian"):
                    raise NotImplementedError(
                        "Calculator cannot evaluate analytical Hessians"
                    )
                value = calculator.get_hessian(atoms=atoms)
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                value = array(value).reshape(3 * len(atoms), 3 * len(atoms))
                if postprocess is not None:
                    from ground_truth import postprocess_hessian  # noqa: PLC0415

                    value = postprocess_hessian(value, postprocess[index])
                result[name] = value.tolist()
            else:
                function = {
                    "energy": atoms.get_potential_energy,
                    "forces": atoms.get_forces,
                    "stress": atoms.get_stress,
                }[name]
                result[name] = array(function()).tolist()
        output.append(result)
    return {"frames": output}


def main():
    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    assets = json.loads(Path("/opt/corral-md/assets.json").read_text())
    output, cache = [], {}
    for job in request["jobs"]:
        try:
            value = calculate(job, assets, request_path.parent, cache)
            output.append({"id": job["id"], "status": "complete", "value": value})
        except NotImplementedError as exc:
            output.append(
                {"id": job["id"], "status": "unsupported", "detail": str(exc)}
            )
        except Exception as exc:
            output.append(
                {
                    "id": job["id"],
                    "status": "error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    output_path.write_text(json.dumps({"jobs": output}, allow_nan=False))


if __name__ == "__main__":
    main()
