"""Scientific calculations in a disposable, network-restricted verifier sandbox.

Requests contain geometry and calculator inputs, never expected answers. Only
this installed entry point runs; agent scripts are not imported. A submitted
checkpoint gets its own sandbox, separate from trusted model calculations.
"""

from __future__ import annotations

import hashlib
import json
import sys
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


def calculate(job, assets, evidence_root, cache=None):
    operation = job["operation"]
    parameters = job.get("parameters", {})
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
