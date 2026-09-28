"""Independent MACE checks for the preparatory Level 1 workflows.

Only evaluator-selected geometry and calculator parameters go to the trusted
Modal worker. Submitted scripts and claimed answers remain local. The worker
uses the release-pinned teacher checkpoint, never a path supplied by an agent.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping

import numpy as np
from ase.build import bulk

from .common import EvidenceError, UnsupportedEvidence, finite_array
from .verification import ModalVerifier, Plan, stored

# This is the immutable teacher asset in modal_app/assets.json. Keep the value
# independent of submission metadata: a submitted digest is only a claim.
PINNED_TEACHER_SHA256 = (
    "01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845"
)
MODEL_TARGET = "independent_model_calculation"
MODEL_PARAMETERS = {
    "model": "teacher.model",
    "default_dtype": "float64",
    "dispersion": False,
}


def teacher_digest_matches(e, task_number: int) -> bool:
    """Check each recorded teacher digest against the evaluator's pinned asset.

    Tasks 3, 4, and 5 do not explicitly require a digest, so a missing digest
    is handled by independent numerical recalculation. Task 8 asks for it.
    """

    names = {
        3: ("teacher_sha256", "teacher_model_sha256"),
        4: ("checkpoint_sha256", "teacher_sha256", "model_sha256"),
        5: ("teacher_sha256", "model_sha256"),
        8: ("teacher_sha256", "teacher_model_sha256"),
    }.get(task_number)
    if names is None:
        raise ValueError("Teacher digest check applies to Tasks 3, 4, 5, and 8")
    if task_number == 4:
        checkpoint = str(e.settings.get("checkpoint", "")).strip().lower()
        checkpoint = checkpoint.replace("_", "-")
        if not (
            "teacher.model" in checkpoint
            or re.search(r"(?<![a-z0-9])mace-mp-0(?![a-z0-9])", checkpoint)
        ):
            return False
    claims = [e.settings[name] for name in names if name in e.settings]
    if task_number == 4 and "model_sha256" in e.results:
        claims.append(e.results["model_sha256"])
    if not claims:
        return task_number != 8
    return all(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
        and value.lower() == PINNED_TEACHER_SHA256
        for value in claims
    )


def replay_task8_default_rng(e) -> bool:
    """Replay the recorded NumPy generator on all 100 training geometries.

    This is deliberately limited to an identifiable ``default_rng`` stream.
    Another documented generator is an unsupported adapter, not a failed
    Gaussian requirement. Coordinates are compared after periodic wrapping and
    atom reordering so ordinary trajectory serialization does not cause zeros.
    """

    generation = e.settings["generation"]["train"]
    library = str(generation.get("library", "")).lower()
    method = str(generation.get("method", "")).lower()
    rng_name = str(generation.get("rng", "")).lower()
    if "numpy" not in library or not any(
        "default_rng" in value for value in (library, method, rng_name)
    ):
        raise UnsupportedEvidence(
            "The recorded generator is not NumPy default_rng; replay needs its adapter"
        )
    seed = generation.get("seed")
    if type(seed) is not int or seed < 0:
        raise EvidenceError("NumPy default_rng needs a nonnegative integer seed")
    data = e.json("regression_data", "datasets", "dataset_index")["train"]
    frames = e.trajectory("train_structures", "training_structures")
    if len(frames) != 100:
        raise EvidenceError("Expected exactly 100 training structures")
    indices = data["generation_index"]
    if (
        not isinstance(indices, list)
        or len(indices) != 100
        or any(type(index) is not int for index in indices)
        or sorted(indices) != list(range(100))
    ):
        raise EvidenceError("Generation indices must cover 0 through 99 exactly")
    row_ids = data.get("row_ids")
    frame_ids = [atoms.info.get("row_id") for atoms in frames]
    if isinstance(row_ids, list) and all(item is not None for item in frame_ids):
        if len(set(frame_ids)) != 100 or set(frame_ids) != set(row_ids):
            return False
        frame_lookup = dict(zip(frame_ids, frames, strict=True))
        ordered = [frame_lookup[item] for item in row_ids]
    else:
        ordered = frames
    frame_by_index = dict(zip(indices, ordered, strict=True))
    ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    cell = ideal.cell.array
    inverse_cell = np.linalg.inv(cell)
    rng = np.random.default_rng(seed)
    for index in range(100):
        atoms = frame_by_index[index]
        if len(atoms) != 64 or not np.array_equal(atoms.numbers, ideal.numbers):
            return False
        if not np.allclose(atoms.cell.array, cell, rtol=0, atol=1e-6):
            return False
        sigma = 0.01 + 0.001 * index
        expected = ideal.positions + rng.normal(0.0, sigma, size=(64, 3))
        actual = finite_array(atoms.positions, shape=(64, 3))
        # Find the ideal site of each saved atom before comparing RNG draws.
        displacement = actual[:, None, :] - ideal.positions[None, :, :]
        fractional = displacement @ inverse_cell
        fractional -= np.rint(fractional)
        distance = np.linalg.norm(fractional @ cell, axis=2)
        mapping = np.argmin(distance, axis=1)
        if len(set(mapping.tolist())) != 64:
            return False
        delta = (actual - expected[mapping]) @ inverse_cell
        delta -= np.rint(delta)
        if not np.all(np.linalg.norm(delta @ cell, axis=1) <= 1e-6):
            return False
    return True


def _model_parameters(e, task_number: int) -> dict:
    """Read declared precision while fixing the model to the trusted teacher."""

    configs: list[Mapping] = [e.settings]
    for key in ("model_settings", "calculator_args"):
        value = e.settings.get(key)
        if isinstance(value, Mapping):
            configs.append(value)
    if task_number == 8:
        value = e.settings.get("generation", {}).get("train", {})
        if isinstance(value, Mapping):
            configs.append(value)
    if task_number == 9:
        value = e.settings.get("runs", {}).get("main", {})
        if isinstance(value, Mapping):
            configs.append(value)
    types = {
        str(config[key])
        for config in configs
        for key in ("default_dtype", "dtype", "precision", "model_dtype")
        if key in config
    }
    if not types:
        dtype = "float64"
    elif len(types) == 1 and next(iter(types)) in {"float32", "float64"}:
        dtype = next(iter(types))
    else:
        raise UnsupportedEvidence("Conflicting or unsupported MACE precision settings")
    dispersion = task_number == 3
    declared_dispersion = {
        config["dispersion"] for config in configs if "dispersion" in config
    }
    if declared_dispersion and declared_dispersion != {dispersion}:
        raise UnsupportedEvidence(
            "Declared dispersion differs from the task calculator"
        )
    return {"model": "teacher.model", "default_dtype": dtype, "dispersion": dispersion}


def _samples(p: Plan, name: str, frames, *, count=5, values=None, properties=None):
    p.frames(
        name,
        frames,
        count=count,
        values=values,
        properties=properties,
        parameters=_model_parameters(p.evidence, p.task_number),
        targets=[MODEL_TARGET],
    )


def _task5_force_plan(p: Plan) -> None:
    from .task_5 import _array, _order, _records

    e = p.evidence
    bases = e.trajectory("strained_structures")
    records = _records(e, "strain_calculations")
    if len(bases) != 5 or len(records) != 5:
        raise EvidenceError("Task 5 requires all five strained structures")
    displaced, values = [], []
    hessian_frames, hessian_values = [], []
    for record in records:
        base = bases[record["structure_index"]]
        method = record["method"]
        if method == "analytic_hessian":
            matrix = _array(e, record["raw_derivatives_eV_A2"], (6, 6))
            if record["derivative_kind"] == "force_jacobian":
                matrix = -matrix
            elif record["derivative_kind"] != "energy_hessian":
                raise UnsupportedEvidence("Unknown analytical derivative convention")
            order = _order(record)
            standard = np.empty((6, 6))
            standard[np.ix_(order, order)] = matrix
            hessian_frames.append(base)
            hessian_values.append({"hessian": standard.tolist()})
            continue
        if method == "central_difference":
            displacements = np.concatenate(
                [
                    _array(e, record["plus_displacements_A"]),
                    _array(e, record["minus_displacements_A"]),
                ]
            )
            forces = np.concatenate(
                [
                    _array(e, record["plus_forces_eV_A"]),
                    _array(e, record["minus_forces_eV_A"]),
                ]
            )
        elif method == "finite_difference":
            displacements = _array(e, record["displacements_A"])
            forces = _array(e, record["forces_eV_A"])
        else:
            raise UnsupportedEvidence(f"Unsupported Task 5 method: {method!r}")
        if displacements.shape != forces.shape or displacements.shape[1:] != (2, 3):
            raise EvidenceError("Task 5 displacement and force arrays must align")
        for delta, force in zip(displacements, forces, strict=True):
            atoms = base.copy()
            atoms.positions += delta
            displaced.append(atoms)
            values.append({"forces": force.tolist()})
    if displaced:
        _samples(
            p,
            "silicon_displacement_forces",
            displaced,
            count=min(len(displaced), 30),
            values=values,
            properties=["forces"],
        )
    if hessian_frames:
        p.frames(
            "silicon_analytic_hessians",
            hessian_frames,
            count=len(hessian_frames),
            values=hessian_values,
            properties=["hessian"],
            parameters=_model_parameters(e, 5),
            targets=[MODEL_TARGET],
        )


def _task6_trace_values(e):
    """Use raw potential-energy trace samples for trajectories without labels."""

    frames = e.trajectory("equilibration_trajectory", "nvt_trajectory", "equilibration")
    if all("energy" in stored(frame) for frame in frames):
        return frames, None
    table = e.table("thermal_trace", "raw_trace")
    if "stage" in table:
        staged = table[table.stage == "nvt"]
        if len(staged):
            table = staged
    timed = all("time_fs" in atoms.info or "time_ps" in atoms.info for atoms in frames)
    if timed:
        frame_times = np.asarray(
            [
                float(atoms.info["time_fs"])
                if "time_fs" in atoms.info
                else 1000 * float(atoms.info["time_ps"])
                for atoms in frames
            ]
        )
    else:
        stride = e.settings["md"].get("sample_interval_steps")
        if type(stride) is not int or stride <= 0:
            raise UnsupportedEvidence(
                "Untimed Task 6 trajectory needs its sample stride"
            )
        # ASE may write the initial state once when a Trajectory is opened and
        # again when an MD callback first runs at step zero. Those identical
        # leading frames shift every later frame index by one (or more).
        initial_copies = 0
        steps = e.settings["md"].get("steps")
        if type(steps) is int and steps >= 0:
            surplus = len(frames) - (steps // stride + 1)
            if surplus > 0:
                initial = frames[0]
                if not all(
                    np.array_equal(frame.numbers, initial.numbers)
                    and np.array_equal(frame.cell.array, initial.cell.array)
                    and np.array_equal(frame.positions, initial.positions)
                    and np.array_equal(frame.get_momenta(), initial.get_momenta())
                    for frame in frames[1 : surplus + 1]
                ):
                    raise UnsupportedEvidence(
                        "Untimed Task 6 trajectory has extra frames with unknown step alignment"
                    )
                initial_copies = surplus
    matched: dict[int, float] = {}
    for row in table.itertuples(index=False):
        step = getattr(row, "step", None)
        energy = getattr(row, "potential_energy_eV", None)
        if energy is None:
            continue
        if timed:
            time = getattr(row, "time_fs", None)
            if time is None:
                continue
            matches = np.flatnonzero(np.isclose(frame_times, time, rtol=0, atol=1e-6))
            if len(matches) != 1:
                continue
            index = int(matches[0])
        elif (
            type(step) in (int, np.int64)
            and step >= 0
            and step % stride == 0
            and step // stride + initial_copies < len(frames)
        ):
            index = step // stride + initial_copies
        else:
            continue
        energy = float(energy)
        if index in matched and not np.isclose(
            matched[index], energy, rtol=0, atol=1e-6
        ):
            raise UnsupportedEvidence(
                "Multiple Task 6 trace energies map to the same saved frame"
            )
        matched[index] = energy
    if len(matched) < 2:
        raise UnsupportedEvidence("No Task 6 trace energies align with saved frames")
    return (
        [frames[index] for index in sorted(matched)],
        [{"energy": matched[index]} for index in sorted(matched)],
    )


def build_level1_plan(e, task_number: int, challenge: str) -> Plan:
    """Build Level 1 jobs from its own artifacts, without Level 2 outputs."""

    if task_number not in range(3, 11):
        raise ValueError("Level 1 MACE verification applies to Tasks 3 through 10")
    p = Plan(e, task_number, challenge)
    if task_number == 3:

        def dimers():
            frames = e.trajectory("dataset", "labeled_dataset", "training_dataset")
            frames = sorted(
                frames, key=lambda a: np.linalg.norm(a.positions[1] - a.positions[0])
            )
            parameters = _model_parameters(e, 3)
            p.frames(
                "teacher_dimer_labels",
                frames,
                count=len(frames),
                parameters=parameters,
                properties=["energy", "forces"],
                operation="reference"
                if parameters
                == {
                    **MODEL_PARAMETERS,
                    "dispersion": True,
                }
                else "mace",
                reference={"name": "task3_teacher_dimers"}
                if parameters
                == {
                    **MODEL_PARAMETERS,
                    "dispersion": True,
                }
                else None,
                fallback_operation="mace"
                if parameters
                == {
                    **MODEL_PARAMETERS,
                    "dispersion": True,
                }
                else None,
                targets=[MODEL_TARGET],
                tolerance={"energy": [1e-5, 1e-3], "forces": [1e-5, 1e-3]},
            )

        p.attempt("teacher_dimer_labels", dimers, [MODEL_TARGET])
    elif task_number == 4:
        p.attempt(
            "palladium_energy_forces",
            lambda: _samples(
                p,
                "palladium_energy_forces",
                e.trajectory("structures", "raw_structures"),
                count=7,
                properties=["energy", "forces"],
            ),
            [MODEL_TARGET],
        )
    elif task_number == 5:
        p.attempt(
            "silicon_force_calculations", lambda: _task5_force_plan(p), [MODEL_TARGET]
        )
    elif task_number == 6:

        def aluminum():
            frames, values = _task6_trace_values(e)
            _samples(
                p,
                "aluminum_equilibration",
                frames,
                count=5,
                values=values,
                properties=["energy"] if values is not None else None,
            )

        p.attempt("aluminum_equilibration", aluminum, [MODEL_TARGET])
    elif task_number == 7:

        def expansion():
            trajectories = e._artifacts["trajectories"]
            if not isinstance(trajectories, Mapping) or not trajectories:
                raise EvidenceError("Task 7 needs a production trajectory")
            for stage in trajectories:
                _samples(
                    p,
                    f"aluminum_npt_{stage}",
                    e.trajectory(f"trajectories.{stage}"),
                    count=5,
                    properties=["energy", "forces", "stress"],
                )

        p.attempt("aluminum_npt_calculations", expansion, [MODEL_TARGET])
    elif task_number == 8:

        def silicon():
            data = e.json("regression_data", "datasets", "dataset_index")["train"]
            frames = e.trajectory("train_structures", "training_structures")
            ids = data["row_ids"]
            if all("row_id" in frame.info for frame in frames):
                lookup = {frame.info["row_id"]: frame for frame in frames}
                ordered = [lookup[item] for item in ids]
            else:
                ordered = frames
            values = [
                {"energy": float(value)}
                for value in finite_array(data["energy_eV"], shape=(100,))
            ]
            _samples(
                p,
                "silicon_training_labels",
                ordered,
                count=5,
                values=values,
                properties=["energy"],
            )

        p.attempt("silicon_training_labels", silicon, [MODEL_TARGET])
    elif task_number == 9:
        p.attempt(
            "copper_teacher_labels",
            lambda: _samples(
                p,
                "copper_teacher_labels",
                e.trajectory("runs.main.trajectory"),
                count=5,
                properties=["energy", "forces"],
            ),
            [MODEL_TARGET],
        )
    else:
        p.attempt(
            "aluminum_heat_capacity_labels",
            lambda: _samples(
                p,
                "aluminum_heat_capacity_labels",
                e.trajectory("production_trajectory", "trajectory"),
                count=5,
                properties=["energy"],
            ),
            [MODEL_TARGET],
        )
    return p


class Level1ModalVerifier(ModalVerifier):
    """Use the existing isolated worker with Level 1 evidence contracts."""

    def evaluate(self, evidence, task_number):
        fingerprint = evidence.fingerprint()
        challenge = secrets.token_hex(32)
        plan = build_level1_plan(evidence, task_number, challenge)
        checks, remotes, aborted = self._evaluate_plan(plan, fingerprint)
        if evidence.fingerprint() != fingerprint:
            raise RuntimeError(
                "Submission changed during independent verification; "
                "retry with frozen evidence"
            )
        return {
            "mode": "modal",
            "evidence_sha256": fingerprint,
            "challenge": challenge,
            "checks": checks,
            "aborted": aborted,
            "max_parallel_calculations": self.max_parallel_calculations,
            "backend": self._backend(remotes),
        }
