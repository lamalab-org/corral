"""Agent-facing format checks. Never grade, execute scripts, or load models."""

from __future__ import annotations

import json
import math
import posixpath
import shlex
from pathlib import Path

import numpy as np
from ase.io.formats import UnknownFileTypeError
from corral_md.calculator_settings import (
    CalculatorSettingsError,
    UnsupportedCalculatorSettings,
    parse_calculator_settings,
)
from corral_md.provenance import provenance_identifiers
from corral_md.submission_examples import load_example
from corral_md.workflow_scoring.common import Evidence

from corral.core.tool import tool
from corral.workspace import confine_workspace_path, workspace_relative_path

# These are public serialization conventions, not numerical grading criteria.
_ALIASES = {
    1: {
        "reference_cell": ("starting_cell",),
        "prepared_state": ("initial_state",),
        "thermo": ("thermodynamics",),
    },
    2: {
        "lammps_inputs": ("lammps_input", "inputs"),
        "raw_logs": ("logs", "lammps_logs"),
        "boundary_states": ("stage_boundaries",),
        "thermal_trace": ("thermal_traces", "trace"),
    },
    3: {"dataset": ("labeled_dataset", "training_dataset")},
    4: {"structures": ("raw_structures",)},
    6: {
        "equilibration_trajectory": ("nvt_trajectory", "equilibration"),
        "nve_trajectory": ("production_trajectory", "trajectory"),
        "thermal_trace": ("thermal_log", "md_log"),
    },
    7: {"stages": ("stage_map",), "thermal_trace": ("raw_trace",)},
    8: {
        "train_structures": ("training_structures",),
        "regression_data": ("datasets", "dataset_index"),
    },
    10: {
        "initial_state": ("initial",),
        "boundary_states": ("stage_boundaries",),
        "production_trajectory": ("trajectory",),
        "thermal_trace": ("thermal_traces",),
    },
}
_STRUCTURES = {
    "reference_cell",
    "prepared_state",
    "heating_end_state",
    "final_state",
    "dataset",
    "structures",
    "reference_structure",
    "strained_structures",
    "equilibration_trajectory",
    "nve_trajectory",
    "md_trajectory",
    "initial_structure",
    "initialized_state",
    "train_structures",
    "id_test_structures",
    "strained_test_structures",
    "input_structure",
    "initial_state",
    "production_trajectory",
    "restartable_final_state",
}
_REQUIRED_SETTINGS = {
    1: ("timestep_ps", "velocity_seed"),
    2: (),  # The saved LAMMPS input carries the protocol.
    3: ("teacher_model", "dispersion", "energy_unit", "force_unit"),
    4: ("checkpoint", "force_constant_method"),
    5: ("model", "masses_amu", "energy_unit", "force_unit", "force_constant_unit"),
    6: (
        "model",
        "model_settings",
        "md.timestep_fs",
        "md.temperature_dof",
        "md.target_temperature_K",
        "md.initial_temperature_K",
        "md.random_seed",
        "md.remove_com_once",
        "md.thermostat",
        "md.equilibration_integrator",
    ),
    7: (
        "model",
        "input_structure",
        "temperature_dof",
        "random_seed",
        "velocity_initializations",
        "remove_com",
        "timestep_fs",
        "thermostat",
        "barostat",
        "target_pressure_bar",
        "isotropic",
    ),
    8: (
        "teacher_model",
        "energy_unit",
        "length_unit",
        "generation.train.seed",
        "generation.train.library",
        "generation.train.method",
    ),
    9: ("teacher", "runs.main"),
    10: (
        "model",
        "model_settings",
        "md.thermostat",
        "md.timestep_fs",
        "md.random_seed",
        "md.initial_temperature_K",
        "md.velocity_initializations",
        "md.manual_velocity_resets",
        "md.ensemble",
        "md.temperature_dof",
        "stages",
    ),
}
_INTEGER_FIELDS = {
    "seed",
    "random_seed",
    "velocity_seed",
    "temperature_dof",
    "steps",
    "sample_interval_steps",
    "trajectory_sample_steps",
    "velocity_initializations",
    "velocity_reinitializations_during_run",
    "velocity_rescalings_during_run",
    "manual_velocity_resets",
    "n_max",
    "l_max",
}
_NULL_TYPES = {
    "detrend": str,
    "mass_weighted": bool,
    "remove_com": bool,
    "average": str,
}
_FORMATS = {".traj", ".extxyz", ".xyz", ".json", ".cif", ".data", ".dump", ".lammpstrj"}
_REQUIRED_ROLES = {
    1: ("prepared_state", "heating_end_state"),
    2: ("lammps_inputs", "raw_logs", "thermal_trace", "boundary_states"),
    3: ("dataset",),
    4: ("structures", "force_constants"),
    5: ("reference_structure", "strained_structures", "strain_calculations"),
    6: ("equilibration_trajectory", "thermal_trace"),
    7: ("initial_structure", "stages", "trajectories", "thermal_trace"),
    8: ("train_structures", "regression_data"),
    9: ("runs", "correlation"),
    10: ("initial_state", "boundary_states", "production_trajectory", "thermal_trace"),
}


class _Validator:
    def __init__(self, workspace, task_number, level, *, require_provenance=False):
        self.root = Path(workspace).resolve()
        self.task_number, self.level = task_number, level
        self.require_provenance = require_provenance
        self.example = load_example(task_number, level=level)
        self.aliases = _ALIASES.get(task_number, {})
        self.errors, self.warnings = [], []
        self.documents, self.files, self.frames = {}, {}, {}
        self.artifacts, self.settings = {}, {}
        self.base = self.root

    def issue(self, code, location, message, suggestion, *, warning=False):
        target = self.warnings if warning else self.errors
        item = {
            "code": code,
            "location": location,
            "message": message,
            "suggestion": suggestion,
        }
        if item not in target and len(target) < 100:
            target.append(item)

    def detail(self, exc):
        return str(exc).replace(str(self.root), "/workspace")[:500]

    def public(self, path):
        return "/workspace/" + path.relative_to(self.root).as_posix()

    def file(self, value, location, *, base=None):
        if not isinstance(value, str) or not value.strip():
            self.issue(
                "file_path",
                location,
                "Expected a nonempty file path.",
                "Link an existing file using a /workspace path or a path relative to the manifest.",
            )
            return None
        try:
            supplied = Path(value)
            if supplied.is_relative_to("/workspace"):
                path = self.root / workspace_relative_path(value)
            elif supplied.is_absolute():
                raise ValueError(
                    "Use a /workspace path, not a controller or external path"
                )
            else:
                path = (base or self.base) / supplied
            path = confine_workspace_path(
                self.root, Path(posixpath.normpath(str(path)))
            )
            if path.is_dir():
                self.issue(
                    "directory_link",
                    location,
                    f"{value} is a directory.",
                    "List its individual evidence files in manifest.artifacts, using a list if needed.",
                )
                return None
            if not path.is_file():
                raise ValueError(f"File does not exist: {value}")
            if not path.stat().st_size:
                raise ValueError(f"File is empty: {value}")
            self.files[location] = path
            return path
        except (OSError, ValueError) as exc:
            self.issue(
                "file_path",
                location,
                self.detail(exc),
                "Correct the link and keep it inside the task workspace.",
            )
            return None

    def document(self, path):
        if path in self.documents:
            return self.documents[path]

        def invalid(value):
            raise ValueError(f"Non-finite JSON number: {value}")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result

        def finite(value):
            if isinstance(value, dict):
                for child in value.values():
                    finite(child)
            elif isinstance(value, list):
                for child in value:
                    finite(child)
            elif isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite JSON number")

        try:
            data = json.loads(
                path.read_text(), parse_constant=invalid, object_pairs_hook=unique
            )
            finite(data)
            self.documents[path] = data
            return data
        except (OSError, UnicodeError, ValueError) as exc:
            self.issue(
                "json",
                self.public(path),
                self.detail(exc),
                "Save valid JSON with unique keys and finite numbers.",
            )
            self.documents[path] = None
            return None

    def links(self, value, location):
        if isinstance(value, dict):
            if not value:
                self.issue(
                    "empty_links",
                    location,
                    "No files are linked.",
                    "List the evidence files used for this role.",
                )
            for name, child in value.items():
                if "<" in name:
                    self.issue(
                        "placeholder",
                        f"{location}.{name}",
                        "An example key is still present.",
                        "Replace it with your actual stage or record ID.",
                    )
                self.links(child, f"{location}.{name}")
        elif isinstance(value, list):
            if not value:
                self.issue(
                    "empty_links",
                    location,
                    "No files are linked.",
                    "List the evidence files used for this role.",
                )
            for index, child in enumerate(value):
                self.links(child, f"{location}[{index}]")
        else:
            self.file(value, location)

    def settings_types(self, value, template, location="settings"):
        if isinstance(template, dict):
            if not isinstance(value, dict):
                self.issue(
                    "type",
                    location,
                    "Expected a JSON object.",
                    "Keep the object layout shown in submission_examples/settings.json.",
                )
                return
            for key, child in template.items():
                if key in value:
                    self.settings_types(value[key], child, f"{location}.{key}")
            return
        if isinstance(template, list):
            if not isinstance(value, list):
                self.issue(
                    "type",
                    location,
                    "Expected a JSON array.",
                    "Use an array matching the settings template.",
                )
            elif template:
                for index, child in enumerate(value):
                    self.settings_types(child, template[0], f"{location}[{index}]")
            return
        name = location.rsplit(".", 1)[-1]
        expected = (
            (int if name in _INTEGER_FIELDS else _NULL_TYPES.get(name, float))
            if template is None
            else type(template)
        )
        if isinstance(template, str) and template.startswith("<number["):
            expected = list
        # A template's 300 and a submitted 300.0 express the same numeric type.
        if type(template) is int:
            expected = float
        if name in _INTEGER_FIELDS:
            expected = int
        good = (
            (type(value) in (int, float) and math.isfinite(value))
            if expected in (int, float)
            else isinstance(value, expected)
        )
        if expected is int:
            good = type(value) is int
        if expected is str:
            good = good and bool(value.strip()) and not value.startswith("<")
        if not good:
            label = {
                int: "integer",
                float: "finite number",
                str: "nonempty string",
                bool: "boolean",
                list: "array",
            }[expected]
            self.issue(
                "type",
                location,
                f"Expected {label}; received {type(value).__name__}.",
                f"Use a {label} here; keep explanatory text in a separate notes field.",
            )

    def require(self, value, dotted, location):
        current = value
        for key in dotted.split("."):
            if not isinstance(current, dict) or key not in current:
                self.issue(
                    "missing_field",
                    f"{location}.{dotted}",
                    "Required field is missing.",
                    "Use the exact key and nesting shown in the task's submission template.",
                )
                return
            current = current[key]

    def role(self, name):
        return next(
            (
                key
                for key in (name, *self.aliases.get(name, ()))
                if key in self.artifacts
            ),
            None,
        )

    def role_files(self, name):
        role = self.role(name)
        if role is None:
            return []
        prefix = f"manifest.artifacts.{role}"
        return [
            path
            for key, path in self.files.items()
            if key == prefix or key.startswith((prefix + "[", prefix + "."))
        ]

    def trajectory(self, path, location):
        if path in self.frames:
            return self.frames[path]
        if path.suffix.lower() == ".restart":
            self.issue(
                "opaque_restart",
                location,
                "Binary LAMMPS restart contents were not inspected.",
                "Retain readable boundary states as well as the restart when the task requests them.",
                warning=True,
            )
            return []
        if path.suffix.lower() not in _FORMATS:
            self.issue(
                "structure_format",
                location,
                f"Unsupported structure format: {path.suffix}.",
                "Save structures as ASE .traj, extended XYZ, or JSON frames; NPZ arrays cannot serve as a trajectory.",
            )
            return []
        try:
            if path.suffix.lower() == ".json" and self.document(path) is None:
                return []
            # Path was confined above. The data reader never runs a calculator or
            # submitted script. No scoring functions or verifier are invoked.
            frames = Evidence({"artifacts": {"frames": str(path)}}).trajectory("frames")
            for index, atoms in enumerate(frames):
                count = len(atoms)
                arrays = {
                    "positions": (atoms.positions, (count, 3)),
                    "cell": (atoms.cell.array, (3, 3)),
                }
                if atoms.has("momenta"):
                    arrays["momenta"] = (atoms.arrays["momenta"], (count, 3))
                stored = getattr(atoms.calc, "results", {})
                if "forces" in stored:
                    arrays["forces"] = (stored["forces"], (count, 3))
                if "energy" in stored:
                    arrays["energy"] = (stored["energy"], ())
                for name, (values, shape) in arrays.items():
                    array = np.asarray(values, dtype=float)
                    if array.shape != shape or not np.isfinite(array).all():
                        self.issue(
                            "array_shape",
                            f"{location}.frames[{index}].{name}",
                            f"Expected finite data with shape {shape}.",
                            "Save all atoms in the same order with complete arrays.",
                        )
            self.frames[path] = frames
            return frames
        except UnknownFileTypeError as exc:
            self.issue(
                "structure_data",
                location,
                self.detail(exc),
                "Export the structures as ASE .traj, extended XYZ, or JSON frames "
                "containing symbols, positions, cell, pbc, and any required momenta. "
                "A JSON metadata object linking an NPZ file is not a trajectory.",
            )
            return []
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AssertionError,
        ) as exc:
            self.issue(
                "structure_data",
                location,
                self.detail(exc),
                "Save readable structures with symbols, positions, cell, and pbc; check array shapes.",
            )
            return []

    def inspect_artifacts(self):
        for location, path in list(self.files.items()):
            if not location.startswith("manifest.artifacts."):
                continue
            role = location.removeprefix("manifest.artifacts.").split("[")[0]
            canonical = next(
                (key for key, aliases in self.aliases.items() if role in aliases), role
            )
            structure = (
                canonical in _STRUCTURES
                or role.startswith("trajectories.")
                or (
                    role.startswith("runs.")
                    and role.rsplit(".", 1)[-1]
                    in {"trajectory", "equilibration", "boundary"}
                )
                or (canonical == "boundary_states" and self.task_number == 10)
            )
            if structure:
                frames = self.trajectory(path, location)
                labels_required = (
                    self.task_number == 3 and canonical == "dataset"
                ) or (self.task_number == 4 and canonical == "structures")
                momenta_required = canonical in {
                    "equilibration_trajectory",
                    "nve_trajectory",
                    "md_trajectory",
                    "production_trajectory",
                    "restartable_final_state",
                }
                for index, atoms in enumerate(frames):
                    stored = getattr(atoms.calc, "results", {})
                    missing = []
                    if labels_required:
                        missing.extend(
                            name for name in ("energy", "forces") if name not in stored
                        )
                    if momenta_required and not atoms.has("momenta"):
                        missing.append("momenta")
                    if missing:
                        self.issue(
                            "stored_fields",
                            f"{location}.frames[{index}]",
                            f"Missing stored fields: {', '.join(missing)}.",
                            "Retain these fields in each frame; saving a calculator or a script is not a saved numerical label.",
                        )
            elif path.suffix.lower() == ".json":
                self.document(path)
            elif path.suffix.lower() in {".npy", ".npz"}:
                try:
                    data = np.load(path, allow_pickle=False)
                    if isinstance(data, np.lib.npyio.NpzFile):
                        with data:
                            for key in data.files:
                                data[
                                    key
                                ]  # Reject object arrays without unpickling them.
                except (OSError, ValueError, TypeError) as exc:
                    self.issue(
                        "array_file",
                        location,
                        self.detail(exc),
                        "Save numerical arrays that load with allow_pickle=False.",
                    )

    def linked_documents(self):
        declared = {
            path
            for location, path in self.files.items()
            if location.startswith("manifest.artifacts.")
        }

        def visit(value, document, location):
            if isinstance(value, dict):
                for key, child in value.items():
                    child_location = f"{location}.{key}"
                    if key in {"arrays_file", "lammps_data"}:
                        path = self.file(child, child_location, base=document.parent)
                        if path is not None and path not in declared:
                            self.issue(
                                "unlisted_reference",
                                child_location,
                                "Referenced data file is not listed in manifest.artifacts.",
                                "Add a manifest artifact link for this file as well as the reference in the JSON document.",
                            )
                    else:
                        visit(child, document, child_location)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, document, f"{location}[{index}]")

        for path, value in self.documents.items():
            if path in declared:
                visit(value, path, self.public(path))

    def table(self, role):
        paths = self.role_files(role)
        if len(paths) != 1:
            return None
        path = paths[0]
        try:
            import pandas as pd

            if path.suffix.lower() == ".json":
                data = self.document(path)
                if data is None:
                    return None
                if isinstance(data, dict) and "rows" in data:
                    data = data["rows"]
                return pd.DataFrame(data)
            return pd.read_csv(path)
        except (OSError, ValueError, TypeError) as exc:
            self.issue(
                "table",
                self.public(path),
                self.detail(exc),
                "Save a JSON table or CSV with named columns.",
            )
            return None

    def task_conventions(self):
        if self.task_number == 2:
            self.lammps_includes()
        if self.task_number == 4 and str(self.settings.get("checkpoint", "")).endswith(
            ".json"
        ):
            self.issue(
                "checkpoint_identity",
                "settings.checkpoint",
                "This field names a JSON document instead of a model.",
                "Record the checkpoint identifier here (for example teacher.model); link the identity document separately.",
            )
        if self.task_number == 5:
            from .workflow_scoring.task_5 import HESSIAN_OPERATIONS

            for path in self.role_files("strain_calculations"):
                records = self.document(path)
                if isinstance(records, dict):
                    records = records.get("calculations", records.get("records", []))
                if isinstance(records, list):
                    operations = {
                        field: tuple(
                            alias for aliases in choices.values() for alias in aliases
                        )
                        for field, choices in HESSIAN_OPERATIONS.items()
                    }
                    for index, record in enumerate(records):
                        if not isinstance(record, dict):
                            continue
                        for key, accepted in operations.items():
                            if record.get(key) not in accepted:
                                self.issue(
                                    "operation_name",
                                    f"{self.public(path)}[{index}].{key}",
                                    "The recorded operation name has no automatic reader.",
                                    f"Use one of {accepted} if it describes the operation actually used; otherwise retain its explanation for independent review.",
                                    warning=True,
                                )
        if self.task_number == 6:
            table = self.table("thermal_trace")
            if table is not None and "stage" in table:
                accepted = (
                    {"equilibration", "initialized", "nvt", "eq"}
                    if self.level == 1
                    else {"equilibration", "nvt"}
                )
                if not table.stage.astype(str).str.lower().isin(accepted).any():
                    self.issue(
                        "stage_label",
                        "artifacts.thermal_trace.stage",
                        "No rows use an accepted equilibration stage label.",
                        "Use equilibration or nvt for the equilibration rows.",
                    )
                if (
                    self.level == 2
                    and not table.stage.astype(str)
                    .str.lower()
                    .isin({"production", "nve"})
                    .any()
                ):
                    self.issue(
                        "stage_label",
                        "artifacts.thermal_trace.stage",
                        "No rows identify the NVE production stage.",
                        "Use production or nve for the production rows.",
                    )
        if self.task_number == 10:
            self.stage_conventions()

    def lammps_includes(self):
        inputs = self.role_files("lammps_inputs")
        linked = inputs + self.role_files("potential")
        for path in linked:
            try:
                for line in path.read_text().replace("&\n", " ").splitlines():
                    parts = shlex.split(line, comments=True)
                    if len(parts) != 2 or parts[0] != "include":
                        continue
                    if "$" in parts[1]:
                        self.issue(
                            "dynamic_include",
                            self.public(path),
                            "A variable-based include was not resolved by format validation.",
                            "Ensure its expanded file is linked under lammps_inputs or potential.",
                            warning=True,
                        )
                    elif (
                        parts[1] != "/workspace/potentials/BKS/pot.mod"
                        and sum(p.name == Path(parts[1]).name for p in linked) != 1
                    ):
                        self.issue(
                            "include_link",
                            self.public(path),
                            f"Include {parts[1]} is not linked unambiguously under a recognized role.",
                            "Link it under manifest.artifacts.potential or lammps_inputs. potential_file is not a recognized include role.",
                        )
            except (OSError, UnicodeError, ValueError) as exc:
                self.issue(
                    "lammps_input",
                    self.public(path),
                    self.detail(exc),
                    "Check the saved input's text and quoting.",
                )

    def stage_conventions(self):
        stages = self.settings.get("stages")
        if not isinstance(stages, list):
            return
        expected_stages = 1 if self.level == 1 else 8
        if len(stages) != expected_stages:
            self.issue(
                "stage_count",
                "settings.stages",
                f"Expected {expected_stages} temperature stages containing equilibration and production.",
                "Give each temperature stage its own ID, start/end times, and production_start_time_fs/production_end_time_fs.",
            )
        for index, stage in enumerate(stages):
            if isinstance(stage, dict):
                for field in self.example["files"]["settings.json"]["stages"][0]:
                    self.require(stage, field, f"settings.stages[{index}]")
        frames = [
            frame
            for path in self.role_files("boundary_states")
            for frame in self.frames.get(path, [])
        ]
        allowed_counts = (1, 2) if self.level == 1 else (16,)
        if frames and len(frames) not in allowed_counts:
            self.issue(
                "boundary_count",
                "artifacts.boundary_states",
                f"Found {len(frames)} boundary frames; expected {allowed_counts}.",
                (
                    "Link the final state alone, or initial and final states. Keep intermediate frames in a separate artifact."
                    if self.level == 1
                    else "Save each stage's start and end states in stage order: 16 frames for the eight-stage cycle."
                ),
            )
        table = self.table("thermal_trace")
        if table is None or not stages:
            return
        if "stage" not in table or "time_fs" not in table:
            self.issue(
                "trace_columns",
                "artifacts.thermal_trace",
                "Expected stage and time_fs columns.",
                "Retain the entire stage's thermal trace, including its initial and final samples.",
            )
            return
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                continue
            rows = table[table.stage == stage.get("id")]
            if rows.empty:
                self.issue(
                    "stage_reference",
                    "artifacts.thermal_trace.stage",
                    f"No trace rows match settings.stages[{index}].id.",
                    "Use the same stage ID in settings, results, and thermal trace.",
                )
                continue
            for field, endpoint in (
                ("start_time_fs", rows.time_fs.iloc[0]),
                ("end_time_fs", rows.time_fs.iloc[-1]),
            ):
                expected = stage.get(field)
                if type(expected) not in (int, float):
                    continue
                try:
                    matches = math.isclose(
                        float(endpoint), expected, rel_tol=0, abs_tol=1e-6
                    )
                except (ValueError, TypeError):
                    matches = False
                if not matches:
                    self.issue(
                        "trace_coverage",
                        "artifacts.thermal_trace.time_fs",
                        f"Trace endpoint does not match the declared {field} ({expected}).",
                        "Include initialization, equilibration, and production in the stage trace; keep the production trajectory separate.",
                    )

    def run(self, manifest_path):
        path = self.file(manifest_path, "manifest", base=self.root)
        if path is None:
            return self.result()
        self.base = path.parent
        manifest = self.document(path)
        if not isinstance(manifest, dict):
            self.issue(
                "manifest_object",
                "manifest",
                "Expected a JSON object.",
                "Submit a completed manifest.json following the task template.",
            )
            return self.result()
        try:
            provenance_identifiers(manifest, required=self.require_provenance)
        except ValueError as exc:
            self.issue(
                "provenance_identifiers",
                "manifest.provenance",
                str(exc),
                "Copy run_id and action_id (and optionally release_id) from the "
                "run_verified_md receipt into manifest.provenance or top-level "
                "manifest fields. Duplicate fields must agree. These format "
                "checks do not verify execution.",
            )
        self.artifacts = manifest.get("artifacts", {})
        if not isinstance(self.artifacts, dict) or not self.artifacts:
            self.issue(
                "artifacts_object",
                "manifest.artifacts",
                "Expected a nonempty object of artifact roles and file links.",
                "Use the artifact-role keys in submission_examples/manifest.json.",
            )
            self.artifacts = {}
        self.links(self.artifacts, "manifest.artifacts")
        for role, example in self.example["manifest"]["artifacts"].items():
            actual_role = self.role(role)
            if actual_role is not None and isinstance(example, str):
                value = self.artifacts[actual_role]
                boundary_list = (
                    self.task_number == 10
                    and self.level == 1
                    and role == "boundary_states"
                    and isinstance(value, list)
                )
                if not isinstance(value, str) and not boundary_list:
                    self.issue(
                        "artifact_link_type",
                        f"manifest.artifacts.{actual_role}",
                        "This role expects one file path.",
                        "Link a single evidence file; put any additional files under separate artifact roles.",
                    )
        for role in _REQUIRED_ROLES[self.task_number]:
            if self.role(role) is None:
                self.issue(
                    "missing_artifact",
                    f"manifest.artifacts.{role}",
                    "Required artifact role is missing.",
                    "Link the retained evidence under this role, using the task's manifest template.",
                )
        for role in self.example["manifest"]["artifacts"]:
            if self.role(role) is None:
                # Templates include optional alternatives. Missing roles are
                # advisory; the scientific evaluator decides required evidence.
                self.issue(
                    "template_role",
                    f"manifest.artifacts.{role}",
                    "This role from the task template is absent.",
                    "Check whether the task requires it; use the template's role name for evidence you retained.",
                    warning=True,
                )
        scripts = manifest.get("scripts")
        if not isinstance(scripts, list) or not scripts:
            self.issue(
                "scripts",
                "manifest.scripts",
                "Expected a nonempty list of saved scripts.",
                "List the scripts used to generate and analyze the submitted evidence.",
            )
        else:
            self.links(scripts, "manifest.scripts")
        path = self.file(manifest.get("settings"), "manifest.settings")
        if path is not None:
            settings = self.document(path)
            if isinstance(settings, dict):
                self.settings = settings
                self.settings_types(settings, self.example["files"]["settings.json"])
                if self.task_number >= 3:
                    for role in (None, "teacher", "student", "md"):
                        try:
                            parse_calculator_settings(settings, role=role)
                        except CalculatorSettingsError as exc:
                            self.issue(
                                "calculator_settings",
                                exc.location,
                                str(exc),
                                "Put calculator options in model_settings (or the "
                                "corresponding role_model_settings object). A calculator "
                                "class name may be recorded separately as calculator. "
                                "Use float32/float64 for precision and a boolean for "
                                "dispersion; duplicate options must agree.",
                                warning=isinstance(exc, UnsupportedCalculatorSettings),
                            )
                for field in _REQUIRED_SETTINGS[self.task_number]:
                    self.require(settings, field, "settings")
            else:
                self.issue(
                    "settings_object",
                    "settings",
                    "Expected a JSON object.",
                    "Keep the settings layout shown in the template.",
                )
        report = manifest.get("report")
        if report is not None and not isinstance(report, dict):
            self.links(report, "manifest.report")
        self.inspect_artifacts()
        self.linked_documents()
        self.task_conventions()
        return self.result()

    def result(self):
        return {
            "valid": not self.errors,
            "task": f"level_{self.level}_task_{self.task_number}",
            "scope": "Submission format and saved-file checks only. Scientific correctness, model identity/execution, numerical results, and benchmark score are not evaluated.",
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_submission(
    manifest_path: str,
    workspace: str | Path,
    task_number: int,
    *,
    level: int = 1,
    require_provenance: bool = False,
) -> dict:
    """Inspect one task's saved submission without modifying it or grading it."""
    return _Validator(
        workspace, task_number, level, require_provenance=require_provenance
    ).run(manifest_path)


def build_validate_submission_tool(
    workspace: str | Path,
    task_number: int,
    *,
    level: int,
    require_provenance: bool = False,
):
    @tool(trusted=True, workspace_access="read")
    def validate_submission(manifest_path: str) -> str:
        """Check the current MD task's saved submission format before submit_answer.

        Returns JSON with valid, errors, and warnings. Each issue names a field
        or file and suggests a repair. Checks links, settings types, readable data,
        structure array shapes, supported stage/trace conventions, and required
        controlled-execution identifiers. Reads only
        workspace files; never runs scripts, simulations, model verification, or
        scientific grading. Fix errors and call again before final submission.

        Args:
            manifest_path: Absolute /workspace path to the completed manifest.json.
        """
        workspace_relative_path(manifest_path)
        return json.dumps(
            _Validator(
                workspace, task_number, level, require_provenance=require_provenance
            ).run(manifest_path)
        )

    return validate_submission
