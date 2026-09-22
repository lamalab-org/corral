"""Read-only evidence access and diagnostic partial-credit scoring.

Nothing in this module executes submitted code, loads executable model formats,
contacts a service, or attaches a calculator to a saved structure.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from corral.workspace import confine_workspace_path, workspace_relative_path


class EvidenceError(ValueError):
    """A required piece of evidence is absent or cannot be interpreted."""


class UnsupportedEvidence(ValueError):
    """Saved evidence needs a verifier other than the available numerical adapter.

    This is not a failed scientific requirement. Missing or inconsistent evidence
    must use EvidenceError instead.
    """


def read_json(path: Path) -> Any:
    def invalid(value: str) -> None:
        raise EvidenceError(f"Non-finite JSON value: {value}")

    return json.loads(path.read_text(), parse_constant=invalid)


def finite_array(value: Any, ndim: int | None = None, shape=None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not array.size or not np.isfinite(array).all():
        raise EvidenceError("Expected nonempty finite numerical data")
    if ndim is not None and array.ndim != ndim:
        raise EvidenceError(f"Expected {ndim} dimensions, got {array.ndim}")
    if shape is not None and array.shape != tuple(shape):
        raise EvidenceError(f"Expected shape {shape}, got {array.shape}")
    return array


def close(actual: Any, expected: Any, rtol=1e-5, atol=1e-8) -> bool:
    a, b = finite_array(actual), finite_array(expected)
    return bool(a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol))


# Grader policy, deliberately absent from the agent-facing evidence contracts.
# This allows numerical reporting differences; it is not a statistical error bar.
# Structural identities, units, partitions, and continuity use their own checks.
RESULT_RTOL = 0.01


def result_close(actual: Any, expected: Any, *, atol=1e-8) -> bool:
    """Compare finite numerical results with a shared relative allowance.

    Use a symmetric scale because callers may supply the computed value on
    either side. Absolute floors are supplied in the quantity's units, so tiny
    diffusion coefficients do not inherit an energy-sized allowance.
    """
    a, b = finite_array(actual), finite_array(expected)
    return bool(
        a.shape == b.shape
        and np.all(
            np.abs(a - b) <= atol + RESULT_RTOL * np.maximum(np.abs(a), np.abs(b))
        )
    )


def optional_results_match(claims: Mapping, expected: Mapping, *, atol=1e-8) -> bool:
    """Check supplied summaries without requiring an adapter's reporting recipe.

    Callers must independently validate the underlying required evidence and
    compute ``expected``. An empty optional summary is not evidence by itself.
    """
    if not isinstance(claims, Mapping):
        raise EvidenceError("Numerical summaries must be an object")
    return all(
        result_close(claims[key], value, atol=atol)
        for key, value in expected.items()
        if key in claims
    )


def scientific_screen(condition: bool, detail: str) -> bool:
    """A heuristic is an automatic verification limit, not a task requirement."""
    if not condition:
        raise UnsupportedEvidence(detail + "; assess the saved evidence independently")
    return True


def _lookup(mapping: Mapping, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
        value = mapping
        for part in name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                break
            value = value[part]
        else:
            return value
    raise EvidenceError(f"Missing evidence: {' / '.join(names)}")


class Evidence:
    """A submission's data and artifacts, with paths confined when contextualized."""

    def __init__(self, submission: Any):
        workspace = getattr(submission, "workspace", None)
        base = getattr(submission, "manifest_dir", None)
        self.root = Path(workspace).resolve() if workspace is not None else None
        self.base = Path(base).resolve() if base is not None else None
        if isinstance(submission, Path):
            submission = str(submission)
        if isinstance(submission, str):
            try:
                value = json.loads(submission)
            except json.JSONDecodeError:
                value = submission
            if isinstance(value, str):
                path = Path(value).resolve()
                self.base = path.parent
                self.root = self.root or path.parent
                value = read_json(path)
        else:
            value = submission
        if not isinstance(value, dict):
            raise EvidenceError("Expected a JSON manifest object or its path")
        try:
            json.dumps(value, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise EvidenceError("Manifest must contain JSON-serializable data") from exc
        self.manifest = value
        self.results = value.get("results", {})
        self._artifacts = value.get("artifacts", {})
        if not isinstance(self.results, dict) or not isinstance(self._artifacts, dict):
            raise EvidenceError("results and artifacts must be JSON objects")
        self.settings = self._document(value.get("settings"))
        self.report = self._document(value.get("report"))
        self._cache: dict[tuple, Any] = {}

    def _path(self, value: Any) -> Path:
        if not isinstance(value, str | Path) or not str(value):
            raise EvidenceError("Expected an artifact path")
        path = Path(value)
        if self.root is not None and path.is_relative_to("/workspace"):
            # Paths inside linked JSON documents need the same translation as
            # the top-level manifest when a snapshot is restored elsewhere.
            try:
                path = confine_workspace_path(
                    self.root, workspace_relative_path(str(value))
                )
            except ValueError as exc:
                raise EvidenceError(str(exc)) from exc
        if ".." in path.parts:
            raise EvidenceError("Artifact path cannot traverse its workspace")
        if not path.is_absolute():
            if self.base is None:
                raise EvidenceError(
                    "Relative artifact needs a manifest path or workspace"
                )
            path = self.base / path
        path = path.resolve()
        if self.root is not None and not path.is_relative_to(self.root):
            raise EvidenceError("Artifact path escapes the scoring workspace")
        if not path.is_file():
            raise EvidenceError(f"Missing artifact: {path.name}")
        return path

    def _document(self, value: Any) -> dict:
        # Missing metadata does not suppress independent checks of raw artifacts.
        if value is None:
            return {}
        try:
            path = self._path(value)
            data = read_json(path)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def fingerprint(self) -> str:
        """Bind external review to the manifest and every linked evidence file.

        Hash bytes, including opaque checkpoints, without deserializing or running
        them. Missing/invalid links remain part of the identity; their ordinary
        rubric checks still fail.
        """
        digest = hashlib.sha256(b"corral_md.workflow\0")
        # Direct-dict submissions may contain nonfinite results; those fail the
        # numerical checks, but must still have an identity for pending review.
        digest.update(json.dumps(self.manifest, sort_keys=True).encode())

        def visit(value, role):
            if isinstance(value, dict):
                for key in sorted(value):
                    visit(value[key], f"{role}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{role}[{index}]")
            elif value is not None:
                digest.update(role.encode() + b"\0")
                try:
                    path = self._path(value)
                    file_hash = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            file_hash.update(chunk)
                    digest.update(file_hash.digest())
                except (OSError, ValueError, TypeError):
                    digest.update(b"unreadable-link")

        for role in ("artifacts", "settings", "scripts", "report"):
            visit(self.manifest.get(role), role)
        return digest.hexdigest()

    def artifact(self, *names: str) -> Path:
        return self._path(_lookup(self._artifacts, names))

    def artifacts(self, *names: str) -> list[Path]:
        value = _lookup(self._artifacts, names)

        def flatten(item):
            if isinstance(item, dict):
                return [p for child in item.values() for p in flatten(child)]
            if isinstance(item, list):
                return [p for child in item for p in flatten(child)]
            return [self._path(item)]

        return flatten(value)

    def result(self, *names: str) -> Any:
        return _lookup(self.results, names)

    def json(self, *names: str) -> Any:
        path = self.artifact(*names)
        key = ("json", str(path))
        if key not in self._cache:
            self._cache[key] = read_json(path)
        return self._cache[key]

    def array(self, *names: str) -> np.ndarray:
        path = self.artifact(*names)
        if path.suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False, mmap_mode="r")
        if path.suffix.lower() == ".json":
            return np.asarray(read_json(path))
        return np.loadtxt(path, delimiter=",")

    def table(self, *names: str) -> pd.DataFrame:
        path = self.artifact(*names)
        if path.suffix.lower() == ".json":
            data = read_json(path)
            if isinstance(data, dict) and "rows" in data:
                data = data["rows"]
            return pd.DataFrame(data)
        return pd.read_csv(path)

    def trajectory(self, *names: str) -> list:
        path = self.artifact(*names)
        key = ("trajectory", str(path))
        if key in self._cache:
            return self._cache[key]
        if path.suffix.lower() == ".json":
            data = read_json(path)
            records = data.get("frames", [data]) if isinstance(data, dict) else data
            if isinstance(records, list) and records and "positions" in records[0]:
                images = []
                for record in records:
                    pbc = record.get("pbc")
                    if not (
                        isinstance(pbc, bool)
                        or (
                            isinstance(pbc, list)
                            and len(pbc) == 3
                            and all(isinstance(value, bool) for value in pbc)
                        )
                    ):
                        raise EvidenceError("JSON frames require explicit boolean pbc")
                    atoms = Atoms(
                        symbols=record.get("symbols"),
                        numbers=record.get("numbers"),
                        positions=record["positions"],
                        cell=record.get("cell"),
                        pbc=pbc,
                    )
                    if "masses" in record:
                        atoms.set_masses(record["masses"])
                    if "momenta" in record:
                        atoms.set_momenta(record["momenta"])
                    atoms.info.update(record.get("info", {}))
                    for field in ("time_fs", "time_ps", "step", "stage"):
                        if field in record:
                            atoms.info[field] = record[field]
                    stored = {
                        k: record[k]
                        for k in ("energy", "forces", "stress")
                        if k in record
                    }
                    if stored:
                        atoms.calc = SinglePointCalculator(atoms, **stored)
                    images.append(atoms)
            else:
                images = read(path, index=":", format="json")
        else:
            formats = {
                ".traj": "traj",
                ".extxyz": "extxyz",
                ".xyz": "extxyz",
                ".dump": "lammps-dump-text",
                ".lammpstrj": "lammps-dump-text",
            }
            if path.suffix.lower() not in formats:
                raise EvidenceError(
                    "Use a documented trajectory data format, not an executable object"
                )
            images = read(path, index=":", format=formats[path.suffix.lower()])
        if not images:
            raise EvidenceError("Trajectory is empty")
        self._cache[key] = images
        return images


class Rubric:
    """Independent checks with explicit possible points and failure diagnostics."""

    def __init__(
        self,
        task_number: int | None = None,
        *,
        binary: bool = False,
        fail_fast: bool = False,
    ):
        self.task_number = task_number
        self.binary = binary
        self.fail_fast = fail_fast
        self.checks: list[dict] = []

    @property
    def failed(self) -> bool:
        return any(
            check["points"] > 0 and check["status"] == "failed" for check in self.checks
        )

    def check(
        self,
        name: str,
        points: float,
        condition: bool | Callable | None,
        detail: str = "",
    ) -> bool:
        if not np.isfinite(points) or points < 0:
            raise ValueError("Rubric points must be finite and nonnegative")
        if self.fail_fast and points > 0 and self.failed:
            self.checks.append(
                {
                    "name": name,
                    "points": float(points),
                    "earned": 0.0,
                    "status": "skipped",
                    "detail": "Not evaluated after an earlier required check failed.",
                }
            )
            return False
        status = "failed"
        try:
            value = condition() if callable(condition) else condition
            if isinstance(value, tuple):
                value, explanation = value
                detail = str(explanation)
            if value is None:
                status = "unverified"
            elif bool(value):
                status = "passed"
        except UnsupportedEvidence as exc:
            status = "unverified"
            detail = str(exc)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
        self.checks.append(
            {
                "name": name,
                "points": float(points),
                "earned": float(points) if status == "passed" else 0.0,
                "status": status,
                "detail": detail,
            }
        )
        return status == "passed"

    def unverified(self, name: str, detail: str, points: float = 0) -> None:
        self.check(name, points, None, detail)

    @property
    def pending_checks(self) -> list[dict]:
        return [
            c for c in self.checks if c["points"] > 0 and c["status"] == "unverified"
        ]

    @property
    def score(self) -> float | None:
        """A verifier limitation must never become an implicit zero or free credit."""
        if self.binary and self.failed:
            return 0.0
        if self.pending_checks:
            return None
        earned = sum(c["earned"] for c in self.checks)
        if self.binary:
            return float(earned >= 100.0)
        return min(1.0, earned / 100.0)

    def apply_review(self, review: Mapping, evidence_sha256: str) -> None:
        """Apply explicit evaluator-supplied decisions, never submission metadata.

        Reviews can resolve only pending checks. They cannot override failed raw
        evidence checks. Validate the whole bundle before applying any decision.
        """
        if (
            not isinstance(review, Mapping)
            or review.get("evidence_sha256") != evidence_sha256
        ):
            raise ValueError("Review must match the current evidence_sha256")
        decisions = review.get("decisions")
        if not isinstance(decisions, Mapping) or not decisions:
            raise ValueError("Review requires a nonempty decisions mapping")
        pending = {c["name"]: c for c in self.pending_checks}
        for name, decision in decisions.items():
            if name not in pending:
                raise ValueError(f"Review can resolve only pending checks: {name}")
            if (
                not isinstance(decision, Mapping)
                or type(decision.get("passed")) is not bool
            ):
                raise ValueError(f"Review decision for {name} requires boolean passed")
            if any(
                not isinstance(decision.get(k), str) or not decision[k].strip()
                for k in ("reviewer", "reason")
            ):
                raise ValueError(
                    f"Review decision for {name} requires reviewer and reason"
                )
        for name, decision in decisions.items():
            check = pending[name]
            check["status"] = "passed" if decision["passed"] else "failed"
            check["earned"] = check["points"] if decision["passed"] else 0.0
            check["review"] = dict(decision)

    def as_dict(self) -> dict:
        score = self.score
        pending = self.pending_checks if score is None else []
        if self.binary:
            bounds = [score, score] if score is not None else [0.0, 1.0]
        else:
            bounds = [
                min(1.0, sum(c["earned"] for c in self.checks) / 100.0),
                min(
                    1.0,
                    (
                        sum(c["earned"] for c in self.checks)
                        + sum(c["points"] for c in self.pending_checks)
                    )
                    / 100.0,
                ),
            ]
        return {
            "task_number": self.task_number,
            "mode": "artifact_only",
            "score": score,
            "status": "pending_review" if score is None else "complete",
            "pending_checks": [c["name"] for c in pending],
            "pending_points": sum(c["points"] for c in pending),
            "score_bounds": bounds,
            "earned_points": sum(c["earned"] for c in self.checks),
            "possible_points": 100.0,
            "checks": self.checks,
        }


def reproducibility(e: Evidence, r: Rubric) -> None:
    def linked_files():
        paths = (
            e.artifacts(*tuple(e._artifacts))
            if len(e._artifacts) == 1
            else [p for key in e._artifacts for p in e.artifacts(key)]
        )
        return bool(paths) and all(p.stat().st_size > 0 for p in paths)

    r.check(
        "manifest_and_artifacts",
        5,
        linked_files,
        "Named artifacts must exist and be nonempty.",
    )
    r.check(
        "recorded_settings",
        2,
        lambda: bool(e.settings)
        and e._path(e.manifest.get("settings")).stat().st_size > 0,
    )

    def scripts():
        values = e.manifest.get("scripts", [])
        return (
            isinstance(values, list)
            and bool(values)
            and all(e._path(p).stat().st_size > 0 for p in values)
        )

    r.check(
        "saved_scripts",
        2,
        scripts,
        "Scripts are inspected as files and never executed.",
    )
    r.check(
        "reported_results",
        1,
        lambda: _contains_number(e.results) and _finite_json(e.results),
    )


def level1_reproducibility(e: Evidence, r: Rubric) -> None:
    """Score the shared evidence contract for a preparatory Level 1 task.

    The submission templates describe the complete paired workflow, so a Level 1
    submission legitimately leaves the Level-2-only artifact roles empty.  Only
    nonempty links are therefore interpreted as submitted evidence here.  The
    task-specific Level 1 evaluator remains responsible for requiring every
    artifact needed by its own prompt.

    Level 1 tasks do not all have a standalone numerical result (for example,
    some prepare a dataset or a restartable state), so the shared ten points are
    assigned to retained artifacts, settings, and scripts rather than to the
    Level 2 result object.
    """

    def linked_values(value: Any) -> list[Any]:
        if isinstance(value, Mapping):
            return [item for child in value.values() for item in linked_values(child)]
        if isinstance(value, list):
            return [item for child in value for item in linked_values(child)]
        if value in (None, ""):
            return []
        return [value]

    def linked_files() -> bool:
        # The submission resolver maps an empty relative placeholder to the
        # manifest directory.  Treat that directory spelling like the original
        # empty value; a real submitted artifact must resolve to a file.
        values = [
            value
            for value in linked_values(e._artifacts)
            if e.base is None or Path(value).resolve() != e.base
        ]
        return bool(values) and all(
            e._path(value).stat().st_size > 0 for value in values
        )

    r.check(
        "manifest_and_linked_artifacts",
        6,
        linked_files,
        "Every nonempty artifact link must exist and contain evidence; unused Level 2 roles may remain empty.",
    )
    r.check(
        "recorded_settings",
        2,
        lambda: bool(e.settings)
        and e._path(e.manifest.get("settings")).stat().st_size > 0,
    )

    def scripts() -> bool:
        values = e.manifest.get("scripts", [])
        return (
            isinstance(values, list)
            and bool(values)
            and all(e._path(path).stat().st_size > 0 for path in values)
        )

    r.check(
        "saved_scripts",
        2,
        scripts,
        "Scripts are retained for reproducibility and are never executed by the scorer.",
    )


def _finite_json(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_json(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_json(v) for v in value)
    return not isinstance(value, int | float) or bool(np.isfinite(value))


def _contains_number(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_number(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_number(v) for v in value)
    return isinstance(value, int | float) and not isinstance(value, bool)
