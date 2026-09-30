"""Read LAMMPS binary restarts with the deployed trusted SimAgent worker."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from .common import EvidenceError, UnsupportedEvidence, finite_array
from .verification import ModalVerifier, Plan, file_ref

if TYPE_CHECKING:
    from pathlib import Path


class ModalRestartReader:
    """Convert an opaque restart without executing submission code locally."""

    def __call__(self, path: Path) -> Atoms:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise EvidenceError("LAMMPS binary restart exceeds the 64 MiB limit")
        digest = file_ref(path)["sha256"]
        plan = Plan(None, 1, secrets.token_hex(16))
        checkpoint = plan.checkpoint(path)
        plan.add(
            "restart_state",
            "lammps_restart",
            {},
            {},
            parameters=checkpoint,
        )
        try:
            response = ModalVerifier()._calculate(plan, digest)
        except Exception as exc:
            raise UnsupportedEvidence(
                "The trusted LAMMPS restart reader is unavailable"
            ) from exc
        if (
            response.get("evidence_sha256") != digest
            or response.get("challenge") != plan.challenge
            or len(response.get("jobs", [])) != 1
            or response["jobs"][0].get("id") != "restart_state"
        ):
            raise UnsupportedEvidence("The restart reader returned unbound evidence")
        result = response["jobs"][0]
        if result.get("status") != "complete":
            status = result.get("status")
            if status == "failed" or (
                status == "error"
                and str(result.get("detail", "")).startswith("ValueError:")
            ):
                raise EvidenceError("The LAMMPS binary restart could not be read")
            raise UnsupportedEvidence(
                "The trusted LAMMPS restart reader did not complete"
            )
        value = result.get("value")
        if not isinstance(value, dict) or value.get("checkpoint_sha256") != digest:
            raise UnsupportedEvidence("The restart reader returned a different file")
        state = value.get("state")
        if not isinstance(state, dict):
            raise UnsupportedEvidence("The restart reader returned no state")
        try:
            numbers = finite_array(state["numbers"], ndim=1)
            if not np.array_equal(numbers, numbers.astype(int)):
                raise EvidenceError("Restart has noninteger atomic numbers")
            size = len(numbers)
            cell = finite_array(state["cell"], shape=(3, 3))
            positions = finite_array(state["positions"], shape=(size, 3))
            masses = finite_array(state["masses"], shape=(size,))
            momenta = finite_array(state["momenta"], shape=(size, 3))
            atom_ids = finite_array(state["atom_ids"], shape=(size,))
            if (
                not np.array_equal(atom_ids, atom_ids.astype(int))
                or len(np.unique(atom_ids)) != size
            ):
                raise EvidenceError("Restart has invalid atom IDs")
            pbc = state["pbc"]
            if not (
                isinstance(pbc, list)
                and len(pbc) == 3
                and all(type(item) is bool for item in pbc)
            ):
                raise EvidenceError("Restart has invalid periodic boundaries")
            atoms = Atoms(
                numbers=numbers.astype(int), positions=positions, cell=cell, pbc=pbc
            )
            atoms.set_masses(masses)
            atoms.set_momenta(momenta)
            atoms.arrays["atom_ids"] = atom_ids.astype(int)
            if type(state.get("timestep")) is int:
                atoms.info["step"] = state["timestep"]
            return atoms
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, EvidenceError):
                raise
            raise UnsupportedEvidence(
                "The restart reader returned invalid state"
            ) from exc
