"""The evaluator accepts only a state tied to the submitted restart bytes."""

import hashlib

import numpy as np
import pytest
from ase.build import bulk

from corral_md.score import check_level1_workflow
from corral_md.workflow_scoring.common import EvidenceError, UnsupportedEvidence
from corral_md.workflow_scoring.restart_reader import ModalRestartReader
from corral_md.workflow_scoring.verification import ModalVerifier


def test_level1_task1_uses_trusted_reader_by_default():
    assert isinstance(check_level1_workflow(1).restart_reader, ModalRestartReader)
    assert check_level1_workflow(1, verification_backend="offline").restart_reader is None


def test_modal_restart_reader_reconstructs_sha_bound_state(tmp_path, monkeypatch):
    binary = tmp_path / "prepared.restart"
    binary.write_bytes(b"opaque restart bytes")
    reference = bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3))
    reference.set_momenta(np.ones((216, 3)))
    state = {
        "numbers": reference.numbers.tolist(),
        "positions": reference.positions.tolist(),
        "cell": reference.cell.array.tolist(),
        "pbc": [True, True, True],
        "atom_ids": list(range(1, 217)),
        "momenta": reference.get_momenta().tolist(),
        "masses": reference.get_masses().tolist(),
        "timestep": 0,
    }

    def calculate(_self, plan, fingerprint):
        assert fingerprint == hashlib.sha256(binary.read_bytes()).hexdigest()
        assert plan.jobs[0]["operation"] == "lammps_restart"
        assert plan.jobs[0]["parameters"]["checkpoint_sha256"] == fingerprint
        return {
            "evidence_sha256": fingerprint,
            "challenge": plan.challenge,
            "jobs": [
                {
                    "id": "restart_state",
                    "status": "complete",
                    "value": {"checkpoint_sha256": fingerprint, "state": state},
                }
            ],
        }

    monkeypatch.setattr(ModalVerifier, "_calculate", calculate)
    atoms = ModalRestartReader()(binary)
    assert np.array_equal(atoms.numbers, reference.numbers)
    assert np.array_equal(atoms.arrays["atom_ids"], range(1, 217))
    assert np.array_equal(atoms.get_momenta(), reference.get_momenta())


@pytest.mark.parametrize(
    "result", [
        {"status": "failed"},
        {"status": "error", "detail": "ValueError: invalid restart"},
    ]
)
def test_restart_reader_distinguishes_invalid_data_from_unavailable_modal(
    tmp_path, monkeypatch, result
):
    binary = tmp_path / "prepared.restart"
    binary.write_bytes(b"invalid")

    def failed(_self, plan, fingerprint):
        return {
            "evidence_sha256": fingerprint,
            "challenge": plan.challenge,
            "jobs": [{"id": "restart_state", **result}],
        }

    monkeypatch.setattr(ModalVerifier, "_calculate", failed)
    with pytest.raises(EvidenceError, match="could not be read"):
        ModalRestartReader()(binary)

    def unavailable(_self, _plan, _fingerprint):
        raise ConnectionError("Modal unavailable")

    monkeypatch.setattr(ModalVerifier, "_calculate", unavailable)
    with pytest.raises(UnsupportedEvidence, match="unavailable"):
        ModalRestartReader()(binary)


def test_restart_reader_rejects_oversized_data_before_upload(tmp_path, monkeypatch):
    binary = tmp_path / "oversized.restart"
    with binary.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(EvidenceError, match="64 MiB"):
        ModalRestartReader()(binary)
