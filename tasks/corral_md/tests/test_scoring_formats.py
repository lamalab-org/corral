"""The task examples stay authoritative while artifact representations vary."""

import json

import numpy as np
import pytest
from corral_md.scoring.common import (
    EvaluationError,
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
)
from corral_md.scoring.contracts import submission_example
from corral_md.scoring.formats import (
    dataset_records,
    md_records,
    normalize_force_record,
    physical_record,
)


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("task", range(1, 11))
def test_each_unchanged_submission_example_defines_required_keys(tmp_path, task, level):
    example = submission_example(task, level)

    def materialize(value):
        if isinstance(value, dict):
            return {key: materialize(child) for key, child in value.items()}
        if isinstance(value, list):
            return [materialize(child) for child in value]
        if isinstance(value, str):
            path = tmp_path / value.rsplit("/", 1)[-1]
            path.write_text("retained scientific data")
            return str(path)
        return value

    manifest = materialize(example)
    evidence = Evidence(manifest, task_number=task, level=level)
    assert evidence.manifest == manifest
    role = next(iter(manifest["artifacts"]))
    del manifest["artifacts"][role]
    with pytest.raises(EvidenceError, match="Missing requested submission field"):
        Evidence(manifest, task_number=task, level=level)


def test_unknown_representation_and_scorer_bug_have_distinct_outcomes():
    pending = Rubric(8, binary=True)
    pending.error("read_dataset", 100, UnsupportedEvidence("unknown record layout"))
    assert pending.as_dict()["score"] is None
    assert pending.as_dict()["status"] == "pending_review"
    wrong = Rubric(8, binary=True)
    wrong.error("labels", 100, EvidenceError("energies disagree with the teacher"))
    assert wrong.as_dict()["score"] == 0
    with pytest.raises(EvaluationError, match="Task 8, check reader_bug"):
        Rubric(8).error("reader_bug", 100, RuntimeError("unexpected internal error"))


def test_pending_required_reader_does_not_run_dependent_checks():
    rubric = Rubric(6, binary=True, fail_fast=True)
    rubric.error("read", 80, UnsupportedEvidence("unsupported layout"))
    rubric.check("dependent", 20, lambda: pytest.fail("dependent check ran"))
    assert rubric.score is None
    assert rubric.checks[-1]["status"] == "skipped"


def test_conflicting_aliases_and_units_are_rejected():
    with pytest.raises(EvidenceError, match="Conflicting"):
        physical_record({"time_fs": 1, "time_ps": 2})
    with pytest.raises(EvidenceError, match="Conflicting"):
        physical_record({"positions": [[0, 0, 0]], "positions_A": [[1, 0, 0]]})


def test_nested_stages_and_frame_arrays_preserve_scientific_values(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "seed": 7,
                    "atomic_numbers": [13],
                    "cell": np.eye(3).tolist(),
                },
                "stages": [
                    {
                        "name": "equilibration",
                        "dt_fs": 1,
                        "n_steps": 2,
                        "sample_interval_steps": 1,
                        "time_fs": [0, 1, 2],
                        "step": [0, 1, 2],
                        "positions": [[[i, 0, 0]] for i in range(3)],
                        "energy_eV": [-3, -2, -1],
                    }
                ],
            }
        )
    )
    rows = md_records(path, 6)
    assert rows[0]["random_seed"] == 7
    samples = [row for row in rows if row["event"] == "sample"]
    assert [row["potential_energy_eV"] for row in samples] == [-3, -2, -1]
    assert samples[2]["positions_A"] == [[2, 0, 0]]
    assert rows[-1]["steps"] == 2


def test_incomplete_flat_log_is_a_scientific_failure(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(
        "\n".join(
            map(
                json.dumps,
                [
                    {"type": "meta", "nsteps": 100},
                    {"type": "sample", "step": 0},
                    {"type": "sample", "step": 50},
                ],
            )
        )
    )
    with pytest.raises(EvidenceError, match="before its declared final step"):
        md_records(path, 6)


def test_dataset_row_identity_does_not_prescribe_a_string_template(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "row_id": f"structure-{i}",
                    "config_index": i,
                    "seed": 17,
                    "positions": [[i, 0, 0]],
                    "cell": np.eye(3).tolist(),
                    "energy_eV": i,
                }
            )
            for i in range(2)
        )
    )
    converted = dataset_records(path, 1)
    assert [row["row_id"] for row in converted[1]["rows"]] == [
        "structure-0",
        "structure-1",
    ]
    assert "model_sha256" not in converted[0]


def test_npz_geometry_is_reconstructed_only_from_recorded_displacements():
    displacement = np.array([[0.01, 0, 0], [-0.01, 0, 0]])
    reference = np.arange(375).reshape(125, 3).astype(float)
    record = normalize_force_record(
        {
            "supercell_positions": reference,
            "supercell_cell": np.eye(3),
            "numbers": np.full(125, 46),
            "displacements": displacement,
            "forces": np.zeros((2, 125, 3)),
        }
    )
    np.testing.assert_allclose(record["positions_A"][:, 0], reference[0] + displacement)
    np.testing.assert_allclose(
        record["positions_A"][:, 1:], np.broadcast_to(reference[1:], (2, 124, 3))
    )
    assert "energies_eV" not in record
    assert "checkpoint_sha256" not in record["metadata"]
