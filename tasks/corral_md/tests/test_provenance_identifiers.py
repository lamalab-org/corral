"""Submission ID layouts agree across format validation and trusted scoring."""

import copy

import pytest
from corral_md.provenance import provenance_identifiers
from corral_md.workflow_scoring.common import Evidence
from corral_md.workflow_scoring.verification import ModalVerifier


@pytest.mark.parametrize("layout", ["top_level", "nested", "mixed", "duplicated"])
def test_supported_layouts_preserve_metadata(layout):
    ids = {"run_id": "run-1", "action_id": "action-1", "release_id": "release-1"}
    manifest = {
        "top_level": ids,
        "nested": {"provenance": ids},
        "mixed": {
            "run_id": "run-1",
            "provenance": {"action_id": "action-1", "release_id": "release-1"},
        },
        "duplicated": {**ids, "provenance": ids},
    }[layout]
    original = copy.deepcopy(manifest)
    assert provenance_identifiers(manifest, required=True) == ids
    assert manifest == original


@pytest.mark.parametrize("key", ["run_id", "action_id", "release_id"])
def test_conflicting_ids_are_rejected_before_backend_access(key, monkeypatch):
    import modal

    monkeypatch.setattr(
        modal.Function, "from_name", lambda *_: pytest.fail("must not access backend")
    )
    manifest = {key: "one", "provenance": {key: "two"}}
    with pytest.raises(ValueError, match="Conflicting"):
        provenance_identifiers(manifest)
    record = ModalVerifier(require_provenance=True).provenance(Evidence(manifest))
    assert record["status"] == "failed"
    assert record["targets"]


@pytest.mark.parametrize("value", [None, "", " ", 1, False, [], {}])
@pytest.mark.parametrize("nested", [False, True])
def test_invalid_identifier_types_and_empty_values(value, nested):
    manifest = {"run_id": value}
    if nested:
        manifest = {"provenance": manifest}
    with pytest.raises(ValueError, match="run_id must be a nonempty string"):
        provenance_identifiers(manifest)


@pytest.mark.parametrize("value", [None, [], "receipt.json"])
def test_provenance_must_be_an_object(value):
    with pytest.raises(ValueError, match="provenance must be a JSON object"):
        provenance_identifiers({"provenance": value})


@pytest.mark.parametrize(
    "manifest", [{}, {"run_id": "run-1"}, {"provenance": {"action_id": "action-1"}}]
)
def test_required_pair_and_optional_absence(manifest):
    with pytest.raises(ValueError, match="Missing controlled-execution identifiers"):
        provenance_identifiers(manifest, required=True)
    assert provenance_identifiers({}) == {}


@pytest.mark.parametrize("key", ["run_id", "action_id", "release_id"])
def test_declared_ids_cannot_override_evaluator_identity(key, monkeypatch):
    import modal

    monkeypatch.setattr(
        modal.Function, "from_name", lambda *_: pytest.fail("must not access backend")
    )
    ids = {"run_id": "run-1", "action_id": "action-1", "release_id": "release-1"}
    verifier = ModalVerifier(**ids, volume_name="simulations", require_provenance=True)
    ids[key] = "different"
    record = verifier.provenance(Evidence({"provenance": ids}))
    assert record["status"] == "failed"
    assert key in record["detail"]


def test_missing_ids_still_require_review():
    record = ModalVerifier(require_provenance=True).provenance(Evidence({}))
    assert record["status"] == "unverified"
    assert record["targets"]
