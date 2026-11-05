"""Tests for user-facing retrosynthesis tool responses."""

from unittest.mock import patch

from retrosynthesis.tools import search_catalog_by_smiles


@patch('retrosynthesis.retrosynthesis_utils.search_by_template')
def test_verify_step_accepts_documented_retro_smarts_without_database_lookup(lookup):
    from retrosynthesis.tools import verify_step

    template = '[C:1][O:2]>>[C:1].[O:2]'
    assert verify_step.execute(molecule_smiles='CO', template_id=template, precursors=['C', 'O']) == 'True'
    assert verify_step.execute(molecule_smiles='CC', template_id=template, precursors=['C', 'O']) == 'False'
    lookup.assert_not_called()


@patch('retrosynthesis.retrosynthesis_utils.search_by_template')
def test_verify_step_still_accepts_numeric_catalog_ids(lookup):
    from retrosynthesis.tools import verify_step

    lookup.return_value = {'mapped_rxn': '[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]'}
    assert verify_step.execute(molecule_smiles='CCO', template_id='123', precursors=['CC=O']) == 'True'
    lookup.assert_called_once_with('123')


def test_deprotection_crosses_worker_serialization_boundary_and_returns_smiles():
    import cloudpickle
    from retrosynthesis.tools import deprotect_molecule

    restored = cloudpickle.loads(cloudpickle.dumps(deprotect_molecule))
    assert restored.execute(molecule_smiles="CC(C)(C)OC(=O)NCCO") == "NCCO"


@patch("retrosynthesis.tools.check_price")
def test_catalog_search_reports_unavailable_smiles(mock_check_price):
    mock_check_price.return_value = {"CC-not-found": []}

    result = search_catalog_by_smiles.execute(smiles_list=["CC-not-found"], limit=5)

    assert result == (
        'No chemicals available in the catalogue with SMILES "CC-not-found".'
    )


@patch("retrosynthesis.tools.check_price")
def test_catalog_search_reports_multiple_unavailable_smiles(mock_check_price):
    mock_check_price.return_value = {"missing-1": [], "missing-2": []}

    result = search_catalog_by_smiles.execute(
        smiles_list=["missing-1", "missing-2"], limit=5
    )

    assert result == (
        'No chemicals available in the catalogue with SMILES "missing-1", "missing-2".'
    )
