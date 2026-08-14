"""Tests for user-facing retrosynthesis tool responses."""

from unittest.mock import patch

from retrosynthesis.tools import search_catalog_by_smiles


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
