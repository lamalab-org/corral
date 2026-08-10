"""Tests for the local PubChem helper."""

import asyncio
from unittest.mock import AsyncMock, call, patch

import pytest
from spectra_elucidation.pubchem_helpers import PubChem


@patch.object(PubChem, "get_data_from_url", new_callable=AsyncMock)
@patch("spectra_elucidation.pubchem_helpers.random.shuffle")
def test_get_compound_isomers_by_formula(mock_shuffle, mock_get_data):
    mock_get_data.side_effect = [
        {"IdentifierList": {"CID": [702, 887]}},
        {"PropertyTable": {"Properties": [{"SMILES": "CCO"}, {"SMILES": "COC"}]}},
    ]

    result = asyncio.run(PubChem.get_compound_isomers_by_formula("C2H6O", limit=2))

    assert result == ["CCO", "COC"]
    mock_shuffle.assert_called_once_with([702, 887])
    assert mock_get_data.await_args_list == [
        call(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/"
            "C2H6O/cids/JSON"
        ),
        call(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/702,887/"
            "property/IsomericSMILES/JSON"
        ),
    ]


@pytest.mark.parametrize(
    ("formula", "limit", "message"),
    [
        ("", 5, "Molecular formula cannot be empty"),
        ("C2H6O", -1, "limit must be greater than or equal to 0"),
    ],
)
def test_get_compound_isomers_by_formula_validates_input(formula, limit, message):
    with pytest.raises(ValueError, match=message):
        asyncio.run(PubChem.get_compound_isomers_by_formula(formula, limit=limit))
