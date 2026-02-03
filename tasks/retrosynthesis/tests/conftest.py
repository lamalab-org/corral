"""Shared fixtures and test data for retrosynthesis tests."""

import os

# Set dummy API keys before any imports that might load retrosynthesis_utils
# These must be set at module level (not in fixtures) so they're available during test collection
os.environ.setdefault("MOLPORT_API_KEY", "dummy_molport_key_for_testing")
os.environ.setdefault("CHEMSPACE_API_KEY", "dummy_chemspace_key_for_testing")
os.environ.setdefault("MCULE_API_KEY", "dummy_mcule_key_for_testing")

import pytest


@pytest.fixture()
def simple_smiles():
    """Simple molecule SMILES strings for testing."""
    return {
        "ethanol": "CCO",
        "benzene": "c1ccccc1",
        "phenol": "c1ccccc1O",
        "benzoic_acid": "c1ccccc1C(=O)O",
        "acetone": "CC(=O)C",
        "toluene": "Cc1ccccc1",
        "methanol": "CO",
        "water": "O",
    }


@pytest.fixture()
def complex_smiles():
    """More complex molecule SMILES strings for testing."""
    return {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    }


@pytest.fixture()
def invalid_smiles():
    """Invalid SMILES strings for testing error handling."""
    return [
        "invalid_smiles",
        "C1CCC",  # Unclosed ring
        "",
        "12345",
        "XYZ",
    ]


@pytest.fixture()
def template_ids():
    """Sample template IDs for testing."""
    return {
        "valid": ["1", "10", "100", "1000"],
        "invalid": ["999999999", "invalid_id", "-1"],
    }


@pytest.fixture()
def reaction_smiles():
    """Sample reaction SMILES strings for testing."""
    return {
        "esterification": "CCO.CC(=O)O>>CCOC(=O)C",
        "oxidation": "CCO>>CC=O",
        "reduction": "CC(=O)C>>CC(O)C",
        "simple": "CC>>C",
    }


@pytest.fixture()
def cas_numbers():
    """Sample CAS numbers for testing."""
    return {
        "ethanol": "64-17-5",
        "benzene": "71-43-2",
        "water": "7732-18-5",
        "methanol": "67-56-1",
    }


@pytest.fixture()
def retrosynthesis_tree_valid():
    """Valid retrosynthesis tree structure for testing."""
    return {
        "type": "mol",
        "smiles": "CC(=O)OC",
        "children": [
            {
                "type": "reaction",
                "template_id": "1",
                "children": [
                    {"type": "mol", "smiles": "CO", "children": []},
                    {"type": "mol", "smiles": "CC(=O)O", "children": []},
                ],
            }
        ],
    }


@pytest.fixture()
def retrosynthesis_tree_invalid():
    """Invalid retrosynthesis tree structures for testing."""
    return {
        "missing_type": {"smiles": "CCO", "children": []},
        "missing_smiles": {"type": "mol", "children": []},
        "invalid_smiles": {"type": "mol", "smiles": "invalid", "children": []},
        "empty": {},
    }


@pytest.fixture()
def functional_groups():
    """Common functional groups for testing."""
    return [
        "alcohol",
        "aldehyde",
        "ketone",
        "carboxylic_acid",
        "ester",
        "ether",
        "amine",
        "amide",
        "nitrile",
        "halogen",
        "alkene",
        "alkyne",
        "aromatic",
    ]


@pytest.fixture()
def bond_types():
    """Common bond types for testing."""
    return ["6-6", "6-7", "6-8", "C-C", "C-O", "C-N", "C=O", "C=C"]
