"""
Tests for the tools in the spectra_elucidation package.
"""

from unittest.mock import MagicMock, patch

import pytest
from spectra_elucidation.tools import (
    carbon_nmr_spectra,
    get_formula_from_smiles,
    hsqc_nmr_spectra,
    ir_spectra,
    mass_spectrometry_spectra,
    obtain_isomers_from_molecular_formula,
    proton_nmr_spectra,
    retrieve_aromatic_protons_shifts,
    retrieve_carbon_shifts,
    retrieve_dbe_formula,
    retrieve_isotope_distribution,
    retrieve_protons_shifts,
    return_possible_fragments,
    search_by_smiles,
    simulate_spectra,
    validate_smiles,
)


class TestGetFormulaFromSmiles:
    """Tests for the get_formula_from_smiles tool."""

    def test_valid_ethanol(self):
        """Test formula generation for ethanol."""
        result = get_formula_from_smiles.execute(smiles="CCO")
        assert result == "C2H6O"

    def test_valid_benzene(self):
        """Test formula generation for benzene."""
        result = get_formula_from_smiles.execute(smiles="C1=CC=CC=C1")
        assert result == "C6H6"

    def test_valid_acetic_acid(self):
        """Test formula generation for acetic acid."""
        result = get_formula_from_smiles.execute(smiles="CC(=O)O")
        assert result == "C2H4O2"

    def test_valid_glycine(self):
        """Test formula generation for glycine."""
        result = get_formula_from_smiles.execute(smiles="C(C(=O)O)N")
        assert result == "C2H5NO2"

    def test_invalid_smiles(self):
        """Test with invalid SMILES string."""
        result = get_formula_from_smiles.execute(smiles="INVALID_SMILES")
        assert result == "Invalid SMILES string"

    def test_aromatic_notation(self):
        """Test with aromatic notation."""
        result = get_formula_from_smiles.execute(smiles="c1ccccc1")
        assert result == "C6H6"

    def test_complex_molecule(self):
        """Test with a more complex molecule (aspirin)."""
        result = get_formula_from_smiles.execute(smiles="CC(=O)Oc1ccccc1C(=O)O")
        assert result == "C9H8O4"


class TestValidateSmiles:
    """Tests for the validate_smiles tool."""

    def test_valid_ethanol(self):
        """Test validation of valid ethanol SMILES."""
        result = validate_smiles.execute(smiles="CCO")
        assert result == "True"

    def test_valid_benzene(self):
        """Test validation of valid benzene SMILES."""
        result = validate_smiles.execute(smiles="C1=CC=CC=C1")
        assert result == "True"

    def test_valid_aromatic(self):
        """Test validation of aromatic notation."""
        result = validate_smiles.execute(smiles="c1ccccc1")
        assert result == "True"

    def test_invalid_smiles(self):
        """Test validation of invalid SMILES."""
        result = validate_smiles.execute(smiles="INVALID_SMILES")
        assert result == "False"

    def test_empty_string(self):
        """Test validation of empty string."""
        result = validate_smiles.execute(smiles="")
        assert result == "True"  # Empty string creates a valid empty molecule in RDKit

    def test_valid_cyclohexane(self):
        """Test validation of cyclohexane."""
        result = validate_smiles.execute(smiles="C1CCCCC1")
        assert result == "True"

    def test_valid_phenol(self):
        """Test validation of phenol."""
        result = validate_smiles.execute(smiles="C1=CC=C(C=C1)O")
        assert result == "True"


class TestRetrieveProtonsShifts:
    """Tests for the retrieve_protons_shifts tool."""

    def test_retrieve_protons_shifts(self):
        """Test that retrieve_protons_shifts returns expected data structure."""
        result = retrieve_protons_shifts.execute()

        # Check that it's a string representation of a list
        assert isinstance(result, str)
        assert result.startswith("[")
        assert result.endswith("]")

        # Check for expected proton types in the result
        assert "Aldehyde" in result
        assert "Aromatic" in result
        assert "Alkene" in result
        assert "delta / ppm" in result
        assert "9.5 - 10.5" in result  # Aldehyde range


class TestRetrieveAromaticProtonsShifts:
    """Tests for the retrieve_aromatic_protons_shifts tool."""

    def test_retrieve_aromatic_protons_shifts(self):
        """Test that retrieve_aromatic_protons_shifts returns expected data structure."""
        result = retrieve_aromatic_protons_shifts.execute()

        # Check that it's a string representation of a list
        assert isinstance(result, str)
        assert result.startswith("[")
        assert result.endswith("]")

        # Check for expected substituents in the result
        assert "NO2" in result
        assert "CHO" in result
        assert "Ortho" in result
        assert "Meta" in result
        assert "Para" in result
        assert "0.95" in result  # NO2 Ortho value


class TestRetrieveCarbonShifts:
    """Tests for the retrieve_carbon_shifts tool."""

    def test_retrieve_carbon_shifts(self):
        """Test that retrieve_carbon_shifts returns expected data structure."""
        result = retrieve_carbon_shifts.execute()

        # Check that it's a string representation of a list
        assert isinstance(result, str)
        assert result.startswith("[")
        assert result.endswith("]")

        # Check for expected functional groups in the result
        assert "CH3-" in result
        assert "Ketones" in result
        assert "Aldehydes" in result
        assert "Shift (ppm)" in result
        assert "200-210 ppm" in result  # Ketones range


class TestRetrieveIsotopeDistribution:
    """Tests for the retrieve_isotope_distribution tool."""

    def test_retrieve_isotope_distribution(self):
        """Test that retrieve_isotope_distribution returns expected data structure."""
        result = retrieve_isotope_distribution.execute()

        # Check that it's a string representation of a dictionary
        assert isinstance(result, str)
        assert result.startswith("{")
        assert result.endswith("}")

        # Check for expected elements in the result
        assert "Carbon" in result
        assert "Hydrogen" in result
        assert "Chlorine" in result
        assert "Bromine" in result
        assert "isotopes" in result
        assert "m/z_peaks" in result


class TestRetrieveDBEFormula:
    """Tests for the retrieve_dbe_formula tool."""

    def test_retrieve_dbe_formula(self):
        """Test that retrieve_dbe_formula returns expected information."""
        result = retrieve_dbe_formula.execute()

        # Check that it's a string with the formula
        assert isinstance(result, str)

        # Check for expected content
        assert "DBE" in result or "Double Bond Equivalent" in result
        assert "2C + 2 + N - H - X" in result
        assert "Benzene" in result or "C6H6" in result
        assert "Interpretation" in result


class TestSearchBySmiles:
    """Tests for the search_by_smiles tool."""

    @patch("spectra_elucidation.tools.vector_database_search")
    def test_search_by_smiles_success(self, mock_vector_search):
        """Test successful search by SMILES."""
        # Mock the return value
        mock_results = [
            {
                "entry_id": "nmrshiftdb2:123",
                "compound_name": "Ethanol",
                "smiles": "CCO",
                "spectrum": {"nucleus": "13C", "shifts": [10.0, 60.0]},
            },
            {
                "entry_id": "nmrshiftdb2:456",
                "compound_name": "Ethanol derivative",
                "smiles": "CCCO",
                "spectrum": {"nucleus": "13C", "shifts": [10.0, 30.0, 70.0]},
            },
        ]
        mock_vector_search.return_value = mock_results

        result = search_by_smiles.execute(smiles="CCO", top_k=10)

        # Check that the search was called with correct parameters
        mock_vector_search.assert_called_once()
        call_kwargs = mock_vector_search.call_args[1]
        assert call_kwargs["query"] == "CCO"
        assert call_kwargs["top_k"] == 10
        assert call_kwargs["collection_name"] == "nmrshiftdb2"
        assert call_kwargs["chemical_model"] == "ibm-research/MoLFormer-XL-both-10pct"

        # Check the result - tool decorator converts to string
        assert isinstance(result, str)
        assert "nmrshiftdb2:123" in result
        assert "Ethanol" in result

    @patch("spectra_elucidation.tools.vector_database_search")
    def test_search_by_smiles_with_top_k(self, mock_vector_search):
        """Test search with custom top_k parameter."""
        mock_vector_search.return_value = []

        _result = search_by_smiles.execute(smiles="C1=CC=CC=C1", top_k=5)

        call_kwargs = mock_vector_search.call_args[1]
        assert call_kwargs["top_k"] == 5

    @patch("spectra_elucidation.tools.vector_database_search")
    def test_search_by_smiles_error(self, mock_vector_search):
        """Test error handling in search."""
        mock_vector_search.side_effect = Exception("Database error")

        with pytest.raises(ValueError, match="Error:.*Database error"):
            search_by_smiles.execute(smiles="CCO")


class TestCarbonNMRSpectra:
    """Tests for the carbon_nmr_spectra tool."""

    @patch("spectra_elucidation.tools.remote_call")
    def test_carbon_nmr_spectra_success(self, mock_remote_call):
        """Test successful carbon NMR spectra prediction."""
        # Mock the remote call
        mock_function = MagicMock()
        mock_function.return_value = "13C NMR: δC 10.0, 60.0 ppm"
        mock_remote_call.return_value = mock_function

        result = carbon_nmr_spectra.execute(h_smiles="CCO")

        # Check that remote_call was called with correct parameters
        mock_remote_call.assert_called_once_with(
            function_name="get_c13_nmr_prediction", env_name="chemenv"
        )
        # Check that the returned function was called with the SMILES
        mock_function.assert_called_once_with(smiles="CCO")

        # Check the result
        assert "13C NMR" in result
        assert "δC" in result or "10.0" in result


class TestProtonNMRSpectra:
    """Tests for the proton_nmr_spectra tool."""

    @patch("spectra_elucidation.tools.remote_call")
    def test_proton_nmr_spectra_success(self, mock_remote_call):
        """Test successful proton NMR spectra prediction."""
        # Mock the remote call
        mock_function = MagicMock()
        mock_function.return_value = (
            "1H NMR: δH 1.2 (t, 3H), 3.6 (q, 2H), 2.5 (s, 1H) ppm"
        )
        mock_remote_call.return_value = mock_function

        result = proton_nmr_spectra.execute(h_smiles="CCO")

        # Check that remote_call was called with correct parameters
        mock_remote_call.assert_called_once_with(
            function_name="get_h_nmr_prediction", env_name="chemenv"
        )
        # Check that the returned function was called with the SMILES
        mock_function.assert_called_once_with(smiles="CCO")

        # Check the result
        assert "1H NMR" in result or "δH" in result


class TestIRSpectra:
    """Tests for the ir_spectra tool."""

    @patch("spectra_elucidation.tools.remote_call")
    def test_ir_spectra_success(self, mock_remote_call):
        """Test successful IR spectra prediction."""
        # Mock the remote call
        mock_function = MagicMock()
        mock_function.return_value = (
            "IR: 3400 cm-1 (O-H stretch), 2900 cm-1 (C-H stretch)"
        )
        mock_remote_call.return_value = mock_function

        result = ir_spectra.execute(h_smiles="CCO")

        # Check that remote_call was called with correct parameters
        mock_remote_call.assert_called_once_with(
            function_name="get_ir_prediction", env_name="chemenv"
        )
        # Check that the returned function was called with the SMILES
        mock_function.assert_called_once_with(smiles="CCO")

        # Check the result
        assert "IR" in result or "cm-1" in result or "3400" in result


class TestHSQCNMRSpectra:
    """Tests for the hsqc_nmr_spectra tool."""

    @patch("spectra_elucidation.tools.make_api_call")
    def test_hsqc_nmr_spectra_success(self, mock_api_call):
        """Test successful HSQC NMR spectra prediction."""
        # Mock the API response with correct format
        mock_api_call.return_value = {
            "spectra": [
                {
                    "info": {"pulseSequence": "hsqc"},
                    "zones": {
                        "values": [
                            {
                                "signals": [
                                    {
                                        "x": {"delta": 7.4, "atoms": [0, 1]},
                                        "y": {"delta": 128.0},
                                    },
                                    {
                                        "x": {"delta": 3.6, "atoms": [2, 3]},
                                        "y": {"delta": 60.0},
                                    },
                                ]
                            }
                        ]
                    },
                }
            ]
        }

        result = hsqc_nmr_spectra.execute(h_smiles="CCO")

        # Check that the API was called
        mock_api_call.assert_called_once()
        call_args = mock_api_call.call_args[0]
        assert (
            call_args[0]
            == "https://lamalab-org--nmr-prediction-api-predict-nmr.modal.run"
        )
        assert call_args[1]["smiles"] == "CCO"

        # Check the result
        assert isinstance(result, str)
        assert "HSQC" in result or "delta" in result or "3.6" in result

    @patch("spectra_elucidation.tools.make_api_call")
    def test_hsqc_nmr_spectra_no_hsqc(self, mock_api_call):
        """Test when no HSQC spectrum is found."""
        # Mock the API response without HSQC
        mock_api_call.return_value = {
            "spectra": [
                {
                    "info": {"pulseSequence": "other"},
                    "peaks": [],
                }
            ]
        }

        result = hsqc_nmr_spectra.execute(h_smiles="CCO")

        assert "No HSQC spectrum found" in result


class TestMassSpectrometrySpectra:
    """Tests for the mass_spectrometry_spectra tool."""

    @patch("spectra_elucidation.tools.make_api_call")
    def test_mass_spectrometry_spectra_success(self, mock_api_call):
        """Test successful mass spectrometry spectra prediction."""
        # Mock the API response with correct format (x, y keys)
        mock_api_call.return_value = [
            {"x": 46.0, "y": 1000},
            {"x": 47.0, "y": 50},
        ]

        result = mass_spectrometry_spectra.execute(h_smiles="CCO")

        # Check that the API was called
        mock_api_call.assert_called_once()
        call_args = mock_api_call.call_args[0]
        assert (
            call_args[0]
            == "https://lamalab-org--nmr-prediction-api-predict-isotopic-distribution.modal.run"
        )
        assert call_args[1]["smiles"] == "CCO"

        # Check the result
        assert isinstance(result, str)
        assert "m/z" in result

    def test_mass_spectrometry_invalid_smiles(self):
        """Test with invalid SMILES string."""
        result = mass_spectrometry_spectra.execute(h_smiles="INVALID")

        assert "Invalid SMILES string" in result


class TestObtainIsomersFromMolecularFormula:
    """Tests for the obtain_isomers_from_molecular_formula tool."""

    @patch("spectra_elucidation.tools.remote_call")
    def test_obtain_isomers_success(self, mock_remote_call):
        """Test successful isomer retrieval."""
        # Mock the remote call
        mock_function = MagicMock()
        mock_function.return_value = ["CCO", "COC"]
        mock_remote_call.return_value = mock_function

        result = obtain_isomers_from_molecular_formula.execute(
            molecular_formula="C2H6O", limit=10
        )

        # Check that remote_call was called with correct parameters
        mock_remote_call.assert_called_once_with(
            function_name="get_compound_isomers_pubchem_by_formula", env_name="chemenv"
        )
        # Check that the returned function was called with the formula
        mock_function.assert_called_once_with(formula="C2H6O", limit=10)

        # Check the result - tool decorator converts to string
        assert isinstance(result, str)
        assert "CCO" in result
        assert "COC" in result


class TestReturnPossibleFragments:
    """Tests for the return_possible_fragments tool."""

    def test_return_possible_fragments_valid_smiles(self):
        """Test fragment generation for valid SMILES."""
        # Use a simple molecule like ethanol
        result = return_possible_fragments.execute(h_smiles="CCO")

        # Check the result - tool decorator converts to string
        assert isinstance(result, str)
        # Should contain fragment information
        assert "[" in result or "C" in result

    def test_return_possible_fragments_complex_molecule(self):
        """Test fragment generation for a more complex molecule."""
        result = return_possible_fragments.execute(h_smiles="C1=CC=CC=C1")

        # Check the result - tool decorator converts to string
        assert isinstance(result, str)

    def test_return_possible_fragments_invalid_smiles(self):
        """Test with invalid SMILES string."""
        with pytest.raises(ValueError, match="Invalid SMILES string"):
            return_possible_fragments.execute(h_smiles="INVALID")


class TestSimulateSpectra:
    """Tests for the simulate_spectra tool."""

    @patch("spectra_elucidation.tools.remote_call")
    def test_simulate_spectra_success(self, mock_remote_call):
        """Test successful spectra simulation."""
        # Mock the remote call
        mock_function = MagicMock()
        mock_function.return_value = {
            "1H NMR": "δH 1.2 (t, 3H), 3.6 (q, 2H) ppm",
            "13C NMR": "δC 10.0, 60.0 ppm",
            "IR": "3400 cm-1 (O-H), 2900 cm-1 (C-H)",
        }
        mock_remote_call.return_value = mock_function

        result = simulate_spectra.execute(smiles="CCO")

        # Check that remote_call was called with correct parameters
        mock_remote_call.assert_called_once_with(
            function_name="simulate_spectra", env_name="chemenv"
        )
        # Check that the returned function was called with the SMILES
        mock_function.assert_called_once_with(smiles="CCO")

        # Check the result - tool decorator converts to string
        assert isinstance(result, str)
        assert "1H NMR" in result
        assert "13C NMR" in result
        assert "IR" in result
