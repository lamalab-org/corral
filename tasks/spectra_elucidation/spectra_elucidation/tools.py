import asyncio
import random
import traceback
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from spectra_elucidation.pubchem_helpers import PubChem
from spectra_elucidation.spectra_utils import (
    SpectraAPI,
    convert_ms_spectrum_to_string,
    enumerate_fragments_from_smiles,
    format_hsqc_spectrum,
    predict_isotopic_distribution,
    predict_nmr_spectra,
)

from corral.core.tool import Tool, tool
from corral.utils.rag import vector_database_search


@tool
def get_formula_from_smiles(smiles: str) -> str:
    """[BRIEF] Generate a chemical formula from a SMILES string using RDKit. [/BRIEF]

    [DETAILED] This function takes a SMILES representation of a molecule and generates its chemical formula in Hill notation (C, H, then alphabetical order). It uses RDKit to parse the SMILES string and calculate the molecular formula. If the SMILES string is invalid or cannot be parsed, it returns an error message. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you have a SMILES string and need to obtain the chemical formula of the corresponding molecule.
    - When you want to validate the chemical structure of a proposed molecule.
    - When you need to convert a SMILES representation into a chemical formula for further analysis or reporting.
    - Recommended for tasks that require chemical formula generation from SMILES strings. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have the SMILES representation of a molecule that you think can produce the analysis results described in the task. Use the tools `carbon_nmr_spectra`, `proton_nmr_spectra`, `ir_spectra`, `hsqc_nmr_spectra`, and `mass_spectrometry_spectra` for obtaining the different spectra. Use the tools `retrieve_protons_shifts`, `retrieve_aromatic_protons_shifts` and `retrieve_carbon_shifts` for having more info about the chemical shifts. [/PREREQUISITE]
    2. [CURRENT] Apply this tool with a valid SMILES string to generate the chemical formula of the molecule. [/CURRENT]
    3. [FOLLOW_UP] Use the generated chemical formula to compare your proposed molecule with the molecular analysis in the task (MS results). [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It uses RDKit to parse the provided SMILES string and create a molecular object.
    - If the SMILES string is valid, it calculates the molecular formula using `rdMolDescriptors.CalcMolFormula`.
    - The formula is returned in Hill notation, which lists carbon (C) atoms first, followed by hydrogen (H) atoms, and then other elements in alphabetical order.
    - If the SMILES string is invalid or cannot be parsed, it returns an error message indicating the issue. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_formula_from_smiles("CCO")`,
        `get_formula_from_smiles("C1=CC=CC=C1")`,
        `get_formula_from_smiles("C(C(=O)O)N")`,
        `get_formula_from_smiles("C1=CC=C(C=C1)C(=O)O")`,
        `get_formula_from_smiles("C1=CC=CC=C1C(=O)O")`,
    ]
    [/SYNTACTICAL]

    Args:
        smiles (str):
            [ARGS_BRIEF] SMILES representation of a molecule [/ARGS_BRIEF]
            [ARGS_DETAILED] The SMILES string representing the chemical structure of the molecule. It should be a valid SMILES notation that RDKit can parse. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Valid SMILES string [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "C1=CC=C(C=C1)C(=O)O" [/ARGS_EXAMPLES]

    Returns:
        str:
            [RETURNS_BRIEF] The chemical formula in Hill notation (C, H, then alphabetical) [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The chemical formula of the molecule represented by the SMILES string, formatted in Hill notation. If the SMILES string is invalid or cannot be parsed, it returns an error message. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] '"C2H6O" for ethanol, "C6H6" for benzene, "C2H5NO" for acetic acid amide, "C7H6O3" for salicylic acid' [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If the SMILES string is invalid or cannot be parsed by RDKit. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when the SMILES string cannot be parsed by RDKit, indicating that it is not a valid SMILES representation of a molecule. It can occur due to syntax errors or unsupported structures in the SMILES string. [/ERROR_DETAILS]
            [ERROR_RECOVERY] If the SMILES string is invalid, try to provide a valid SMILES string. You can validate the SMILES using the `validate_smiles` tool. Otherwise do not try to solve the error. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - It may return "Invalid SMILES string" if the provided SMILES cannot be parsed.
        - RDKit may remove certain chemical features or properties during sanitization, which can sometimes lead to unexpected results.
    [/LIMITATIONS]
    """
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return "Invalid SMILES string"
        return rdMolDescriptors.CalcMolFormula(mol)

    except Exception as e:
        return f"Error: {e!s}"


@tool
def search_by_smiles(smiles: str, top_k: int = 10) -> list[dict[str, Any]]:
    """[BRIEF] Search the NMRShift database for entries matching or chemically similar to the given SMILES. [/BRIEF]

    [DETAILED] This function searches the NMRShift database for entries that match or are chemically similar to the provided SMILES string. It uses a vector database search to find the top `top_k` results based on chemical similarity. The function returns a list of matching entries, each containing relevant information about the compound. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it to validate your proposed molecule against the NMRShift database.
    - Use it when you have a SMILES string and want to find related compounds in the NMRShift database.
    - When you need to retrieve chemical shifts or other NMR-related information for a specific compound. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have a SMILES representation of the molecule you want to search for in the NMRShift database, or the SMILES of a structure that you think can be valid. [/PREREQUISITE]
    2. [CURRENT] Apply this tool with the SMILES string to search for matching or chemically similar entries in the NMRShift database. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved entries to validate your proposed molecule, compare chemical shifts, or gather additional information about the compound. You can also use the `get_formula_from_smiles` to validate the molecular formula. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It uses a vector database search to find entries in the NMRShift database that match or are chemically similar to the provided SMILES string.
    - The search is performed in the "nmrshiftdb2" collection of the NMRShift database.
    - The search uses chemical embeddings generated by the "ibm-research/MoLFormer-XL-both-10pct" model, which allows for chemical similarity searches based on the provided SMILES.
    - The function retrieves the top `top_k` results based on chemical similarity to the provided SMILES.
    - Each result contains relevant information about the compound, such as its SMILES, chemical shifts, and other properties.
    - The function returns a list of matching entries, each represented as a dictionary containing the relevant information. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `search_by_smiles("CCO")`,
        `search_by_smiles("C1=CC=CC=C1")`,
        `search_by_smiles("C(C(=O)O)N")`,
        `search_by_smiles("C1=CC=C(C=C1)C(=O)O")`,
        `search_by_smiles("C1=CC=C")`,
    ]
    [/SYNTACTICAL]

    Args:
        smiles (str):
            [ARGS_BRIEF] The SMILES representation of the compound to search for in the NMRShift database [/ARGS_BRIEF]
            [ARGS_DETAILED] The SMILES string representing the chemical structure of the molecule to search for in the NMRShift database. It should be a valid SMILES notation that can be processed by the vector database search. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Valid SMILES string [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "C1=CC=C(C=C1)C(=O)O" [/ARGS_EXAMPLES]

        top_k (int, optional):
            [ARGS_BRIEF] The maximum number of results to return. Defaults to 10 [/ARGS_BRIEF]
            [ARGS_DETAILED] The maximum number of search results to return from the NMRShift database. It should be a positive integer. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Any positive integer (e.g., 10, 20, 50) [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] 10, 20, 50 [/ARGS_EXAMPLES]

    Returns:
        list[dict[str, Any]]:
            [RETURNS_BRIEF] A list of dictionaries containing the most relevant entries from the NMRShift database [/RETURNS_BRIEF]
            [RETURNS_DETAILED] Each dictionary contains relevant information about the compound, such as its SMILES, chemical shifts, and other properties. The results are sorted by similarity score in descending order. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "[{"entry_id": "nmrshiftdb2:234", "compound_name": "Benzene", "smiles": "c1ccccc1", "spectrum": {"nucleus": "13C",...]" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If an error occurs during the search process, such as errors in the embedding of the query for example. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when there is an error in performing the vector database search, or problems with the vector database itself. It can also occur if the `top_k` parameter is not a positive integer. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try another tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function requires a valid SMILES string that can be processed by the vector database search.
        - The search results are limited to the `top_k` parameter, which defaults to 10.
        - The search is based on chemical embeddings, which may not capture all chemical similarities perfectly.
    [/LIMITATIONS]
    """
    collection_name = "nmrshiftdb2"
    db_path = Path(__file__).resolve().parents[3] / "vector_databases" / "nmrshiftdb2"

    top_k = int(top_k) if not isinstance(top_k, int) else top_k

    try:
        return vector_database_search(
            query=smiles,
            collection_name=collection_name,
            path=db_path,
            top_k=top_k,
            chemical_model="ibm-research/MoLFormer-XL-both-10pct",
        )
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}/n/nFull traceback:/n{error_details}") from e


@tool
def retrieve_protons_shifts() -> str:
    """[BRIEF] Retrieve the proton chemical shifts ranges for hydrocarbons. [/BRIEF]

    [DETAILED] This function retrieves the proton chemical shifts ranges for various types of hydrocarbons. The chemical shifts (delta) are reported in parts per million (ppm) relative to tetramethylsilane (TMS) as the reference standard. It returns a list of dictionaries, each containing the type of proton and its corresponding chemical shift range. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to know the typical chemical shifts of protons in hydrocarbons for NMR spectroscopy analysis.
    - When you want to validate the chemical shifts of protons in a proposed molecule against known ranges.
    - Recommended for tasks that require understanding the chemical environment of protons in hydrocarbons, such as NMR spectra interpretation or chemical structure elucidation. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have the proton spectra you want to analyze the proton chemical shifts. Use the tool `proton_nmr_spectra` to obtain the proton spectra. Use with the tools `retrieve_aromatic_protons_shifts` and `retrieve_carbon_shifts` for having more info about the chemical shifts. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to retrieve the proton chemical shifts ranges for hydrocarbons. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved chemical shifts to compare with the experimental NMR spectra of the sample at hand, or to validate the chemical shifts of protons in the proposed molecule. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns a predefined list of dictionaries containing the proton types and their corresponding chemical shift ranges in ppm.
    - Each dictionary contains the type of proton (e.g., aldehyde, aromatic, alkene) and its chemical shift range relative to TMS.
    - The chemical shifts are based on typical values observed in NMR spectroscopy for various types of protons in hydrocarbons.
    - The function does not perform any calculations or database queries; it simply returns the predefined list of chemical shifts. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `retrieve_protons_shifts()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] A list of dictionaries as an string containing the proton chemical shifts ranges for hydrocarbons [/RETURNS_BRIEF]
            [RETURNS_DETAILED] Each dictionary (as string) contains the type of proton and its corresponding chemical shift range in ppm. The ranges are based on typical values observed in NMR spectroscopy for various types of protons in hydrocarbons. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "[{"Proton": "Aldehyde", "delta / ppm": "9.5 - 10.5"}, {"Proton": "Aromatic", "delta / ppm": "6.5 - 8.2"}, ...]" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions. [/ERROR_WHEN]
            [ERROR_DETAILS] The tool returns precomputed reference data and has no failure modes. [/ERROR_DETAILS]
            [ERROR_RECOVERY] No recovery needed. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function returns a predefined list of chemical shifts and does not perform any calculations or database queries.
        - The chemical shifts are based on typical values and may not be applicable to all compounds or conditions.
        - The ranges provided are approximate and may vary depending on the specific molecular environment and experimental conditions.
    [/LIMITATIONS]
    """

    return str(
        [
            {"Proton": "Aldehyde", "delta / ppm": "9.5 - 10.5"},
            {"Proton": "Aromatic", "delta / ppm": "6.5 - 8.2"},
            {"Proton": "Alkene", "delta / ppm": "4.5 - 6.1"},
            {"Proton": "Alkyne", "delta / ppm": "2.0 - 3.2"},
            {"Proton": "Acetal", "delta / ppm": "4.5 - 6.0"},
            {"Proton": "Alkoxy", "delta / ppm": "3.4 - 4.8"},
            {"Proton": "Methyl (CH3-R)", "delta / ppm": "~0.9"},
            {"Proton": "N-methyl", "delta / ppm": "3.0 - 3.5"},
            {"Proton": "Methoxy", "delta / ppm": "3.3 - 3.8"},
            {
                "Proton": "CH3 attached to double bonds/aromatics",
                "delta / ppm": "1.8 - 2.5",
            },
            {"Proton": "Methyl (CH3-CO-)", "delta / ppm": "1.8 - 2.7"},
            {"Proton": "Methylene (CH2-O-)", "delta / ppm": "~3.6 - 4.7"},
            {"Proton": "Methylene (CH2-O-)", "delta / ppm": "~3.6 - 4.7"},
            {"Proton": "Methylene (CH2-R1R2)", "delta / ppm": "~1.3"},
            {"Proton": "Methine (CH-R1R2R3)", "delta / ppm": "~1.5"},
            {"Proton": "Cyclopropane", "delta / ppm": "0.22"},
            {"Proton": "Me4Si (TMS)", "delta / ppm": "0.0"},
            {"Proton": "Metal hydride", "delta / ppm": "-5 to -20"},
        ]
    )


@tool
def retrieve_aromatic_protons_shifts() -> str:
    """[BRIEF] Retrieve the proton shifts ranges for aromatic hydrocarbons. [/BRIEF]

    [DETAILED] This function retrieves the proton chemical shifts ranges for aromatic hydrocarbons. The values represent chemical shift changes (in ppm) caused by substituents on a benzene ring. The values show how much a substituent shifts the resonance of protons at ortho, meta, and para positions relative to unsubstituted benzene. Positive values indicate downfield shifts (deshielding), while negative values indicate upfield shifts (shielding). All shifts are relative to tetramethylsilane (TMS) as the reference standard. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to know the typical chemical shifts of protons in aromatic hydrocarbons for NMR spectroscopy analysis.
    - When you want to validate the chemical shifts of protons in a proposed aromatic molecule against known ranges.
    - Recommended for tasks that require understanding the chemical environment of protons in aromatic rings, such as NMR spectra interpretation or chemical structure elucidation. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have the proton spectra of the sample at hand. Complement it with the use the tools `retrieve_protons_shifts` and `retrieve_carbon_shifts` for having more info about the chemical shifts. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to retrieve the proton chemical shifts ranges for aromatic hydrocarbons. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved chemical shifts to compare with the experimental NMR spectra and propose your candidate molecules, or to validate the chemical shifts of protons in the proposed aromatic molecule. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns a predefined list of dictionaries containing the substituent effects on proton chemical shifts in aromatic rings.
    - Each dictionary contains the substituent name and its corresponding chemical shift changes (in ppm) for ortho, meta, and para positions.
    - The chemical shifts are based on typical values observed in NMR spectroscopy for various substituents on aromatic rings.
    - The function does not perform any calculations or database queries; it simply returns the predefined list of chemical shifts. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `retrieve_aromatic_protons_shifts()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] A list of dictionaries as a string, containing the substituent effects on proton chemical shifts in aromatic rings [/RETURNS_BRIEF]
            [RETURNS_DETAILED] Each dictionary contains the substituent name and its corresponding chemical shift changes (in ppm) for ortho, meta, and para positions. The shifts are based on typical values observed in NMR spectroscopy for various substituents on aromatic rings. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "[{"Substituent": "NO2", "Ortho": 0.95, "Meta": 0.17, "Para": 0.33}, {"Substituent": "CHO", "Ortho": 0.58, "Meta": 0.21, "Para": 0.27}, ...]" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions. [/ERROR_WHEN]
            [ERROR_DETAILS] The tool returns precomputed reference data and has no failure modes. [/ERROR_DETAILS]
            [ERROR_RECOVERY] No recovery needed. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function returns a predefined list of chemical shifts and does not perform any calculations or database queries.
        - The chemical shifts are based on typical values and may not be applicable to all aromatic compounds or conditions.
        - The shifts provided are approximate and may vary depending on the specific molecular environment and experimental conditions.
    [/LIMITATIONS]
    """
    return str(
        [
            {"Substituent": "NO2", "Ortho": 0.95, "Meta": 0.17, "Para": 0.33},
            {"Substituent": "CHO", "Ortho": 0.58, "Meta": 0.21, "Para": 0.27},
            {"Substituent": "COCl", "Ortho": 0.83, "Meta": 0.16, "Para": 0.30},
            {"Substituent": "COOH", "Ortho": 0.8, "Meta": 0.14, "Para": 0.2},
            {"Substituent": "COOCH3", "Ortho": 0.74, "Meta": 0.07, "Para": 0.2},
            {"Substituent": "COCH3", "Ortho": 0.64, "Meta": 0.09, "Para": 0.3},
            {"Substituent": "CN", "Ortho": 0.27, "Meta": 0.11, "Para": 0.3},
            {"Substituent": "C6H5", "Ortho": 0.18, "Meta": 0, "Para": 0.08},
            {"Substituent": "CCl3", "Ortho": 0.8, "Meta": 0.2, "Para": 0.2},
            {"Substituent": "CHCl2", "Ortho": 0.1, "Meta": 0.06, "Para": 0.1},
            {"Substituent": "CH2Cl", "Ortho": 0, "Meta": 0.01, "Para": 0},
            {"Substituent": "CH3", "Ortho": -0.17, "Meta": -0.09, "Para": -0.18},
            {"Substituent": "CH2CH3", "Ortho": -0.15, "Meta": -0.06, "Para": -0.18},
            {"Substituent": "CH(CH3)2", "Ortho": -0.14, "Meta": -0.09, "Para": -0.18},
            {"Substituent": "C(CH3)3", "Ortho": 0.01, "Meta": -0.1, "Para": -0.24},
            {"Substituent": "CH2OH", "Ortho": -0.1, "Meta": -0.1, "Para": -0.1},
            {"Substituent": "CH2NH2", "Ortho": 0, "Meta": 0, "Para": 0.22},
            {"Substituent": "F", "Ortho": -0.3, "Meta": -0.02, "Para": -0.22},
            {"Substituent": "Cl", "Ortho": 0.02, "Meta": 0.06, "Para": -0.04},
            {"Substituent": "Br", "Ortho": 0.22, "Meta": -0.13, "Para": -0.03},
            {"Substituent": "I", "Ortho": 0.4, "Meta": -0.26, "Para": -0.03},
            {"Substituent": "OCH3", "Ortho": -0.43, "Meta": -0.09, "Para": -0.37},
            {"Substituent": "OCOCH3", "Ortho": -0.21, "Meta": -0.02, "Para": -0.4},
            {"Substituent": "OH", "Ortho": -0.5, "Meta": -0.14, "Para": -0.4},
            {
                "Substituent": "p-CH3C6H4SO3",
                "Ortho": -0.26,
                "Meta": -0.05,
                "Para": -0.25,
            },
            {"Substituent": "NH2", "Ortho": -0.75, "Meta": -0.24, "Para": -0.63},
            {"Substituent": "SCH3", "Ortho": -0.03, "Meta": 0, "Para": -0.3},
            {"Substituent": "N(CH3)2", "Ortho": -0.6, "Meta": -0.1, "Para": -0.62},
        ]
    )


@tool
def retrieve_carbon_shifts() -> str:
    """[BRIEF] Retrieve the carbon chemical shifts ranges for various functional groups in organic compounds. [/BRIEF]

    [DETAILED] This function retrieves the carbon chemical shifts ranges for various functional groups in organic compounds. The chemical shifts (deltas) are reported in parts per million (ppm) relative to tetramethylsilane (TMS) as the reference standard. These values can be used to interpret 13C NMR spectra and identify carbon environments in unknown compounds. It returns a list of dictionaries, each containing the functional group and its corresponding chemical shift range. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to know the typical chemical shifts of carbon atoms in various functional groups for NMR spectroscopy analysis.
    - When you want to validate the chemical shifts of carbon atoms in a proposed molecule against known ranges.
    - Recommended for tasks that require understanding the chemical environment of carbon atoms in organic compounds, such as NMR spectra interpretation or chemical structure elucidation. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have an NMR spectrum for which you want to analyze the carbon chemical shifts. You can use the `carbon_nmr_spectra` tool to obtain the carbon spectra of the sample at hand. Use the tools `retrieve_protons_shifts` and `retrieve_aromatic_protons_shifts` for having more info about the chemical shifts. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to retrieve the carbon chemical shifts ranges for various functional groups in organic compounds. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved chemical shifts to compare with the experimental NMR spectra of the sample, or to validate the chemical shifts of carbon atoms in the proposed molecules. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns a predefined list of dictionaries containing the carbon chemical shifts ranges for different functional groups.
    - Each dictionary contains the functional group and its corresponding chemical shift range in ppm.
    - The chemical shifts are based on typical values observed in NMR spectroscopy for various functional groups in organic compounds.
    - The function does not perform any calculations or database queries; it simply returns the predefined list of chemical shifts. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `retrieve_carbon_shifts()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] A list of dictionaries as string containing the carbon chemical shifts ranges for various functional groups in organic compounds [/RETURNS_BRIEF]
            [RETURNS_DETAILED] Each dictionary contains the functional group and its corresponding chemical shift range in ppm. The ranges are based on typical values observed in NMR spectroscopy for various functional groups in organic compounds. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "[{"Group": "CH3-", "Shift (ppm)": "10-30 ppm"}, {"Group": "R3C-, R₂CH, RCH₂", "Shift (ppm)": "25-50 ppm"}, ...]" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions. [/ERROR_WHEN]
            [ERROR_DETAILS] The tool returns precomputed reference data and has no failure modes. [/ERROR_DETAILS]
            [ERROR_RECOVERY] No recovery needed. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function returns a predefined list of chemical shifts and does not perform any calculations or database queries.
        - The chemical shifts are based on typical values and may not be applicable to all compounds or conditions.
        - The ranges provided are approximate and may vary depending on the specific molecular environment and experimental conditions.
    [/LIMITATIONS]
    """
    return str(
        [
            {"Group": "CH3-", "Shift (ppm)": "10-30 ppm"},
            {"Group": "R3C-, R₂CH, RCH₂", "Shift (ppm)": "25-50 ppm"},
            {"Group": "=CH2", "Shift (ppm)": "105-120 ppm"},
            {"Group": "=CH", "Shift (ppm)": "110-140 ppm"},
            {"Group": "=CR2", "Shift (ppm)": "130-150 ppm"},
            {"Group": "Alkyne", "Shift (ppm)": "70-85 ppm"},
            {"Group": "Ar-H", "Shift (ppm)": "115-130 ppm"},
            {"Group": "Ar-C", "Shift (ppm)": "130-150 ppm"},
            {"Group": "Ketones", "Shift (ppm)": "200-210 ppm"},
            {"Group": "Aldehydes", "Shift (ppm)": "190-200 ppm"},
            {"Group": "Conjugated C=O", "Shift (ppm)": "180-200 ppm"},
            {"Group": "Carboxylic acids", "Shift (ppm)": "170-180 ppm"},
            {"Group": "Carboxylic esters", "Shift (ppm)": "160-170 ppm"},
            {"Group": "Phenols (C1)", "Shift (ppm)": "150-160 ppm"},
            {"Group": "Furans (C2)", "Shift (ppm)": "140-150 ppm"},
            {"Group": "Acetals", "Shift (ppm)": "90-110 ppm"},
            {"Group": "R3C-O", "Shift (ppm)": "70-85 ppm"},
            {"Group": "R2HC-O", "Shift (ppm)": "60-80 ppm"},
            {"Group": "RH2C-O", "Shift (ppm)": "45-65 ppm"},
            {"Group": "H3C-O", "Shift (ppm)": "50-60 ppm"},
            {"Group": "Epoxides", "Shift (ppm)": "40-60 ppm"},
        ]
    )


@tool(hidden_args=["h_smiles"])
def carbon_nmr_spectra(h_smiles: str) -> str:
    """[BRIEF] Return the 13C NMR spectra for the sample at hand. [/BRIEF]

    [DETAILED] This function measures the 13C NMR spectra for the sample at hand. It uses the `get_c13_nmr_prediction` function to measure the experiment NMR spectra. The function returns the 13C NMR spectra as a string. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you want to measure the 13C NMR spectra for the sample at hand.
    - When you need to answer questions about the carbon environments in the sample.
    - Recommended for tasks that require understanding the carbon environments in a molecule.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the correct step is to measure NMR. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to measure the 13C NMR experiment for the corresponding compound. [/CURRENT]
    3. [FOLLOW_UP] Use the resulting NMR spectra to analyze the carbon environments in the proposed molecule. You can use the `retrieve_carbon_shifts` tool to validate the carbon chemical shifts.[/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It uses the `get_c13_nmr_prediction` function to measure the 13C NMR experiment for the sample at hand.
    - This function takes the sample at hand and measures the 13C NMR spectra.
    - The function returns the 13C NMR spectra as a string, following the ACS-inspired publication format.
    - If some error occurs during the NMR spectra generation process, the function returns an error message.[/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `carbon_nmr_spectra()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] The 13C NMR spectra for the molecule in the sample at hand. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The function returns the 13C NMR spectra as a string. If some error occurs during the NMR spectra generation process, it returns an error message. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "13C NMR spectra: δC 10.0, 20.0, 30.0 ppm" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If an error occurs during the NMR spectra measurement process. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when there is an error in performing the measurement of the NMR spectra. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try another tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The cost of measuring the NMR spectra repeatedly may be high, so use it judiciously.
        - The returned NMR spectra may not be accurate for all compounds, especially for complex or unusual structures.
        - The experiment can only measure for the compound at hand.
    [/LIMITATIONS]
    """
    return asyncio.run(SpectraAPI.get_c13_nmr_prediction(h_smiles))


@tool(hidden_args=["h_smiles"])
def proton_nmr_spectra(h_smiles: str) -> str:
    """[BRIEF] Returns the 1H NMR spectra for a given SMILES string. [/BRIEF]

    [DETAILED] This function returns the 1H NMR spectra for a given SMILES string. It uses the `get_h_nmr_prediction` function to run the experiment NMR spectra. The function returns the 1H NMR spectra as a string. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you want to measure the 1H NMR spectra for a given SMILES string.
    - When you need to answer questions about the proton environments in the sample.
    - Recommended for tasks that require understanding the proton environments in a molecule. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the correct step is to measure NMR. Do not perform unnecessary experiments. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to measure the 1H NMR experiment for the corresponding compound. [/CURRENT]
    3. [FOLLOW_UP] Use the resulting NMR spectra to analyze the proton environments in the proposed molecule. You can use the `retrieve_protons_shifts` and `retrieve_aromatic_protons_shifts` tools to validate the proton chemical shifts. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]]

    [CONTEXTUAL] How this tool works:
    - This function takes the sample of the compound and measures the 1H NMR spectra.
    - The function returns the 1H NMR spectra as a string, following the ACS-inspired publication format.
    - If some error occurs during the NMR spectra generation process, the function returns an error message. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `proton_nmr_spectra()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] The 1H NMR spectra for the molecule in the sample at hand. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The function returns the 1H NMR spectra as a string, following the conventions of NMR spectra notation. If some error occurs during the NMR spectra generation process, it returns an error message. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "Predicted 1H NMR spectra: δH 7.40 (d, J = 7.9 Hz, 4H), 7.24 (s, 1H), 7.18 (dd, J = 8.0, 1.8 Hz, 4H), 7.03 (d, J = 1.4 Hz, 4H), 2.46 (s, 4H), 1.62 (s, 12H), 1.21 (s, 36H)" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If an error occurs during the NMR spectra generation process. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when there is an error in generating the NMR spectra. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try another tool. Try elucidate the protons with the other tools. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The cost of measuring the NMR spectra repeatedly may be high, so use it judiciously.
        - The experiment NMR spectra may not be accurate for all compounds, especially for complex or unusual structures.
        - The experiment can only measure the compound of the sample at hand.
    [/LIMITATIONS]
    """
    return asyncio.run(SpectraAPI.get_h_nmr_prediction(h_smiles))


@tool(hidden_args=["h_smiles"])
def ir_spectra(h_smiles: str) -> str:
    """[BRIEF] Returns the IR spectra for the sample at hand. This spectra may not be accurate for all compounds. It works best for identifying functional groups such as C=O. [/BRIEF]

    [DETAILED] The function returns the IR spectra as a string, following the conventions of IR spectra notation. If some error occurs during the IR spectra measurement process, it returns an error message. Very recommended for C=O group elucidation. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you want the IR spectra for the compound at hand.
    - When you need to answer questions about the functional groups in the sample, such as C=O stretching.
    - Recommended for tasks that require understanding the functional groups in a molecule. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the correct step is to measure IR. Do not perform unnecessary experiments. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to measure the IR spectra for the corresponding compound. [/CURRENT]
    3. [FOLLOW_UP] Use the resulting IR spectra to analyze the functional groups in the proposed molecule. You can use the `carbon_nmr_spectra` and `hsqc_nmr_spectra` tools to obtain complementary information. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - This function takes the sample of the compound and measures the IR spectra.
    - The function returns the IR spectra as a string, following a similar format as the ACS guidelines.
    - If some error occurs during the IR spectra measurement process, the function returns an error message. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `ir_spectra()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] The IR spectra for the molecule in the sample at hand. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The function returns the IR spectra as a string, following the conventions of IR spectra notation. If some error occurs during the IR spectra measurement process, it returns an error message. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "Predicted IR spectra: 3400 cm-1 (O-H stretch), 1700 cm-1 (C=O stretch), 1600 cm-1 (C=C stretch)" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If an error occurs during the IR spectra measurement process. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when there is an error in measuring the IR spectra. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try another tool. Try elucidate the functional groups with the other tools. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The cost of measuring the IR spectra repeatedly may be high, so use it judiciously.
        - The experiment IR spectra may not be accurate for all compounds, especially for complex or unusual structures.
        - The experiment can only measure the IR spectra for the compound of the sample at hand.
    [/LIMITATIONS]
    """
    return asyncio.run(SpectraAPI.get_ir_prediction(h_smiles))


@tool(hidden_args=["h_smiles"])
def hsqc_nmr_spectra(h_smiles: str) -> str:
    """[BRIEF] Returns the HSQC (Heteronuclear Single Quantum Coherence) NMR spectra for the sample at hand. [/BRIEF]

    [DETAILED] This function returns the HSQC NMR spectra for the sample at hand. It returns the HSQC spectrum data as a string, formatted in standard NMR notation, similar to the ACS conventions. If some error occurs during the measurement, it returns an appropriate message. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you want to measure the HSQC NMR spectra for the sample to elucidate its structure.
    - When you need to analyze the correlation between hydrogen and carbon atoms in the sample.
    - Recommended for tasks that require understanding the connectivity between hydrogen and carbon atoms in a molecule. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the correct step is to measure HSQC NMR. Do not perform unnecessary experiments. Use first the tools `carbon_nmr_spectra` and `proton_nmr_spectra` and try to link the fragments from there. Only measure this experiment if some of the peaks are ambiguous. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to measure the HSQC NMR experiment for the corresponding compound. [/CURRENT]
    3. [FOLLOW_UP] Use the resulting HSQC spectrum to analyze the correlation between hydrogen and carbon atoms in the proposed molecule. You can use the `retrieve_protons_shifts` and `retrieve_carbon_shifts` tools to validate the chemical shifts. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - This function measures the HSQC NMR spectra for the sample at hand.
    - It runs the local nmr-processing predictor and selects its HSQC result.
    - The function then parses the data from the measurement to extract the HSQC spectrum data and formats it in standard NMR notation, similar to the ACS convention.
    - If an error occurs during the measurement, it returns an appropriate message. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `hsqc_nmr_spectra()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] The HSQC NMR spectra for the molecule in the sample at hand. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The function returns the HSQC NMR spectra as a string, formatted in standard NMR notation. If some error occurs during the measurement, it returns an appropriate message. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "HSQC: delta H/delta C 7.40/128.0 (2H), 7.24/128.5 (2H), 7.18/129.0 (2H), 7.03/130.0 (2H), 2.46/20.0 (3H), 1.62/15.0 (6H), 1.21/10.0 (9H)." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If an error occurs during the HSQC spectrum measurement. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when there is an error in performing the HSQC spectrum measurement. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try another tool. Try elucidate the connectivity with the other tools and by using the data from the Carbon and Proton NMR. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The cost of measuring the HSQC NMR spectra repeatedly may be high, so use it judiciously.
        - The experiment HSQC spectra may not be accurate for all compounds, especially for complex or unusual structures.
        - The experiment can only measure for the compound of the sample at hand.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(h_smiles)
    if mol is None:
        return "Invalid SMILES string provided."

    spectra = predict_nmr_spectra(h_smiles)
    for spectrum in spectra["spectra"]:
        info = spectrum.get("info", {})
        pulse_sequence = info.get("pulseSequence", "")
        if pulse_sequence == "hsqc":
            return format_hsqc_spectrum(spectrum)
    return "No HSQC spectrum found for the provided SMILES."


@tool(hidden_args=["h_smiles"])
def mass_spectrometry_spectra(h_smiles: str) -> str:
    """[BRIEF] Returns the mass spectrometry spectra for the sample at hand using the Electrospray Ionization (ESI) technique. [/BRIEF]

    [DETAILED] This function returns the mass spectrometry spectra for the sample at hand by running the local isotopic distribution predictor. The function returns the mass spectrometry spectra as a string in the format "m/z 100.1 (intensity 500), 101.2 (intensity 450), ...". [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you want to measure the mass spectrometry spectra for the sample to elucidate its structure.
    - When you need to answer questions about the mass-to-charge ratio (m/z) of the sample.
    - When you need to analyze the isotopic distribution of the sample.
    - When you need to know the number of double bond equivalents in the sample.
    - Recommended for tasks that require understanding the mass spectrum of a molecule. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the correct step is to measure mass spectrometry. Do not perform unnecessary experiments. If the task is to elucidate the structure of the sample, then this tool is appropriate to begin with. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to measure the mass spectrometry experiment for the corresponding compound. [/CURRENT]
    3. [FOLLOW_UP] Use the resulting mass spectrometry spectra to analyze the mass-to-charge ratio (m/z) of the proposed molecule. You can use the `retrieve_isotope_distribution` tool to obtain complementary information about the isotopic distribution of the molecule. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It runs the local isotopic distribution predictor used by the deployed spectra service.
    - The predictor returns the mass spectrometry spectra data as JSON-compatible peak data.
    - The function then parses the response to extract the mass spectrometry spectrum data and formats it in a string format.
    - If some error occurs during the experiment, it returns an appropriate message. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mass_spectrometry_spectra()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] The mass spectrometry spectra for the molecule in the sample at hand. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The function returns the mass spectrometry spectra as a string in the format "m/z 100.1 (intensity 500), 101.2 (intensity 450), ...". If there is some error during the measurement, it returns an appropriate message. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "m/z 100.1 (intensity 500), 101.2 (intensity 450), 102.3 (intensity 400), ..." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If an error occurs during isotopic distribution prediction. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when there is an error while generating the mass spectrometry spectrum data. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try another tool. Try elucidate the mass spectrum with the other tools. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The cost of measuring the mass spectrometry spectra repeatedly may be high, so use it judiciously.
        - The experiment mass spectrometry spectra may not be accurate for all compounds, especially for complex or unusual structures.
        - The experiment can only measure for the compound of the sample at hand.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(h_smiles)
    if mol is None:
        return "Invalid SMILES string provided."

    return convert_ms_spectrum_to_string(predict_isotopic_distribution(h_smiles))


@tool
def retrieve_isotope_distribution() -> str:
    """[BRIEF] Retrieve the isotopic distribution of common elements in organic chemistry. [/BRIEF]

    [DETAILED] This function returns a predefined dictionary containing the isotopic distribution of common elements in organic chemistry, including Carbon, Hydrogen, Sulfur, Chlorine, Bromine, Iodine, Fluorine, Nitrogen, and Oxygen. Each element's isotopes are listed with their natural abundance and m/z values. The function also includes the m/z peaks for each element. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to analyze the isotopic distribution of common elements in organic compounds.
    - When you need to answer questions about the isotopic composition of a molecule.
    - Recommended for tasks that require understanding the isotopic distribution of elements in a molecule, such as mass spectrometry analysis or chemical structure elucidation. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the correct step is to retrieve the isotopic distribution. Measure the `mass_spectrometry_spectra` tool first to obtain the mass-to-charge ratio (m/z) of the molecule. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to retrieve the isotopic distribution of common elements in organic chemistry. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved isotopic distribution to analyze the mass spectrometry spectra of the proposed molecule. After that, proceed with the other spectra tools or answer the task at hand. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns a predefined dictionary containing the isotopic distribution of common elements in organic chemistry.
    - Each element's isotopes are listed with their natural abundance and m/z values.
    - The function also includes the m/z peaks for each element.
    - The isotopic distribution is based on typical values observed in organic compounds. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `retrieve_isotope_distribution()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] A string representation of a dictionary containing the isotopic distribution of common elements in organic chemistry. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The function returns a string representation of a dictionary containing the isotopic distribution of common elements in organic chemistry, including their isotopes, natural abundance, m/z values, and m/z peaks. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "{"Carbon": {"isotopes": {"12C": {"abundance": 98.89, "m/z": 12},"13C": {"abundance": 1.11, "m/z": 13}<more elements...}}}" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions. [/ERROR_WHEN]
            [ERROR_DETAILS] The tool returns precomputed reference data and has no failure modes. [/ERROR_DETAILS]
            [ERROR_RECOVERY] No recovery needed. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The isotopic distribution is based on typical values observed in organic compounds and may not be applicable to all compounds.
        - The function does not perform any calculations or database queries; it returns a predefined dictionary.
    [/LIMITATIONS]
    """
    return str(
        {
            "Carbon": {
                "isotopes": {
                    "12C": {"abundance": 98.89, "m/z": 12},
                    "13C": {"abundance": 1.11, "m/z": 13},
                },
                "m/z_peaks": ["M+1"],
            },
            "Hydrogen": {
                "isotopes": {
                    "1H": {"abundance": 99.98, "m/z": 1},
                    "2H": {"abundance": 0.02, "m/z": 2},
                },
                "m/z_peaks": ["M+1"],
            },
            "Sulfur": {
                "isotopes": {
                    "32S": {"abundance": 94.93, "m/z": 32},
                    "33S": {"abundance": 0.76, "m/z": 33},
                    "34S": {"abundance": 4.29, "m/z": 34},
                },
                "m/z_peaks": ["M+2"],
            },
            "Chlorine": {
                "isotopes": {
                    "35Cl": {"abundance": 75.77, "m/z": 35},
                    "37Cl": {"abundance": 24.23, "m/z": 37},
                },
                "m/z_peaks": ["M+2"],
            },
            "Bromine": {
                "isotopes": {
                    "79Br": {"abundance": 50.69, "m/z": 79},
                    "81Br": {"abundance": 49.31, "m/z": 81},
                },
                "m/z_peaks": ["M+2"],
            },
            "Iodine": {
                "isotopes": {
                    "127I": {"abundance": 100, "m/z": 127},
                },
                "m/z_peaks": ["M+2"],
            },
            "Fluorine": {
                "isotopes": {
                    "19F": {"abundance": 100, "m/z": 19},
                },
                "m/z_peaks": ["M+1"],
            },
            "Nitrogen": {
                "isotopes": {
                    "14N": {"abundance": 99.63, "m/z": 14},
                    "15N": {"abundance": 0.37, "m/z": 15},
                },
                "m/z_peaks": ["M+1"],
            },
            "Oxygen": {
                "isotopes": {
                    "16O": {"abundance": 99.76, "m/z": 16},
                    "17O": {"abundance": 0.04, "m/z": 17},
                    "18O": {"abundance": 0.20, "m/z": 18},
                },
                "m/z_peaks": ["M+2"],
            },
        }
    )


@tool
def retrieve_dbe_formula() -> str:
    """[BRIEF] Retrieve the formula for calculating the Double Bond Equivalent (DBE). [/BRIEF]

    [DETAILED] This function returns a string containing the formula for calculating the Double Bond Equivalent (DBE), also known as Degree of Unsaturation (DU) or Index of Hydrogen Deficiency (IHD). The formula is essential in organic chemistry for deducing molecular structures from empirical formulas. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to calculate the Double Bond Equivalent (DBE) for a given molecular formula.
    - When you need to analyze the degree of unsaturation in a molecule.
    - Recommended for tasks that require understanding the structural features of a molecule, such as chemical structure elucidation or database searches. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the correct step is to calculate the DBE. Do not perform unnecessary calculations or measurements. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to retrieve the formula for calculating the DBE. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved formula to calculate the DBE for a given molecular formula. You can use the `obtain_isomers` tool to explore different structural variations of a compound based on its DBE. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns a string containing the formula for calculating the Double Bond Equivalent (DBE).
    - The formula is based on the number of carbon, hydrogen, nitrogen, and halogen atoms in a molecule.
    - The function also provides an interpretation of the DBE values and examples of calculations for common organic compounds.
    - The DBE calculation is essential for understanding the structural features of a molecule, such as the presence of rings and multiple bonds. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `retrieve_dbe_formula()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] A string containing the formula for calculating the Double Bond Equivalent (DBE). [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The function returns a string containing the formula for calculating the Double Bond Equivalent (DBE), along with an interpretation of the DBE values and examples of calculations for common organic compounds. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "Double Bond Equivalent (DBE) = <more details>." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions. [/ERROR_WHEN]
            [ERROR_DETAILS] The tool returns precomputed reference data and has no failure modes. [/ERROR_DETAILS]
            [ERROR_RECOVERY] No recovery needed. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The formula is based on the number of carbon, hydrogen, nitrogen, and halogen atoms in a molecule, and does not account for other elements such as oxygen.
        - The interpretation of the DBE values is based on typical organic compounds and may not apply to all molecules.
    [/LIMITATIONS]
    """
    return """
Double Bond Equivalent (DBE), also known as Degree of Unsaturation (DU) or Index of Hydrogen Deficiency (IHD), quantifies the number of rings and multiple bonds (double or triple) in a molecule. This calculation is essential in organic chemistry for deducing molecular structures from empirical formulas.

Formula:
    DBE = (2C + 2 + N - H - X) / 2

Where:
    C = Number of carbon atoms
    H = Number of hydrogen atoms
    N = Number of nitrogen atoms
    X = Number of halogen atoms (F, Cl, Br, I)

Note:
    Oxygen atoms do not affect the DBE calculation.

Interpretation:
    - DBE = 0: Fully saturated molecule (no rings or multiple bonds)
    - DBE = 1: One ring or one double bond
    - DBE = 2: Two rings, two double bonds, or one ring plus one double bond
    - DBE = 3: Three rings, three double bonds, or combinations thereof

Example Calculations:
    1. C6H6 (Benzene):
        DBE = (2x6 + 2 + 0 - 6 - 0) / 2 = 4
        Interpretation: 1 ring + 3 double bonds

    2. C6H12 (Cyclohexane):
        DBE = (2x6 + 2 + 0 - 12 - 0) / 2 = 1
        Interpretation: 1 ring

    3. C6H10Cl2 (1,2-Dichlorocyclohexane):
        DBE = (2x6 + 2 + 0 - 10 - 2) / 2 = 1
        Interpretation: 1 ring

    4. C6H10 (Cyclohexene):
        DBE = (2x6 + 2 + 0 - 10 - 0) / 2 = 2
        Interpretation: 1 ring + 1 double bond
"""


@tool
def obtain_isomers_from_molecular_formula(
    molecular_formula: str, limit: int
) -> list[str]:
    """[BRIEF] Obtain isomers for a given molecular formula. [/BRIEF]

    [DETAILED] This function retrieves isomers for a given molecular formula using the local PubChem helper. It returns a list of isomer SMILES strings. The list of isomers might not be accurate since it is based on the PubChem database. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you want to find isomers for a given molecular formula.
    - When you need to explore different structural variations of a compound.
    - Recommended for tasks that require understanding the structural diversity of a molecule, such as chemical structure elucidation or database searches. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Obtain the molecular formula for the compound of interest. You can use the `get_formula_from_smiles` tool to convert a SMILES string to its molecular formula. [/PREREQUISITE]
        2. [CURRENT] Call this tool with the molecular formula to retrieve isomers. [/CURRENT]
        3. [FOLLOW_UP] Use the list of isomers for further analysis or processing. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - It uses the in-repository PubChem helper to query PubChem's PUG REST API for the given molecular formula.
        - The function returns a list of SMILES strings representing the isomers of the input compound that match the molecular formula.
        - The accuracy of the isomers is dependent on the underlying database (e.g., PubChem).
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `obtain_isomers_from_molecular_formula("C2H6O")`,
        `obtain_isomers_from_molecular_formula("C6H6")`,
        `obtain_isomers_from_molecular_formula("C6H12")`,
        `obtain_isomers_from_molecular_formula("C6H10O")`,
        `obtain_isomers_from_molecular_formula("C6H10Cl2")`,
    ]
    [/SYNTACTICAL]

    Args:
        molecular_formula (str):
            [ARGS_BRIEF] The molecular formula of the compound for which to retrieve isomers. [/ARGS_BRIEF]
            [ARGS_DETAILED] The molecular formula representing the chemical composition of the compound for which to retrieve isomers. It should be a valid molecular formula notation that can be processed by the isomer retrieval function. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Valid molecular formula string [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "C2H6O", "C6H6", "C6H12", "C6H10O", "C6H10Cl2" [/ARGS_EXAMPLES]

        limit (int):
            [ARGS_BRIEF] The maximum number of isomers to retrieve. 0 means no limit. [/ARGS_BRIEF]
            [ARGS_DETAILED] An integer specifying the maximum number of isomers to retrieve for the given molecular formula. This parameter helps to limit the number of results returned by the function. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Positive integer [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] 5, 10, 20 [/ARGS_EXAMPLES]

    Returns:
        list[str]:
            [RETURNS_BRIEF] A list of SMILES strings representing the isomers of the input compound. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The function returns a list of SMILES strings representing the isomers of the input compound. If no isomers are found, it returns an empty list. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] `["CCO", "C1=CC=CC=C1"]` [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If the PubChem API call fails or returns an error. [/ERROR_WHEN]
            [ERROR_DETAILS] This can occur due to network connectivity issues, PubChem API unavailability, an invalid molecular formula, or timeouts when querying formulas with many isomers. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify the molecular formula is valid (e.g., "C6H6" not "XYZ"). If the error is a timeout, try reducing the `limit` parameter. If it is a network issue, retry after some time. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - This tool can be really slow for molecular formulas with many isomers, if the limit is set too high.
        - The list of isomers may not be exhaustive or accurate, as it is based on the PubChem database.
        - The function may not find all possible isomers, especially for complex or unusual structures.
    [/LIMITATIONS]
    """
    return asyncio.run(
        PubChem.get_compound_isomers_by_formula(molecular_formula, limit=limit)
    )


@tool
def validate_smiles(smiles: str) -> bool:
    """[BRIEF] Validate a SMILES string to check if it represents a valid chemical structure. [/BRIEF]

    [DETAILED] This function checks if a given SMILES string can be converted into a valid chemical structure using RDKit. It returns True if the SMILES is valid, otherwise returns False. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to validate a SMILES string before performing further chemical analysis or processing.
    - When you want to ensure that the SMILES string represents a valid chemical structure before submitting it for final answer or further analysis.
    - Recommended for tasks that require checking the validity of chemical structures represented in SMILES format, such as chemical database searches or structure-based predictions. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Obtain a good guess for the molecule in the sample at hand. [/PREREQUISITE]
    2. [CURRENT] Call this tool with the SMILES string to validate it. [/CURRENT]
    3. [FOLLOW_UP] If the SMILES string is valid, proceed with further analysis or submit the final answer. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It uses RDKit's `Chem.MolFromSmiles` function to attempt to convert the SMILES string into a chemical structure.
    - If the conversion is successful, it indicates that the SMILES string is valid, and the function returns True.
    - If the conversion fails (i.e., the SMILES string is invalid), it returns False.
    - The function does not perform any chemical analysis or processing; it simply checks the validity of the SMILES string. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `validate_smiles("CCO")`,  # Valid SMILES
        `validate_smiles("C1=CC=CC=C1")`,  # Valid SMILES for benzene
        `validate_smiles("InvalidSMILES")`,  # Invalid SMILES
        `validate_smiles("C1CCCCC1")`,  # Valid SMILES for cyclohexane
        `validate_smiles("C1=CC=C(C=C1)O")`,  # Valid SMILES for phenol
    ]
    [/SYNTACTICAL]

    Args:
        smiles (str):
            [ARGS_BRIEF] The SMILES representation to validate [/ARGS_BRIEF]
            [ARGS_DETAILED] The SMILES string representing the chemical structure of the molecule to validate. It should be a valid SMILES notation that can be processed by the validation function. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] "valid SMILES string" [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "C1=CC=C(C=C1)C(=O)O" [/ARGS_EXAMPLES]

    Returns:
        bool:
            [RETURNS_BRIEF] True if the SMILES string is valid, False otherwise. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The function returns True if the SMILES string can be converted into a valid chemical structure, otherwise it returns False. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] `True`, `False` [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions. [/ERROR_WHEN]
            [ERROR_DETAILS] The tool returns precomputed reference data and has no failure modes. [/ERROR_DETAILS]
            [ERROR_RECOVERY] No recovery needed. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function only checks the validity of the SMILES string; it does not perform any chemical analysis or processing.
        - It relies on RDKit's ability to parse SMILES strings, so any limitations of RDKit's SMILES parser will apply.
        - The function does not handle specific chemical properties or characteristics; it simply checks if the SMILES string can be converted into a valid chemical structure.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


@tool(hidden_args=["h_smiles"])
def return_possible_fragments(h_smiles: str) -> list[str]:
    """[BRIEF] Return a list of fragments of the sample at hand by removing one or two atoms from the molecule. [/BRIEF]

    [DETAILED] This function generates some fragments from the sample at hand. It returns a list of unique fragment SMILES strings. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to generate fragments for the sample at hand.
    - When you want to explore different possible options based on the spectra.
    - When you are at an endpoint and need to consider some potential fragments. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the correct step is to generate fragments. Ensure that you really tried to guess all the posibilities from the information available. [/PREREQUISITE]
    2. [CURRENT] Call this tool to generate fragments for the sample at hand. [/CURRENT]
    3. [FOLLOW_UP] Evaluate the fragments for getting the final molecule. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - This tool takes the spectra information and generates fragments based on known molecules.
    - It uses a combination of cheminformatics techniques to identify potential fragment structures.
    - It leverages existing databases and algorithms to find some candidates.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `return_possible_fragments()`
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        list[str]:
            [RETURNS_BRIEF] A list of unique fragment SMILES strings. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] This list contains all the unique SMILES representations of the fragments found by database lookup. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] Example SMILES strings: ["C1=CC=CC=C1", "C1=CC=CC=C1O", ...] [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the provided SMILES string is invalid or cannot be parsed by RDKit. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when `Chem.MolFromSmiles` returns None, indicating the input SMILES does not represent a valid molecular structure. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify the SMILES string is valid using the `validate_smiles` tool before calling this tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - This tool might not return all the fragments, but there the ones returned are 100% accurate.
        - The quality of the generated fragments depends on the underlying database and its coverage.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(h_smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    fragments = enumerate_fragments_from_smiles(h_smiles)

    final_fragments = []
    for fragment in fragments:
        try:
            fragment_mol = Chem.MolFromSmiles(fragment)
        except Exception:
            continue
        if fragment_mol is None:
            continue
        final_fragments.append(fragment)

    random.shuffle(final_fragments)
    return final_fragments


@tool
def simulate_spectra(smiles: str) -> dict[str, str]:
    """[BRIEF] Simulate 1H NMR, 13C NMR, and IR spectra for a given molecule using its SMILES string, that allows validation of a proposed candidate. [/BRIEF]

    [DETAILED] This function simulates the 1H NMR, 13C NMR, and IR spectra for a molecule represented by its SMILES string.
    It uses the in-repository `SpectraAPI` helper to request and format the predictions from the configured public NMR and IR services. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it to validate the chemical structure of a proposed molecule by simulating its spectra.
    - When you want to validate some hypothetical molecule against the experimental data in the task description.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Run the spectra tools, and analyze the results thoroughly. Generate different candidate molecules and reason which ones could fit the spectra, until you have a good guess for the molecule in the sample at hand. [/PREREQUISITE]
    2. [CURRENT] Apply this tool with the SMILES string of the proposed molecule to simulate its spectra and validate if can be the solution to the task. [/CURRENT]
    3. [FOLLOW_UP] Submit the answer if the simulated spectra is similar to the experimental, or go back to step 1 and propose new candidate molecules. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It uses `SpectraAPI.get_all_predictions` to request 1H NMR, 13C NMR, and IR predictions concurrently.
    - The helper formats each successful response into a concise literature-style string.
    - The function returns a dictionary containing `h_nmr`, `c13_nmr`, and `ir` entries.
    - If a prediction is unavailable, its entry contains a failure message while successful predictions are preserved.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `simulate_spectra("CCO")`,
        `simulate_spectra("C1=CC=CC=C1")`,
        `simulate_spectra("C(C(=O)O)N")`,
        `simulate_spectra("C1=CC=C(C=C1)C(=O)O")`,
        `simulate_spectra("C1=CC=C")`,
    ]
    [/SYNTACTICAL]

    Args:
        smiles (str):
            [ARGS_BRIEF] The SMILES representation of the compound to simulate spectra for [/ARGS_BRIEF]
            [ARGS_DETAILED] The SMILES string representing the chemical structure of the molecule for which the spectra will be simulated.
            It should be a valid SMILES notation accepted by the configured prediction services. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: "valid SMILES string" [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "C1=CC=C(C=C1)C(=O)O" [/ARGS_EXAMPLES]

    Returns:
        dict[str, str]:
            [RETURNS_BRIEF] The simulated spectra of the compound [/RETURNS_BRIEF]
            [RETURNS_DETAILED] A dictionary containing the simulated spectra for 1H NMR, 13C NMR, and IR.
            Each key corresponds to a type of spectrum, and the value is a string representation of the simulated spectrum.
            If some spectra are not available, the value will be None for those keys. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] Examples: {"h_nmr": "simulated_1H_NMR_spectrum", "c13_nmr": "simulated_13C_NMR_spectrum", "ir": "simulated_IR_spectrum"} [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If a network error occurs while contacting a prediction service. [/ERROR_WHEN]
            [ERROR_DETAILS] Individual service failures are returned in the corresponding result entry. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Retry later or use the available individual spectrum tools. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The SMILES string must be valid and represent a structure accepted by the prediction services.
        - The services may not be able to simulate spectra for all compounds, especially if they are complex or not well-defined.
        - The function relies on public prediction services whose availability and behavior may change over time.
    [/LIMITATIONS]
    """
    return asyncio.run(SpectraAPI.get_all_predictions(smiles))


def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    return {
        "get_formula_from_smiles": get_formula_from_smiles,
        "retrieve_protons_shifts": retrieve_protons_shifts,
        "retrieve_aromatic_protons_shifts": retrieve_aromatic_protons_shifts,
        "retrieve_carbon_shifts": retrieve_carbon_shifts,
        "search_by_smiles": search_by_smiles,
        "carbon_nmr_spectra": carbon_nmr_spectra,
        "proton_nmr_spectra": proton_nmr_spectra,
        "ir_spectra": ir_spectra,
        "hsqc_nmr_spectra": hsqc_nmr_spectra,
        "mass_spectrometry_spectra": mass_spectrometry_spectra,
        "retrieve_isotope_distribution": retrieve_isotope_distribution,
        "retrieve_dbe_formula": retrieve_dbe_formula,
        "obtain_isomers_from_molecular_formula": obtain_isomers_from_molecular_formula,
        "validate_smiles": validate_smiles,
        "return_possible_fragments": return_possible_fragments,
        "simulate_spectra": simulate_spectra,
    }
