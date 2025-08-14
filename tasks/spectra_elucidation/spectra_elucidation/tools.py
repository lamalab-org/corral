import random
import re
import traceback
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import modal
import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from corral.base import Tool
from corral.utils import (
    remote_call,
    tool,
    vector_database_search,
)

_ELEMENT_PAT = re.compile(r"([A-Z][a-z]?)(\d*)")
_PAREN_PAT = re.compile(r"\(([^()]*)\)(\d*)")
_DOT_PAT = re.compile(r"·|\.")

get_isomers = modal.Function.lookup("chemenv", "get_compound_isomers_pubchem")
get_proton_spectrum = modal.Function.lookup("chemenv", "get_h_nmr_prediction")
get_carbon_spectrum = modal.Function.lookup("chemenv", "get_c13_nmr_prediction")


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
            [BRIEF] SMILES representation of a molecule [/BRIEF]
            [DETAILED] The SMILES string representing the chemical structure of the molecule. It should be a valid SMILES notation that RDKit can parse. [/DETAILED]
            [SYNTACTICAL] Valid SMILES string [/SYNTACTICAL]
            [EXAMPLES] "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "C1=CC=C(C=C1)C(=O)O" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] The chemical formula in Hill notation (C, H, then alphabetical) [/BRIEF]
            [DETAILED] The chemical formula of the molecule represented by the SMILES string, formatted in Hill notation. If the SMILES string is invalid or cannot be parsed, it returns an error message. [/DETAILED]
            [EXAMPLES] '"C2H6O" for ethanol, "C6H6" for benzene, "C2H5NO" for acetic acid amide, "C7H6O3" for salicylic acid' [/EXAMPLES]

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
            [BRIEF] The SMILES representation of the compound to search for in the NMRShift database [/BRIEF]
            [DETAILED] The SMILES string representing the chemical structure of the molecule to search for in the NMRShift database. It should be a valid SMILES notation that can be processed by the vector database search. [/DETAILED]
            [SYNTACTICAL] Valid SMILES string [/SYNTACTICAL]
            [EXAMPLES] "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "C1=CC=C(C=C1)C(=O)O" [/EXAMPLES]

        top_k (int, optional):
            [BRIEF] The maximum number of results to return. Defaults to 10 [/BRIEF]
            [DETAILED] The maximum number of search results to return from the NMRShift database. It should be a positive integer. [/DETAILED]
            [SYNTACTICAL] Any positive integer (e.g., 10, 20, 50) [/SYNTACTICAL]
            [EXAMPLES] 10, 20, 50 [/EXAMPLES]

    Returns:
        list[dict[str, Any]]:
            [BRIEF] A list of dictionaries containing the most relevant entries from the NMRShift database [/BRIEF]
            [DETAILED] Each dictionary contains relevant information about the compound, such as its SMILES, chemical shifts, and other properties. The results are sorted by similarity score in descending order. [/DETAILED]
            [EXAMPLES] "[{"entry_id": "nmrshiftdb2:234", "compound_name": "Benzene", "smiles": "c1ccccc1", "spectrum": {"nucleus": "13C",...]" [/EXAMPLES]

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
            [BRIEF] A list of dictionaries as an string containing the proton chemical shifts ranges for hydrocarbons [/BRIEF]
            [DETAILED] Each dictionary (as string) contains the type of proton and its corresponding chemical shift range in ppm. The ranges are based on typical values observed in NMR spectroscopy for various types of protons in hydrocarbons. [/DETAILED]
            [EXAMPLES] "[{"Proton": "Aldehyde", "delta / ppm": "9.5 - 10.5"}, {"Proton": "Aromatic", "delta / ppm": "6.5 - 8.2"}, ...]" [/EXAMPLES]

    [RAISES] Exceptions:
        None
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
            [BRIEF] A list of dictionaries as a string, containing the substituent effects on proton chemical shifts in aromatic rings [/BRIEF]
            [DETAILED] Each dictionary contains the substituent name and its corresponding chemical shift changes (in ppm) for ortho, meta, and para positions. The shifts are based on typical values observed in NMR spectroscopy for various substituents on aromatic rings. [/DETAILED]
            [EXAMPLES] "[{"Substituent": "NO2", "Ortho": 0.95, "Meta": 0.17, "Para": 0.33}, {"Substituent": "CHO", "Ortho": 0.58, "Meta": 0.21, "Para": 0.27}, ...]" [/EXAMPLES]

    [RAISES] Exceptions:
        None
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
            [BRIEF] A list of dictionaries as string containing the carbon chemical shifts ranges for various functional groups in organic compounds [/BRIEF]
            [DETAILED] Each dictionary contains the functional group and its corresponding chemical shift range in ppm. The ranges are based on typical values observed in NMR spectroscopy for various functional groups in organic compounds. [/DETAILED]
            [EXAMPLES] "[{"Group": "CH3-", "Shift (ppm)": "10-30 ppm"}, {"Group": "R3C-, R₂CH, RCH₂", "Shift (ppm)": "25-50 ppm"}, ...]" [/EXAMPLES]

    [RAISES] Exceptions:
        None
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
            [BRIEF] The 13C NMR spectra for the molecule in the sample at hand. [/BRIEF]
            [DETAILED] The function returns the 13C NMR spectra as a string. If some error occurs during the NMR spectra generation process, it returns an error message. [/DETAILED]
            [EXAMPLES] "13C NMR spectra: δC 10.0, 20.0, 30.0 ppm" [/EXAMPLES]

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
    return remote_call(function_name="get_c13_nmr_prediction", env_name="chemenv")(
        smiles=h_smiles
    )


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
            [BRIEF] The 1H NMR spectra for the molecule in the sample at hand. [/BRIEF]
            [DETAILED] The function returns the 1H NMR spectra as a string, following the conventions of NMR spectra notation. If some error occurs during the NMR spectra generation process, it returns an error message. [/DETAILED]
            [EXAMPLES] "Predicted 1H NMR spectra: δH 7.40 (d, J = 7.9 Hz, 4H), 7.24 (s, 1H), 7.18 (dd, J = 8.0, 1.8 Hz, 4H), 7.03 (d, J = 1.4 Hz, 4H), 2.46 (s, 4H), 1.62 (s, 12H), 1.21 (s, 36H)" [/EXAMPLES]

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
    return remote_call(function_name="get_h_nmr_prediction", env_name="chemenv")(
        smiles=h_smiles
    )


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
            [BRIEF] The IR spectra for the molecule in the sample at hand. [/BRIEF]
            [DETAILED] The function returns the IR spectra as a string, following the conventions of IR spectra notation. If some error occurs during the IR spectra measurement process, it returns an error message. [/DETAILED]
            [EXAMPLES] "Predicted IR spectra: 3400 cm-1 (O-H stretch), 1700 cm-1 (C=O stretch), 1600 cm-1 (C=C stretch)" [/EXAMPLES]

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
    return remote_call(function_name="get_ir_prediction", env_name="chemenv")(
        smiles=h_smiles
    )


def make_api_call(url: str, payload: dict) -> dict:
    """
    Make a POST request to the specified URL with the given payload.

    Args:
        url (str): The URL to which the request is sent.
        payload (dict): The data to be sent in the request body.

    Returns:
        dict: The JSON response from the server.
    """
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def format_hsqc_spectrum(zones_dict: dict) -> str:
    """
    Parse HSQC spectra data from dictionary format to standard NMR notation.

    Args:
        zones_dict (dict): Dictionary containing zones data with signals
        frequency (str): NMR frequency (default: "600 MHz")
        solvent (str): NMR solvent (default: "DMSO-d6")

    Returns:
        str: Formatted HSQC notation string
    """
    if "zones" not in zones_dict or "values" not in zones_dict["zones"]:
        return "Invalid input format"

    signals_data = []

    # Extract signals from each zone
    for zone in zones_dict["zones"]["values"]:
        if "signals" in zone:
            for signal in zone["signals"]:
                # Extract chemical shifts
                h_delta = signal["x"]["delta"]  # 1H chemical shift
                c_delta = signal["y"]["delta"]  # 13C chemical shift

                # Count hydrogen atoms from x.atoms (1H nuclei)
                h_count = len(signal["x"]["atoms"]) if "atoms" in signal["x"] else 1

                # Store the data for sorting
                signals_data.append(
                    {"h_delta": h_delta, "c_delta": c_delta, "h_count": h_count}
                )

    # Sort signals by 1H chemical shift (ascending order)
    signals_data.sort(key=lambda x: x["h_delta"])

    # Format each signal
    formatted_signals = []
    for signal in signals_data:
        h_delta = signal["h_delta"]
        c_delta = signal["c_delta"]
        h_count = signal["h_count"]

        # Format chemical shifts (remove trailing zeros)
        h_str = f"{h_delta:.2f}".rstrip("0").rstrip(".")
        c_str = f"{c_delta:.1f}".rstrip("0").rstrip(".")

        # Create the signal string
        signal_str = f"{h_str}/{c_str} ({h_count}H)"
        formatted_signals.append(signal_str)

    # Combine all signals
    signals_part = ", ".join(formatted_signals)

    # Create final HSQC string
    return f"HSQC: delta H/delta C {signals_part}."


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
    - It makes a POST request to an external API that measures the HSQC NMR experiment for the sample at hand.
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
            [BRIEF] The HSQC NMR spectra for the molecule in the sample at hand. [/BRIEF]
            [DETAILED] The function returns the HSQC NMR spectra as a string, formatted in standard NMR notation. If some error occurs during the measurement, it returns an appropriate message. [/DETAILED]
            [EXAMPLES] "HSQC: delta H/delta C 7.40/128.0 (2H), 7.24/128.5 (2H), 7.18/129.0 (2H), 7.03/130.0 (2H), 2.46/20.0 (3H), 1.62/15.0 (6H), 1.21/10.0 (9H)." [/EXAMPLES]

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
    URL = "https://lamalab-org--nmr-prediction-api-predict-nmr.modal.run"
    mol = Chem.MolFromSmiles(h_smiles)
    if mol is None:
        return "Invalid SMILES string provided."
    payload = {"smiles": h_smiles}

    spectra = make_api_call(URL, payload)
    for spectrum in spectra["spectra"]:
        info = spectrum.get("info", {})
        pulse_sequence = info.get("pulseSequence", "")
        if pulse_sequence == "hsqc":
            return format_hsqc_spectrum(spectrum)
    return "No HSQC spectrum found for the provided SMILES."


def convert_ms_spectrum_to_string(spectrum_data):
    """
    Convert MS spectrum data from JSON format to string format.

    Args:
        spectrum_data (list): List of dictionaries with 'x' (m/z) and 'y' (intensity) keys

    Returns:
        str: Formatted string in the format "m/z 100.1 (intensity 500), 101.2 (intensity 450), ..."
    """
    if not spectrum_data:
        return ""

    # Convert each data point to the desired format
    formatted_peaks = []
    for peak in spectrum_data:
        mz = peak["x"]
        intensity = peak["y"]
        formatted_peaks.append(f"{mz} (intensity {intensity})")

    # Join all peaks with ", " and prepend "m/z "
    return "m/z " + ", ".join(formatted_peaks)


@tool(hidden_args=["h_smiles"])
def mass_spectrometry_spectra(h_smiles: str) -> str:
    """[BRIEF] Returns the mass spectrometry spectra for the sample at hand using the Electrospray Ionization (ESI) technique. [/BRIEF]

    [DETAILED] This function returns the mass spectrometry spectra for the sample at hand by making a POST request to an external API that measures the mass spectrometry experiment. The function returns the mass spectrometry spectra as a string in the format "m/z 100.1 (intensity 500), 101.2 (intensity 450), ...". [/DETAILED]

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
    - It makes a POST request to an external API that measures the mass spectrometry experiment for the sample at hand.
    - The API returns the mass spectrometry spectra data as a JSON response.
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
            [BRIEF] The mass spectrometry spectra for the molecule in the sample at hand. [/BRIEF]
            [DETAILED] The function returns the mass spectrometry spectra as a string in the format "m/z 100.1 (intensity 500), 101.2 (intensity 450), ...". If there is some error during the measurement, it returns an appropriate message. [/DETAILED]
            [EXAMPLES] "m/z 100.1 (intensity 500), 101.2 (intensity 450), 102.3 (intensity 400), ..." [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If an error occurs during the API call to retrieve the mass spectrometry spectrum data. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when there is an error in performing the API call to retrieve the mass spectrometry spectrum data. [/ERROR_DETAILS]
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

    URL = "https://lamalab-org--nmr-prediction-api-predict-isotopic-distribution.modal.run"
    payload = {"smiles": h_smiles}

    return convert_ms_spectrum_to_string(make_api_call(URL, payload))


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
            [BRIEF] A string representation of a dictionary containing the isotopic distribution of common elements in organic chemistry. [/BRIEF]
            [DETAILED] The function returns a string representation of a dictionary containing the isotopic distribution of common elements in organic chemistry, including their isotopes, natural abundance, m/z values, and m/z peaks. [/DETAILED]
            [EXAMPLES] "{"Carbon": {"isotopes": {"12C": {"abundance": 98.89, "m/z": 12},"13C": {"abundance": 1.11, "m/z": 13}<more elements...}}}" [/EXAMPLES]

    [RAISES] Exceptions:
        None
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
            [BRIEF] A string containing the formula for calculating the Double Bond Equivalent (DBE). [/BRIEF]
            [DETAILED] The function returns a string containing the formula for calculating the Double Bond Equivalent (DBE), along with an interpretation of the DBE values and examples of calculations for common organic compounds. [/DETAILED]
            [EXAMPLES] "Double Bond Equivalent (DBE) = <more details>." [/EXAMPLES]

    [RAISES] Exceptions:
        None
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
def obtain_isomers_from_molecular_formula(molecular_formula: str) -> list[str]:
    """[BRIEF] Obtain isomers for a given molecular formula. [/BRIEF]

    [DETAILED] This function retrieves isomers for a given molecular formula using the `get_isomers_from_molecular_formula` remote function. It returns a list of isomer SMILES strings. The list of isomers might not be accurate since it is based on the PubChem database. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you want to find isomers for a given molecular formula.
    - When you need to explore different structural variations of a compound.
    - Recommended for tasks that require understanding the structural diversity of a molecule, such as chemical structure elucidation or database searches. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. Obtain the molecular formula for the compound of interest. You can use the `get_formula_from_smiles` tool to convert a SMILES string to its molecular formula.
        2. Call this tool with the molecular formula to retrieve isomers.
        3. Use the list of isomers for further analysis or processing.
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - It uses the `get_isomers_from_molecular_formula` remote function to retrieve isomers for the given molecular formula.
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
            [BRIEF] The molecular formula of the compound for which to retrieve isomers. [/BRIEF]
            [DETAILED] The molecular formula representing the chemical composition of the compound for which to retrieve isomers. It should be a valid molecular formula notation that can be processed by the isomer retrieval function. [/DETAILED]
            [SYNTACTICAL] Valid molecular formula string [/SYNTACTICAL]
            [EXAMPLES] "C2H6O", "C6H6", "C6H12", "C6H10O", "C6H10Cl2" [/EXAMPLES]

    Returns:
        list[str]:
            [BRIEF] A list of SMILES strings representing the isomers of the input compound. [/BRIEF]
            [DETAILED] The function returns a list of SMILES strings representing the isomers of the input compound. If no isomers are found, it returns an empty list. [/DETAILED]
            [EXAMPLES] `["CCO", "C1=CC=CC=C1"]` [/EXAMPLES]

    [RAISES] Exceptions:
        None
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The list of isomers may not be exhaustive or accurate, as it is based on the PubChem database.
        - The function may not find all possible isomers, especially for complex or unusual structures.
    [/LIMITATIONS]
    """
    return remote_call(
        function_name="get_compound_isomers_pubchem_by_formula", env_name="chemenv"
    )(formula=molecular_formula)


@tool
def validate_smiles(smiles: str) -> bool:
    """[BRIEF] Validate a SMILES string to check if it represents a valid chemical structure. [/BRIEF]

    [DETAILED] This function checks if a given SMILES string can be converted into a valid chemical structure using RDKit. It returns True if the SMILES is valid, otherwise returns False. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to validate a SMILES string before performing further chemical analysis or processing.
    - When you want to ensure that the SMILES string represents a valid chemical structure before submitting it for final answer or further analysis.
    - Recommended for tasks that require checking the validity of chemical structures represented in SMILES format, such as chemical database searches or structure-based predictions. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. Obtain a good guess for the molecule in the sample at hand.
    2. Call this tool with the SMILES string to validate it.
    3. If the SMILES string is valid, proceed with further analysis or submit the final answer.
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
            [BRIEF] The SMILES representation to validate [/BRIEF]
            [DETAILED] The SMILES string representing the chemical structure of the molecule to validate. It should be a valid SMILES notation that can be processed by the validation function. [/DETAILED]
            [SYNTACTICAL] "valid SMILES string" [/SYNTACTICAL]
            [EXAMPLES] "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "C1=CC=C(C=C1)C(=O)O" [/EXAMPLES]

    Returns:
        bool:
            [BRIEF] True if the SMILES string is valid, False otherwise. [/BRIEF]
            [DETAILED] The function returns True if the SMILES string can be converted into a valid chemical structure, otherwise it returns False. [/DETAILED]
            [EXAMPLES] `True`, `False` [/EXAMPLES]

    [RAISES] Exceptions:
        None
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function only checks the validity of the SMILES string; it does not perform any chemical analysis or processing.
        - It relies on RDKit's ability to parse SMILES strings, so any limitations of RDKit's SMILES parser will apply.
        - The function does not handle specific chemical properties or characteristics; it simply checks if the SMILES string can be converted into a valid chemical structure.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def differs_by_one_atom(parent_formula, fragment_formula):
    """Check if fragment differs from parent by exactly one atom"""
    total_diff = 0
    all_elements = set(parent_formula.keys()) | set(fragment_formula.keys())

    for element in all_elements:
        parent_count = parent_formula.get(element, 0)
        fragment_count = fragment_formula.get(element, 0)
        diff = abs(parent_count - fragment_count)
        total_diff += diff

    # If total difference is exactly 1, it means one atom was added/removed
    return total_diff == 1 or total_diff == 0


def _parse_simple(formula: str) -> dict[str, int]:
    """
    Parse an already-expanded, dot-free formula into {element: count}.
    """
    counts = Counter()
    for el, cnt in _ELEMENT_PAT.findall(formula):
        counts[el] += int(cnt or 1)
    return counts


def _expand_parentheses(formula: str) -> str:
    """
    Recursively expand parentheses so that C6H5(CH3) becomes C6H5C1H3 etc.
    """
    while True:
        m = _PAREN_PAT.search(formula)
        if not m:
            return formula
        inner, mult = m.groups()
        mult = int(mult or 1)
        expanded = "".join(
            f"{el}{int(cnt or 1)*mult}" for el, cnt in _ELEMENT_PAT.findall(inner)
        )
        formula = formula[: m.start()] + expanded + formula[m.end() :]


def parse_molecular_formula(formula: str) -> dict[str, int]:
    """
    Parse molecular formula into an element-count mapping, handling
    parentheses and dot adducts.
    """
    parts = _DOT_PAT.split(formula.replace(" ", ""))
    total = Counter()
    for part in parts:
        expanded = _expand_parentheses(part)
        total += _parse_simple(expanded)
    return dict(total)


def enumerate_fragments_from_smiles(
    smi: str,
    max_cuts: int = 2,
    skip_ring_bonds: bool = True,
    min_heavy_atoms: int = 2,
    max_combos: int = 100000,
):
    """
    Generate fragment SMILES by breaking up to `max_cuts` bonds.
    Returns a sorted list of unique canonical SMILES of fragments.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # Candidate bonds: heavy-atom bonds only; optionally skip ring bonds
    cand_bond_idxs = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        if skip_ring_bonds and b.IsInRing():
            continue
        cand_bond_idxs.append(b.GetIdx())

    # Early exit: nothing to cut
    if not cand_bond_idxs or max_cuts < 1:
        # Still return the whole molecule as one "fragment"
        return [Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)]

    uniq = set()
    tried = 0

    # Always include the intact molecule as a "fragment"
    uniq.add(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))

    for ncuts in range(1, min(max_cuts, len(cand_bond_idxs)) + 1):
        for cutset in combinations(cand_bond_idxs, ncuts):
            tried += 1
            if tried > max_combos:
                # Safety valve against combinatorial explosion
                break

            # Make a copy and remove the selected bonds
            rw = Chem.RWMol(mol)
            for bidx in cutset:
                b = rw.GetBondWithIdx(bidx)
                rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

            # Get connected components as fragments; don't resanitize each time
            frags = Chem.rdmolops.GetMolFrags(
                rw.GetMol(), asMols=True, sanitizeFrags=False
            )

            for frag in frags:
                # Filter tiny pieces
                if (
                    sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() > 1)
                    < min_heavy_atoms
                ):
                    continue

                # You can optionally sanitize; radicals/valence issues are fine for MS-like fragments
                # Chem.SanitizeMol(frag, catchErrors=True)  # optional
                smi_frag = Chem.MolToSmiles(frag, isomericSmiles=True, canonical=True)
                uniq.add(smi_frag)
        else:
            continue
        break  # broke due to max_combos

    return sorted(uniq)


@tool(hidden_args=["h_smiles"])
def return_possible_fragments(h_smiles: str) -> list[str]:
    """[BRIEF] Return possible fragments for the sample at hand. [/BRIEF]

    [DETAILED] This function generates some possible fragments by web lookup, and similarity check with similar spectra. It returns a list of unique fragment SMILES strings. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to generate possible fragments for the sample at hand.
    - When you want to explore different possible options based on the spectra.
    - When you are at an endpoint and need to consider some potential fragments. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the correct step is to generate possible fragments. Ensure that you really tried to guess all the posibilities from the information available. [/PREREQUISITE]
    2. [CURRENT] Call this tool to generate possible fragments for the sample at hand. [/CURRENT]
    3. [FOLLOW_UP] Evaluate the generated fragments for relevance and potential. Use the fragments that you consider most promising for further analysis. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - This tool takes the spectra information and generates possible fragments based on known molecules.
    - It uses a combination of cheminformatics techniques to identify potential fragment structures.
    - It leverages existing databases and algorithms to find some possible candidates.
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
            [BRIEF] A list of unique fragment SMILES strings. [/BRIEF]
            [DETAILED] This list contains all the unique SMILES representations of the candidate fragments found by database lookup. [/DETAILED]
            [EXAMPLES] Example SMILES strings: ["C1=CC=CC=C1", "C1=CC=CC=C1O", ...] [/EXAMPLES]

    [RAISES] Exceptions:
        None
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function may not find all possible fragments, especially for complex molecules.
        - The quality of the generated fragments depends on the underlying database and its coverage.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(h_smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    parent_formula = CalcMolFormula(mol)

    fragments = enumerate_fragments_from_smiles(h_smiles)

    final_fragments = []
    isomers = []
    for fragment in fragments:
        try:
            fragment_mol = Chem.MolFromSmiles(fragment)
        except Exception:
            continue
        if fragment_mol is None:
            continue
        fragment_formula = CalcMolFormula(fragment_mol)
        if not differs_by_one_atom(
            parse_molecular_formula(parent_formula),
            parse_molecular_formula(fragment_formula),
        ):
            try:
                isomers.extend(get_isomers.remote(fragment))
            except Exception:
                continue
            final_fragments.append(fragment)

    target_len = 3 * len(fragments)
    while len(final_fragments) < target_len and isomers:
        isomer = random.choice(isomers)
        mol = Chem.MolFromSmiles(isomer)
        if mol is not None:
            final_fragments.append(isomer)

    random.shuffle(final_fragments)
    return final_fragments


def parse_c13_nmr(c13_data: str) -> list[float]:
    """Parse C13 NMR delta values from string."""
    if not c13_data or "failed" in c13_data.lower():
        return []

    # Extract numbers from the delta string
    deltas = re.findall(r"\d+\.\d+", c13_data)
    return [float(d) for d in deltas]


def parse_h_nmr(h_data: str) -> list[float]:
    """Parse H NMR delta values from string."""
    if not h_data or "failed" in h_data.lower():
        return []

    # Extract delta values (numbers before parentheses or commas)
    deltas = re.findall(r"(\d+\.\d+)\s*\(", h_data)
    return [float(d) for d in deltas]


def compare_nmr_spectra(
    ground_spectra: dict, fragment_spectra: dict, tolerance: float = 0.5
) -> str:
    """
    Compare NMR spectra between ground truth and fragment.
    """
    results = []

    # Compare C13 NMR
    ground_c13 = parse_c13_nmr(ground_spectra.get("c13_nmr", ""))
    fragment_c13 = parse_c13_nmr(fragment_spectra.get("c13_nmr", ""))

    if not ground_c13 and not fragment_c13:
        c13_match = "C13 NMR: Both spectra failed - no comparison possible"
    elif not ground_c13:
        c13_match = "C13 NMR: Ground truth failed - no comparison possible"
    elif not fragment_c13:
        c13_match = "C13 NMR: Fragment failed - no comparison possible"
    else:
        # Check if fragment signals are subset of ground truth signals
        matched_signals = 0
        total_fragment_signals = len(fragment_c13)

        for frag_delta in fragment_c13:
            for ground_delta in ground_c13:
                if abs(frag_delta - ground_delta) <= tolerance:
                    matched_signals += 1
                    break

        if matched_signals == total_fragment_signals and total_fragment_signals > 0:
            c13_match = f"C13 NMR: MATCHED - All {total_fragment_signals} fragment signals found in ground truth"
        else:
            c13_match = f"C13 NMR: NOT MATCHED - Only {matched_signals}/{total_fragment_signals} fragment signals found in ground truth"

    results.append(c13_match)

    # Compare H NMR
    ground_h = parse_h_nmr(ground_spectra.get("h_nmr", ""))
    fragment_h = parse_h_nmr(fragment_spectra.get("h_nmr", ""))

    if not ground_h and not fragment_h:
        h_match = "H NMR: Both spectra failed - no comparison possible"
    elif not ground_h:
        h_match = "H NMR: Ground truth failed - no comparison possible"
    elif not fragment_h:
        h_match = "H NMR: Fragment failed - no comparison possible"
    else:
        # Check if fragment signals are subset of ground truth signals
        matched_signals = 0
        total_fragment_signals = len(fragment_h)

        for frag_delta in fragment_h:
            for ground_delta in ground_h:
                if abs(frag_delta - ground_delta) <= tolerance:
                    matched_signals += 1
                    break

        if matched_signals == total_fragment_signals and total_fragment_signals > 0:
            h_match = f"H NMR: MATCHED - All {total_fragment_signals} fragment signals found in ground truth"
        else:
            h_match = f"H NMR: NOT MATCHED - Only {matched_signals}/{total_fragment_signals} fragment signals found in ground truth"

    results.append(h_match)

    return " | ".join(results)


@tool(hidden_args=["h_smiles"])
def validate_fragment_matching(fragment: str, h_smiles: str) -> dict[str, list[str]]:
    """[BRIEF] Validate fragment matching with the NMR spectra of the sample at hand. [/BRIEF]

    [DETAILED] This tool compares the NMR spectra of a fragment with the ground truth spectra obtained from the full molecule. It checks for matching signals within a specified tolerance. If the fragment's signals are found within the ground truth signals, it is considered a valid match. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have a fragment and want to validate its NMR spectra against the full molecule's spectra.
    - When you need to ensure that the fragment's signals are present in the ground truth spectra.
    - When you want to confirm that the fragment is a valid representation of the full molecule based on NMR data. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. Obtain some candidates fragments. This can be done either by joining pieces from the different spectra or with the tool `return_possible_fragments`.
    2. Call this tool with the fragment SMILES to validate the fragment's NMR spectra.
    3. Analyze the results and determine if the fragment is a valid representation of the full molecule. [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The tool takes a fragment SMILES string as the input.
    - It simulates the NMR spectra for the fragment.
    - The spectra of the fragment is then compared to the spectra of the sample at hand.
    - If the fragment's signals are found within the sample signals, it is considered a valid match. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `validate_fragment_matching("C1=CC=CC=C1")`,
        `validate_fragment_matching("C1=CC=CC=C1O")`,
        `validate_fragment_matching("C1=CC=CC=C1N")`,
        `validate_fragment_matching("C1=CC=CC=C1C")`,
        `validate_fragment_matching("C1=CC=CC=C1F")`
    ]
    [/SYNTACTICAL]

    Args:
        fragment (str):
            [BRIEF] The SMILES representation of the fragment to validate. [/BRIEF]
            [DETAILED] The SMILES string representing the chemical structure of the fragment to validate. It should be a valid SMILES notation that can be processed by the validation function. [/DETAILED]
            [SYNTACTICAL] "valid SMILES string" [/SYNTACTICAL]
            [EXAMPLES] "C1=CC=CC=C1", "C1=CC=CC=C1O", "C1=CC=CC=C1N", "C1=CC=CC=C1C", "C1=CC=CC=C1F" [/EXAMPLES]

    Returns:
        dict[str, list[str]]:
            [BRIEF] A dictionary containing the validation results and spectra comparison. [/BRIEF]
            [DETAILED] The dictionary will include keys for "valid", "invalid", and "spectra_comparison", with corresponding values based on the validation process. [/DETAILED]
            [EXAMPLES] {'valid': 'Cc1ccccc1CCl', 'invalid': '', 'spectra_comparison': 'C13 NMR: NOT MATCHED - Only 4/8 fragment signals found in ground truth | H NMR: MATCHED - All 3 fragment signals found in ground truth'} [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] The provided SMILES string is invalid. [/ERROR_WHEN]
            [ERROR_DETAILS] The SMILES string could not be parsed into a valid molecular structure. [/ERROR_DETAILS]
            [ERROR_SOLUTION] Please provide a valid SMILES string that can be processed by the validation function. [/ERROR_SOLUTION]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function relies on the RDKit library for SMILES parsing.
        - The accuracy of the NMR spectra simulation may vary depending on the complexity of the fragment.
        - The simulation might fail or produce inaccurate results for highly complex or unusual molecular structures.
    [/LIMITATIONS]

    """
    if Chem.MolFromSmiles(h_smiles) is None:
        raise ValueError("Invalid SMILES string provided.")

    if Chem.MolFromSmiles(fragment) is None:
        return {"invalid": [fragment]}

    # Get spectra for both molecules
    ground_spectra = {
        "c13_nmr": get_carbon_spectrum.remote(h_smiles),
        "h_nmr": get_proton_spectrum.remote(h_smiles),
    }
    fragment_spectra = {
        "c13_nmr": get_carbon_spectrum.remote(fragment),
        "h_nmr": get_proton_spectrum.remote(fragment),
    }

    # Compare the spectra
    comparison_result = compare_nmr_spectra(ground_spectra, fragment_spectra)

    return {
        "valid": fragment if Chem.MolFromSmiles(fragment) is not None else "",
        "invalid": "" if Chem.MolFromSmiles(fragment) is not None else fragment,
        "spectra_comparison": comparison_result,
    }


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
        "validate_fragment_matching": validate_fragment_matching,
    }
