import gc
import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from corral.base import Tool
from corral.io import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from corral.utils import (
    create_vector_database,
    remote_call,
    tool,
    vector_database_search,
    web_search,
)

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/temp")


@tool
def enhanced_brave_search(
    query: str, num_results: int = 5, min_similarity: float = 0.75
) -> list[dict]:
    """Perform a web search using Brave Search, then filter and rank results using embeddings.

    Args:
        query (str): The search query string
        num_results (int, optional): Maximum number of results to return. Defaults to 5
        min_similarity (float, optional): Minimum similarity score threshold. Defaults to 0.75

    Returns:
        list[dict]: A list of dictionaries containing the most relevant search results
        with their content and metadata, sorted by similarity score

    Raises:
        ValueError: If BRAVE_SEARCH_API_KEY environment variable is not set
    """
    return web_search(
        query=query,
        num_results=num_results,
        min_similarity=min_similarity,
    )


def process_pubchem_json(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Process PubChem JSON data by extracting the smallest TOCHeading units.

    Args:
        data[dict]: JSON data in PubChem format

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - name: The section name (from TOCHeading)
            - root_path: Path to the section in the original JSON
            - description: The section description
            - original: The original unprocessed section data
    """
    result = []

    def extract_sections(section, path=""):
        if isinstance(section, dict):
            if "TOCHeading" in section:
                current_path = (
                    f"{path}.{section['TOCHeading']}" if path else section["TOCHeading"]
                )

                if "Section" in section:
                    for subsection in section["Section"]:
                        extract_sections(subsection, current_path)
                else:
                    result.append(
                        {
                            "name": section.get("TOCHeading", ""),
                            "root_path": current_path,
                            "description": section.get("Description", ""),
                            "original": section,
                        }
                    )
            else:
                for key, value in section.items():
                    new_path = f"{path}.{key}" if path else key
                    extract_sections(value, new_path)
        elif isinstance(section, list):
            for i, item in enumerate(section):
                new_path = f"{path}[{i}]"
                extract_sections(item, new_path)

    if "Record" in data:
        extract_sections(data["Record"], "Record")
    else:
        extract_sections(data)

    return result


def delete_vector_db(collection_name: str) -> None:
    """Delete the vector database collection

    Args:
        collection_name (str): The name of the collection to delete
    """
    try:
        persist_directory = Path(Path.cwd()) / "vector_db"
        client = chromadb.PersistentClient(path=str(persist_directory))
        if collection_name in [c.name for c in client.list_collections()]:
            client.delete_collection(name=collection_name)

        gc.collect()

        import subprocess

        collection_path = persist_directory / collection_name
        if collection_path.exists():
            subprocess.run(["rm", "-rf", str(collection_path)], check=True)
            logger.info(f"Removed collection directory: {collection_path}")
    except Exception as e:
        logger.warning(f"Warning: Exception during cleanup: {e!s}")


@tool
def relevant_pubchem_sections(
    compound: str, query: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """
    Retrieves the most relevant sections from a JSON file containing compound data
    based on a given compound and a text query.
    Useful for searching through PubChem records for a specific compound.

    Args:
        compound (str): The compound to search for. It can be a SMILES string, a PubChem CID, or InChI notation.
        query (str): Text query to search for in the full record of the compound.
        top_k (int, optional): Number of top results to return. Defaults to 5.

    Returns:
        list[dict]: A list of dictionaries containing the most relevant sections
    """
    collection_name = f"compounds_db_{uuid.uuid4().hex}"

    try:
        full_record = remote_call(
            function_name="get_pubchem_full_record", env_name="chemenv"
        )(compound=compound)

        chunks = process_pubchem_json(full_record)

        create_vector_database(chunks=chunks, collection_name=collection_name)
        return vector_database_search(
            query=query, collection_name=collection_name, top_k=top_k
        )

    except Exception as e:
        return [{"error": f"Error: {e!s}"}]

    finally:
        delete_vector_db(collection_name)


@tool
def get_formula_from_smiles(smiles: str) -> str:
    """
    Generate a chemical formula from a SMILES string using RDKit.

    Args:
        smiles (str): SMILES representation of a molecule

    Returns:
        str: The chemical formula in Hill notation (C, H, then alphabetical)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return "Invalid SMILES string"
        return rdMolDescriptors.CalcMolFormula(mol)

    except Exception as e:
        return f"Error: {e!s}"


@tool
def simulate_spectra(smiles: str) -> dict[str, str]:
    """
    Simulate 1H NMR, 13C NMR, and IR spectra for a given molecule using its SMILES string.
    If some of the spectra are not available, the function will return None for those spectra.
    Use this to complement the PubChem data.

    Args:
        smiles (str): The SMILES representation of the compound.

    Returns:
        dict: The simulated spectra of the compound.
    """
    return remote_call(function_name="simulate_spectra", env_name="chemenv")(
        smiles=smiles
    )


@tool
def search_by_smiles(smiles, top_k=10):
    """
    Search the NMRShift database for entries matching the given SMILES.

    Args:
        smiles (str): SMILES string to search for
        top_k (int, optional): Maximum number of results to return. Defaults to 10

    Returns:
        list: Matching entries
    """
    collection_name = "nmrshiftdb2"
    db_path = Path(__file__).resolve().parents[3] / "vector_databases" / "nmrshiftdb2"

    query = f"SMILES: {smiles}"

    return vector_database_search(
        query=query,
        collection_name=collection_name,
        path=db_path,
        top_k=top_k,
        metadata_only=True,
    )


@tool
def retrieve_protons_shifts() -> list[dict[str, str]]:
    """
    Retrieve the proton shifts ranges for hydrocarbons.

    Returns:
        list: A list of dictionaries containing the proton shifts ranges
    """

    return str(
        [
            {"Proton": "Aldehyde", "δ / ppm": "9.5 - 10.5"},
            {"Proton": "Aromatic", "δ / ppm": "6.5 - 8.2"},
            {"Proton": "Alkene", "δ / ppm": "4.5 - 6.1"},
            {"Proton": "Alkyne", "δ / ppm": "2.0 - 3.2"},
            {"Proton": "Acetal", "δ / ppm": "4.5 - 6.0"},
            {"Proton": "Alkoxy", "δ / ppm": "3.4 - 4.8"},
            {"Proton": "Methyl (CH₃-R)", "δ / ppm": "~0.9"},
            {"Proton": "N-methyl", "δ / ppm": "3.0 - 3.5"},
            {"Proton": "Methoxy", "δ / ppm": "3.3 - 3.8"},
            {
                "Proton": "CH₃ attached to double bonds/aromatics",
                "δ / ppm": "1.8 - 2.5",
            },
            {"Proton": "Methyl (CH₃-CO-)", "δ / ppm": "1.8 - 2.7"},
            {"Proton": "Methylene (CH₂-O-)", "δ / ppm": "~3.6 - 4.7"},
            {"Proton": "Methylene (CH₂-R₁R₂)", "δ / ppm": "~1.3"},
            {"Proton": "Methine (CH-R₁R₂R₃)", "δ / ppm": "~1.5"},
            {"Proton": "Cyclopropane", "δ / ppm": "0.22"},
            {"Proton": "Me₄Si (TMS)", "δ / ppm": "0.0"},
            {"Proton": "Metal hydride", "δ / ppm": "-5 to -20"},
        ]
    )


@tool
def retrieve_aromatic_protons_shifts() -> list[dict[str, str]]:
    """
    Retrieve the proton shifts ranges for aromatic hydrocarbons.

    Returns:
        list: A list of dictionaries containing the proton shifts ranges
    """
    return str(
        [
            {"Substituent": "NO₂", "Ortho": 0.95, "Meta": 0.17, "Para": 0.33},
            {"Substituent": "CHO", "Ortho": 0.58, "Meta": 0.21, "Para": 0.27},
            {"Substituent": "COCl", "Ortho": 0.83, "Meta": 0.16, "Para": 0.30},
            {"Substituent": "COOH", "Ortho": 0.8, "Meta": 0.14, "Para": 0.2},
            {"Substituent": "COOCH₃", "Ortho": 0.74, "Meta": 0.07, "Para": 0.2},
            {"Substituent": "COCH₃", "Ortho": 0.64, "Meta": 0.09, "Para": 0.3},
            {"Substituent": "CN", "Ortho": 0.27, "Meta": 0.11, "Para": 0.3},
            {"Substituent": "C₆H₅", "Ortho": 0.18, "Meta": 0, "Para": 0.08},
            {"Substituent": "CCl₃", "Ortho": 0.8, "Meta": 0.2, "Para": 0.2},
            {"Substituent": "CHCl₂", "Ortho": 0.1, "Meta": 0.06, "Para": 0.1},
            {"Substituent": "CH₂Cl", "Ortho": 0, "Meta": 0.01, "Para": 0},
            {"Substituent": "CH₃", "Ortho": -0.17, "Meta": -0.09, "Para": -0.18},
            {"Substituent": "CH₂CH₃", "Ortho": -0.15, "Meta": -0.06, "Para": -0.18},
            {"Substituent": "CH(CH₃)₂", "Ortho": -0.14, "Meta": -0.09, "Para": -0.18},
            {"Substituent": "C(CH₃)₃", "Ortho": 0.01, "Meta": -0.1, "Para": -0.24},
            {"Substituent": "CH₂OH", "Ortho": -0.1, "Meta": -0.1, "Para": -0.1},
            {"Substituent": "CH₂NH₂", "Ortho": 0, "Meta": 0, "Para": 0.22},
            {"Substituent": "F", "Ortho": -0.3, "Meta": -0.02, "Para": -0.22},
            {"Substituent": "Cl", "Ortho": 0.02, "Meta": 0.06, "Para": -0.04},
            {"Substituent": "Br", "Ortho": 0.22, "Meta": -0.13, "Para": -0.03},
            {"Substituent": "I", "Ortho": 0.4, "Meta": -0.26, "Para": -0.03},
            {"Substituent": "OCH₃", "Ortho": -0.43, "Meta": -0.09, "Para": -0.37},
            {"Substituent": "OCOCH₃", "Ortho": -0.21, "Meta": -0.02, "Para": -0.4},
            {"Substituent": "OH", "Ortho": -0.5, "Meta": -0.14, "Para": -0.4},
            {
                "Substituent": "p-CH₃C₆H₄SO₃",
                "Ortho": -0.26,
                "Meta": -0.05,
                "Para": -0.25,
            },
            {"Substituent": "NH₂", "Ortho": -0.75, "Meta": -0.24, "Para": -0.63},
            {"Substituent": "SCH₃", "Ortho": -0.03, "Meta": 0, "Para": -0.3},
            {"Substituent": "N(CH₃)₂", "Ortho": -0.6, "Meta": -0.1, "Para": -0.62},
        ]
    )


@tool
def retrieve_carbon_shifts() -> list[dict[str, str]]:
    """
    Retrieve the carbon shifts ranges for hydrocarbons.

    Returns:
        list: A list of dictionaries containing the carbon shifts ranges
    """
    return str(
        [
            {"Group": "CH₃-", "Shift (ppm)": "10-30 ppm"},
            {"Group": "R₃C-, R₂CH, RCH₂", "Shift (ppm)": "25-50 ppm"},
            {"Group": "=CH₂", "Shift (ppm)": "105-120 ppm"},
            {"Group": "=CH", "Shift (ppm)": "110-140 ppm"},
            {"Group": "=CR₂", "Shift (ppm)": "130-150 ppm"},
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
            {"Group": "R₃C-O", "Shift (ppm)": "70-85 ppm"},
            {"Group": "R₂HC-O", "Shift (ppm)": "60-80 ppm"},
            {"Group": "RH₂C-O", "Shift (ppm)": "45-65 ppm"},
            {"Group": "H₃C-O", "Shift (ppm)": "50-60 ppm"},
            {"Group": "Epoxides", "Shift (ppm)": "40-60 ppm"},
        ]
    )


def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)
    fs_manager = FSManager("file", base_path=BASE_WORK_DIR)
    return {
        "list_files": ListFilesTool(fs_manager),
        "read_file": ReadFileTool(fs_manager),
        "write_file": WriteFileTool(fs_manager),
        "file_info": FileInfoTool(fs_manager),
        "cat_files": CatFilesTool(fs_manager),
        "copy_file": CopyFileTool(fs_manager),
        "enhanced_brave_search": enhanced_brave_search,
        "relevant_pubchem_sections": relevant_pubchem_sections,
        "get_formula_from_smiles": get_formula_from_smiles,
        "simulate_spectra": simulate_spectra,
    }
