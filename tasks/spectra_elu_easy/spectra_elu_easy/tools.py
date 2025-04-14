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
