import gc
import uuid
from pathlib import Path
from typing import Any

import chromadb
import requests
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from corral.utils import (
    create_vector_database,
    make_api_request,
    remote_call,
    tool,
    vector_database_search,
    web_search,
)


@tool
def enhanced_brave_search(query: str, num_results: int = 5) -> list[dict]:
    """Perform a web search using Brave Search, then filter and rank results using embeddings.

    Args:
        query (str): The search query string
        num_results (int, optional): Maximum number of results to return. Defaults to 5

    Returns:
        list[dict]: A list of dictionaries containing the most relevant search results
        with their content and metadata, sorted by similarity score

    Raises:
        ValueError: If BRAVE_SEARCH_API_KEY environment variable is not set
    """
    return web_search(
        query=query,
        num_results=num_results,
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
def smiles_to_name(compound: str) -> str:
    """
    Returns the IUPAC name of a compound given its SMILES representation.

    Args:
        smiles (str): The SMILES representation of the compound.

    Returns:
        str: The IUPAC name of the compound.
    """
    return remote_call(function_name="get_iupac_name", env_name="chemenv")(
        smiles=compound
    )


@tool
def get_smiles_from_name(compound: str) -> str:
    """
    Returns the SMILES representation of a compound given its IUPAC name.

    Args:
        name (str): The IUPAC name of the compound.

    Returns:
        str: The SMILES representation of the compound.
    """
    return remote_call(function_name="get_smiles_from_name", env_name="chemenv")(
        name=compound
    )


@tool
def get_pka_from_smiles(smiles: str) -> str:
    """
    Returns the pKa value of a compound given its SMILES representation.

    Args:
        smiles (str): The SMILES representation of the compound.

    Returns:
        str: The pKa value of the compound.
    """
    return remote_call(function_name="pka_from_smiles", env_name="chemenv")(
        smiles=smiles
    )


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
def get_element_info(element: str) -> str:
    """
    Returns information about a chemical element given its symbol.
    The information will include the element's name, symbol, atomic number,
    atomic mass, electronic configuration, electronegativity, group, period,
    and block.

    Args:
        identifier (str): The symbol of the chemical element, e.g., "H" for Hydrogen.

    Returns:
        str: Information about the element.
    """
    return remote_call(function_name="get_element_info", env_name="chemenv")(
        identifier=element
    )


@tool
def get_number_of_isomers(compound: str) -> str:
    """
    Returns the number of isomers for a given compound based on the compounds with the same empirical formula in PubChem as `compound`.
    Note that this implies that this number is not exhaustive

    Args:
        compound (str): The compound to search for. It can be a SMILES string, a PubChem CID, or InChI notation.

    Returns:
        str: The number of isomers for the compound.
    """
    return remote_call(function_name="get_number_isomers_pubchem", env_name="chemenv")(
        compound=compound
    )


@tool
def get_ghs_classification_pubchem(compound: str) -> dict:
    """
    Returns the GHS classification of a compound from PubChem.

    Args:
        compound (str): The compound to search for. It can be a SMILES string, a PubChem CID, or InChI notation.

    Returns:
        dict: The GHS classification of the compound.
    """
    return remote_call(
        function_name="get_ghs_classification_pubchem", env_name="chemenv"
    )(compound=compound)


@tool
def get_ms_spectra_pubchem(compound: str) -> dict:
    """
    Returns the MS spectra of a compound from PubChem.

    Args:
        compound (str): The compound to search for. It can be a SMILES string, a PubChem CID, or InChI notation.

    Returns:
        dict: The MS spectra of the compound.
    """
    return remote_call(function_name="get_ms_spectra_pubchem", env_name="chemenv")(
        compound=compound
    )


@tool
def get_h_nmr_spectra_pubchem(compound: str) -> dict:
    """
    Returns the 1H-NMR spectra of a compound from PubChem.

    Args:
        compound (str): The compound to search for. It can be a SMILES string, a PubChem CID, or InChI notation.

    Returns:
        dict: The H-NMR spectra of the compound.
    """
    return remote_call(function_name="get_h_nmr_spectra_pubchem", env_name="chemenv")(
        compound=compound
    )


@tool
def get_c_nmr_spectra_pubchem(compound: str) -> str:
    """
    Returns the C-NMR spectra of a compound from PubChem.

    Args:
        compound (str): The compound to search for. It can be a SMILES string, a PubChem CID, or InChI notation.

    Returns:
        dict: The C-NMR spectra of the compound.
    """
    return remote_call(function_name="get_c_nmr_spectra_pubchem", env_name="chemenv")(
        compound=compound
    )


@tool
def simulate_spectra(smiles: str) -> dict[str, str]:
    """
    Simulate 1H NMR, 13C NMR, and IR spectra for a given molecule using its SMILES string.
    If some of the spectra are not available, the function will return None for those spectra.
    Use this to complement the PubChem data, or to provide an estimate of the spectra for a compound.

    Args:
        smiles (str): The SMILES representation of the compound.

    Returns:
        dict: The simulated spectra of the compound.
    """
    return remote_call(function_name="simulate_spectra", env_name="chemenv")(
        smiles=smiles
    )


@tool
def get_functional_groups(smiles: str) -> list[str]:
    """
    Returns the names of all the functional groups of a compound given its SMILES representation.

    Args:
        smiles (str): The SMILES representation of the compound.

    Returns:
        list[str]: The names of the functional groups of the compound.
    """
    return remote_call(function_name="get_functional_groups", env_name="chemenv")(
        smiles=smiles
    )


def fetch_all_studies(drug_name: str) -> list[dict[str, Any]]:
    """Fetch all studies related to a specific drug from ClinicalTrials.gov

    Args:
        drug_name (str): The name of the drug to search for

    Returns:
        list[dict]: A list of dictionaries containing study data
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.term": drug_name, "pageSize": 100, "format": "json"}
    all_studies = []
    next_page_token = None

    try:
        while True:
            if next_page_token:
                params["pageToken"] = next_page_token

            data = make_api_request(
                url=base_url, method="GET", params=params, verbose=True
            )

            studies = data.get("studies", [])
            all_studies.extend(studies)

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

        return all_studies
    except requests.exceptions.RequestException as e:
        logger.warning(f"Warning: Exception during fetching studies: {e!s}")
        return []


def parse_study_data(studies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse the study data to extract relevant information

    Args:
        studies (list[dict]): A list of dictionaries containing study data

    Returns:
        list[dict]: A list of dictionaries with parsed study data
    """
    parsed_data = []
    for study in studies:
        nct_id = (
            study.get("protocolSection", {})
            .get("identificationModule", {})
            .get("nctId", "N/A")
        )
        official_title = (
            study.get("protocolSection", {})
            .get("identificationModule", {})
            .get("officialTitle", "N/A")
        )
        status = (
            study.get("protocolSection", {})
            .get("statusModule", {})
            .get("overallStatus", "N/A")
        )
        conditions = (
            study.get("protocolSection", {})
            .get("conditionsModule", {})
            .get("conditions", [])
        )
        interventions = (
            study.get("protocolSection", {})
            .get("armsInterventionsModule", {})
            .get("interventions", [])
        )
        brief_summary = (
            study.get("protocolSection", {})
            .get("descriptionModule", {})
            .get("briefSummary", "N/A")
        )
        detailed_description = (
            study.get("protocolSection", {})
            .get("descriptionModule", {})
            .get("detailedDescription", "N/A")
        )
        primary_outcomes = (
            study.get("protocolSection", {})
            .get("outcomesModule", {})
            .get("primaryOutcomes", [])
        )
        secondary_outcomes = (
            study.get("protocolSection", {})
            .get("outcomesModule", {})
            .get("secondaryOutcomes", [])
        )

        parsed_data.append(
            {
                "NCT ID": nct_id,
                "Official Title": official_title,
                "Status": status,
                "Conditions": conditions,
                "Interventions": interventions,
                "Brief Summary": brief_summary,
                "Detailed Description": detailed_description,
                "Primary Outcomes": primary_outcomes,
                "Secondary Outcomes": secondary_outcomes,
            }
        )

    return parsed_data


def _search_clinical_trials(search_term: str, query: str, top_k: int = 5) -> list[dict]:
    """
    Helper function that fetches clinical trial data and performs semantic search.

    Args:
        search_term (str): Term to search for in ClinicalTrials.gov
        query (str): Text query for semantic search on the retrieved trials
        top_k (int): Number of top results to return (default: 5)

    Returns:
        list[dict]: List of dictionaries with relevant clinical trial data
    """
    studies = fetch_all_studies(search_term)
    parsed_studies = parse_study_data(studies)

    collection_name = f"clinical_trials_{uuid.uuid4().hex}"

    try:
        chunks = []
        metadata_map = {}

        for i, study in enumerate(parsed_studies):
            study_text = (
                f"Title: {study['Official Title']}\n"
                f"Summary: {study['Brief Summary']}\n"
                f"Status: {study['Status']}\n"
                f"Conditions: {', '.join(study['Conditions'])}\n"
                f"Description: {study['Detailed Description']}"
            )

            chunks.append(study_text)
            metadata_map[str(i)] = {"study_index": i}

        create_vector_database(
            chunks=chunks,
            collection_name=collection_name,
        )
        return vector_database_search(
            query=query, collection_name=collection_name, top_k=top_k
        )

    except Exception as e:
        return [{"error": f"Error searching clinical trials: {e!s}"}]

    finally:
        delete_vector_db(collection_name)


@tool
def search_clinical_trials_by_query(query: str, top_k: int = 5) -> list[dict]:
    """
    Fetches clinical trial data for a specific query from ClinicalTrials.gov
    returns the most relevant trials based on the query.

    Args:
        query (str): Text query to find relevant clinical trials
        top_k (int): Number of top results to return if query is provided. Defaults to 5

    Returns:
        list[dict]: List of dictionaries with parsed clinical trial data, optionally filtered by relevance
    """
    return _search_clinical_trials(search_term=query, query=query, top_k=top_k)


@tool
def search_clinical_trials_by_drug(
    drug_name: str, query: str, top_k: int = 5
) -> list[dict]:
    """
    Fetches ALL clinical trial data for a specific drug from ClinicalTrials.gov
    with semantic search capabilities from a query
    This function is costly so it should be used with caution and as a last resource.

    Args:
        drug_name: Name of the drug to search for in clinical trials
        query: Text query to find relevant trials
        top_k: Number of top results to return if query is provided (default: 5)

    Returns:
        list[dict]: List of dictionaries with parsed clinical trial data, optionally filtered by relevance
    """
    return _search_clinical_trials(search_term=drug_name, query=query, top_k=top_k)


@tool
def search_materials_compatibility(
    material: str, chemical: str, top_k: int = 10
) -> list[dict]:
    """
    Search for materials compatibility data in a dedicated dataset based on the provided material and
    chemical.

    Args:
        material (str): The material to search for.
        chemical (str): The chemical to search for.
        top_k (int): The number of top results to return.

    Returns:
        list[dict]: A list of dictionaries containing the most relevant materials compatibility data.
    """
    collection_name = "materials_compatibility"
    path = (
        Path(__file__).resolve().parents[3]
        / "vector_databases"
        / "materials_compatibility"
    )

    try:
        query = f"Compatibility of {material} with {chemical}"
        return vector_database_search(
            query=query, collection_name=collection_name, path=str(path), top_k=top_k
        )

    except Exception as e:
        return [{"error": f"Error searching materials compatibility: {e!s}"}]
