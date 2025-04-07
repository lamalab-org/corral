from __future__ import annotations

import gc
import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
import modal
import numpy as np
import requests
from langchain_community.tools.brave_search.tool import BraveSearch
from loguru import logger
from rdkit import Chem
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from corral.utils import (
    create_vector_database,
    embed_text,
    remote_call,
    tool,
    vector_database_search,
)


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
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError(
            "BRAVE_SEARCH_API_KEY environment variable is required but not set"
        )

    try:
        # TODO: Avoid using LangChain
        brave_search_tool = BraveSearch.from_api_key(api_key=api_key)
        initial_results = brave_search_tool._run(query)

        if not initial_results:
            return []

        query_embedding = np.array(embed_text(chunks=[query])[0]).reshape(1, -1)
        result_texts = [f"{r['title']}: {r['snippet']}" for r in initial_results]
        result_embeddings = np.array(embed_text(chunks=result_texts))

        similarity_scores = sklearn_cosine_similarity(
            query_embedding, result_embeddings
        ).flatten()

        results_with_scores = []
        for i, result in enumerate(initial_results):
            similarity = float(similarity_scores[i])

            results_with_scores.append(
                {
                    "title": result["title"],
                    "snippet": result["snippet"],
                    "link": result["link"],
                    "similarity_score": similarity,
                }
            )

        filtered_results = [
            r for r in results_with_scores if r["similarity_score"] >= min_similarity
        ]
        sorted_results = sorted(
            filtered_results, key=lambda x: x["similarity_score"], reverse=True
        )

        return sorted_results[:num_results]

    except Exception:
        return []


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
        # Force Python garbage collection
        gc.collect()
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
        remote_pubchem_record = modal.Function.from_name(
            "chemenv", "get_pubchem_full_record"
        )
        full_record = remote_pubchem_record.remote(compound)

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
        compound (str): The SMILES representation of the compound.

    Returns:
        str: The IUPAC name of the compound.
    """
    return remote_call(function_name="get_iupac_name", env_name="chemenv")(
        compound=compound
    )


@tool
def get_smiles_from_name(compound: str) -> str:
    """
    Returns the SMILES representation of a compound given its IUPAC name.

    Args:
        compound (str): The IUPAC name of the compound.

    Returns:
        str: The SMILES representation of the compound.
    """
    return remote_call(function_name="get_smiles_from_name", env_name="chemenv")(
        compound=compound
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

        return Chem.rdMolDescriptors.CalcMolFormula(mol)

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
        element (str): The symbol of the chemical element, e.g., "H" for Hydrogen.

    Returns:
        str: Information about the element.
    """
    return remote_call(function_name="get_element_info", env_name="chemenv")(
        element=element
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
def get_compound_isomers(compound: str) -> list:
    """
    Returns the isomers of a given compound based on the compounds with the same empirical formula in PubChem as `compound`.
    Note that this implies that this number is not exhaustive

    Args:
        compound (str): The compound to search for. It can be a SMILES string, a PubChem CID, or InChI notation.

    Returns:
        list: The isomers of the compound.
    """
    return remote_call(
        function_name="get_compound_isomers_pubchem", env_name="chemenv"
    )(compound=compound)


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def fetch_study_page(base_url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fetch a single page of study data with retry logic

    Args:
        base_url (str): The base URL for the API endpoint
        params (dict): The parameters to include in the API request

    Returns:
        dict: The JSON response from the API
    """
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()


def fetch_all_studies(drug_name: str) -> list[dict[str, Any]]:
    """Fetch all studies related to a specific drug from ClinicalTrials.gov

    Args:
        drug_name (str): The name of the drug to search for

    Returns:
        list[dict]: A list of dictionaries containing study data
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.interventionName": drug_name, "pageSize": 100, "format": "json"}
    all_studies = []
    next_page_token = None

    try:
        while True:
            if next_page_token:
                params["pageToken"] = next_page_token

            data = fetch_study_page(base_url, params)
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


@tool
def search_clinical_trials(drug_name: str, query: str, top_k: int = 5) -> list[dict]:
    """
    Fetches clinical trial data for a specific drug from ClinicalTrials.gov
    with semantic search capabilities from a query

    Args:
        drug_name: Name of the drug to search for in clinical trials
        query: Text query to find relevant trials
        top_k: Number of top results to return if query is provided (default: 5)

    Returns:
        List of dictionaries with parsed clinical trial data, optionally filtered by relevance
    """
    studies = fetch_all_studies(drug_name)
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
