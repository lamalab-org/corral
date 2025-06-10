import gc
import traceback
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
    r"""[BRIEF] Perform a web search using Brave Search, then filter and rank results using embeddings. [\BRIEF]

    [DETAILED] This tool performs a web search using Brave Search API,
    retrieves the top results for a query, and then filters and ranks them based on their
    relevance to the search query using embeddings. It returns a list of the most
    relevant search results, each containing the content and metadata of the
    search result, sorted by similarity score. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when the other specific seach tools are not available or when you need to perform a general web search.
    - When you need to find information that is not available in the local vector database or other specialized databases.
    - When you need to retrieve some information that is not available with the other tools.
    - Recommended for general query seaches that do not require specific databases or structured data. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the other more specific tools are not suitable for the query. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with a descriptive query string to perform a web search. [\CURRENT]
    3. [FOLLOW_UP] Use the information from the web search combined with retrieved from other tools
    to solve the task validating the tasks with the available tools. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Checks if a BRAVE_SEARCH_API_KEY environment variable is set.
    - If not set, raises a ValueError.
    - Uses the Brave Search API to perform a web search with the provided query.
    - Retrieves the top `num_results` results.
    - Filters and ranks the results based on their relevance to the search query using embeddings.
    - Returns a list of dictionaries containing the most relevant search results,
    each with its content and metadata, sorted by similarity score that is also included. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `online_search("What is the chemical formula of caffeine?", num_results=10)`,
        `online_search("What is the chemical formula of aspirin?", num_results=7)`
        `online_search("What is the boiling point of water?", num_results=5)`,
        `online_search("What is the molecular weight of glucose?", num_results=10)`,
        `online_search("What is the structure of benzene?", num_results=5)`,
    ]
    [\SYNTACTICAL]

    Args:
        query (str):
                        [BRIEF] The search query string [\BRIEF]
                        [DETAILED] The query string to search for in the Brave Search API. It should be a descriptive string that represents the information you are looking for. [\DETAILED]
                        [SYNTACTICAL] Format: "string with no special requirements" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What is the chemical formula of caffeine?", "What is the boiling point of water?", "What is the structure of benzene?" [\EXAMPLES]

        num_results (int, optional):
                        [BRIEF] Maximum number of results to return. Defaults to 5 [\BRIEF]
                        [DETAILED] The maximum number of search results to return from the Brave Search API. It should be a positive integer. [\DETAILED]
                        [SYNTACTICAL] Format: "any positive integer (e.g., 5, 10, 20)" [\SYNTACTICAL]
                        [EXAMPLES] Examples: 5, 10, 20 [\EXAMPLES]

    Returns:
        list[dict]:
                        [BRIEF] A list of dictionaries containing the most relevant search results with their content and metadata, sorted by similarity score [\BRIEF]
                        [DETAILED] Each dictionary contains the content of the search result, its metadata, and a similarity score indicating how relevant the result is to the search query. The results are sorted by similarity score in descending order. [\DETAILED]
                        [EXAMPLES] Examples: [{"content": "Result 1 content", "metadata": {"source": "some_source.com"}, "similarity_score": 0.95}, {"content": "Result 2 content", "metadata": {"source": "second_source.org"}, "similarity_score": 0.90}, ...] [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If the BRAVE_SEARCH_API_KEY environment variable is not set or if an error occurs during the search. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when the BRAVE_SEARCH_API_KEY is not set, or if there is an error in making the API request to Brave Search, such as network issues or invalid query parameters. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires a valid BRAVE_SEARCH_API_KEY environment variable to be set.
        - The number of results returned is limited by the `num_results` parameter, which defaults to 5.
        - The search results are filtered and ranked based on their relevance to the search query using embeddings, which may not always yield the most relevant results.
        - The results content might not be completely accurate.
        - The results content might not be complete.
        - If some error occurs during the search, it will return an empty list.
    [/LIMITATIONS]
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
    r"""[BRIEF] Retrieve relevant sections from PubChem JSON data based on a compound and a text query. [\BRIEF]

    [DETAILED] This function retrieves the most relevant sections from a JSON file containing compound data
    based on a given compound and a text query. It processes the PubChem JSON data to extract sections,
    creates a vector database from the sections, and performs a search based on the query.
    It returns a list of the most relevant sections, each containing the section name, root path, description,
    and original data. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - When other more specific tools do not work for the task at hand.
    - Use it when you need to search through PubChem records for a specific compound.
    - When you want to retrieve relevant sections from PubChem data based on a text query.
    - Recommended for tasks that require detailed information about a compound from PubChem, such as chemical properties, structures, or biological activities. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the specific other tools such as `get_ghs_classification_pubchem` do not work for the task at hand. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the compound and a descriptive query string to retrieve relevant sections from PubChem. [\CURRENT]
    3. [FOLLOW_UP] Use the information from the retrieved sections to answer the task. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Retrieves the full record of a compound from PubChem using the `get_pubchem_full_record` remote function.
    - Processes the JSON data to extract sections based on the `TOCHeading` field.
    - Creates a vector database from the extracted sections, with metadata including section names and paths.
    - Searches the vector database for sections that are most relevant to the provided text query.
    - Returns a list of the most relevant sections, each containing the section name, root path, description, and original data. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `relevant_pubchem_sections("2244", "IR spectra")`,
        `relevant_pubchem_sections("C1=CC=CC=C1", "Chemical structure")`,
        `relevant_pubchem_sections("C1=CC=CC=C1", "Chemical properties")`
        `relevant_pubchem_sections("C1=CC=CC=C1", "Chemical reactions")`
        `relevant_pubchem_sections("C1=CC=CC=C1", "Mass spectrometry properties")`,
    ]
    [\SYNTACTICAL]

    Args:
        compound (str):
                        [BRIEF] The compound to search for in PubChem [\BRIEF]
                        [DETAILED] The compound to search for in PubChem. It can be a SMILES string, a PubChem CID, or InChI notation. [\DETAILED]
                        [SYNTACTICAL] Format: "SMILES string, PubChem CID, or InChI notation" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "2244" [\EXAMPLES]

        query (str):
                        [BRIEF] The text query to search for in the full record of the compound [\BRIEF]
                        [DETAILED] The text query to search for in the full record of the compound. It should be a descriptive string that represents the information you are looking for in the PubChem record. [\DETAILED]
                        [SYNTACTICAL] Format: "string with no special requirements" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "IR spectra", "Chemical structure", "Chemical properties", "Mass spectrometry properties" [\EXAMPLES]

        top_k (int, optional):
                        [BRIEF] The number of top results to return. Defaults to 5. [\BRIEF]
                        [DETAILED] The number of top results to return from the search. It should be a positive integer. [\DETAILED]
                        [SYNTACTICAL] Format: "any positive integer (e.g., 5, 10, 20)" [\SYNTACTICAL]
                        [EXAMPLES] Examples: 5, 10, 20 [\EXAMPLES]

    Returns:
        list[dict[str, Any]]:
                        [BRIEF] A list of dictionaries containing the most relevant sections from the PubChem record. [\BRIEF]
                        [DETAILED] Each dictionary contains the section name, root path, description, and original data of the section. The sections are sorted by relevance to the provided text query. [\DETAILED]
                        [EXAMPLES] Examples: [{"name": "Mass Spectrometry", "root_path": "Record.Section[0].TOCHeading", "description": "Mass spectrometry properties of the compound", "original": {...}}, {"name": "NMR Spectra", "root_path": "Record.Section[1].TOCHeading", "description": "NMR spectra of the compound", "original": {...}}, ...] [\EXAMPLES]

    [RAISES] Exceptions:
        Exception:
                        [ERROR_WHEN] If an error occurs during the retrieval or processing of the PubChem record. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in retrieving the full record of the compound from PubChem, processing the JSON data, or creating the vector database. It can occur due to network issues, invalid compound identifiers, or other unexpected errors. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] If it has to do with the input, try to solve the error trying a valid compound. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The retrieval of the full record from PubChem may fail if the compound identifier is invalid or not found.
        - The processing of the JSON data may not extract all relevant sections if the structure of the PubChem record changes.
        - The vector database creation and search may fail if there are issues with the ChromaDB client or the underlying storage.
        - The retrieval might not retrieve all meaningful sections if the PubChem record does not contain the expected structure.
        - The search results are limited to the `top_k` parameter, which defaults to 5.
    [/LIMITATIONS]
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
        raise [{"error": f"Error: {e!s}"}] from e

    finally:
        delete_vector_db(collection_name)


@tool
def smiles_to_name(compound: str) -> str:
    r"""[BRIEF] Convert a SMILES representation of a compound to its IUPAC name. [\BRIEF]

    [DETAILED] This function takes a SMILES representation of a compound and returns its IUPAC name.
    It uses a remote function call to the `get_iupac_name` function in the `chemenv` environment.
    The remote function searches in different datasets such as PubChem, OPSIN and Cactus to find the IUPAC name of the compound. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you have a SMILES representation of a compound and need to find its IUPAC name.
    - When you need to convert a SMILES string to a human-readable name for the compound.
    - Recommended for tasks that require the IUPAC name of a compound from its SMILES representation. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have a valid SMILES representation of the compound. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the SMILES string to convert it to its IUPAC name. [\CURRENT]
    3. [FOLLOW_UP] Use the IUPAC name in subsequent tasks, for further analysis or to solve the task at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a SMILES representation of a compound as input.
    - Calls the remote function `get_iupac_name` in the `chemenv` environment with the SMILES string.
    - The remote function searches in various chemical databases (like PubChem, OPSIN, and Cactus) to find the IUPAC name of the compound.
    - Returns the IUPAC name of the compound as a string. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `smiles_to_name("CCO")`,
        `smiles_to_name("C1=CC=CC=C1")`,
        `smiles_to_name("C(C(=O)O)N")`,
        `smiles_to_name("C1=CC=CC=C1O")`,
        `smiles_to_name("C1=CC=CC=C1C(=O)O")`,
    ]
    [\SYNTACTICAL]

    Args:
        smiles (str):
                        [BRIEF] The SMILES representation of the compound [\BRIEF]
                        [DETAILED] The SMILES representation of the compound to convert to its IUPAC name. It should be a valid SMILES string that represents the chemical structure of the compound. [\DETAILED]
                        [SYNTACTICAL] Format: "valid SMILES string" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] The IUPAC name of the compound [\BRIEF]
                        [DETAILED] The IUPAC name of the compound derived from its SMILES representation. It is a human-readable name that follows the IUPAC nomenclature rules for chemical compounds. [\DETAILED]
                        [EXAMPLES] Examples: "Ethanol", "Benzene", "Acetic acid" [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the remote function call or if the SMILES string is invalid [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `get_iupac_name`, such as network issues, invalid SMILES string, or if the compound cannot be found in the chemical databases. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid smiles. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The SMILES string must be valid and represent a chemical structure that can be interpreted by the remote function.
        - The remote function may not find the IUPAC name for all compounds, especially if they are not well-documented in the chemical databases.
        - The function relies on the availability of the remote service and its databases, which may change over time.
    [/LIMITATIONS]
    """
    try:
        return remote_call(function_name="get_iupac_name", env_name="chemenv")(
            smiles=compound
        )
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


@tool
def get_smiles_from_name(compound: str) -> str:
    r"""[BRIEF] Convert an IUPAC name of a compound to its SMILES representation. [\BRIEF]

    [DETAILED] This function takes an IUPAC name of a compound and returns its SMILES representation.
    It uses a remote function call to the `get_smiles_from_name` function in the `chemenv` environment.
    The remote function searches in the datasets PubChem and Cactus to find the SMILES representation of the compound. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you have an IUPAC name of a compound and need to find its SMILES representation.
    - When you need to convert a human-readable name to a SMILES string for computational purposes.
    - Recommended for tasks that require the SMILES representation of a compound from its IUPAC name. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have a valid IUPAC name of the compound. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the IUPAC name to convert it to its SMILES representation. [\CURRENT]
    3. [FOLLOW_UP] Use the SMILES representation in subsequent tasks, for further analysis or to answer the question of the task. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes an IUPAC name of a compound as input.
    - Calls the remote function `get_smiles_from_name` in the `chemenv` environment with the IUPAC name.
    - The remote function searches in the chemical databases of PubChem and Cactus to find the SMILES representation of the compound.
    - Returns the SMILES representation of the compound as a string. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_smiles_from_name("Ethanol")`,
        `get_smiles_from_name("Benzene")`,
        `get_smiles_from_name("Acetic acid")`,
        `get_smiles_from_name("Cyclohexane")`,
        `get_smiles_from_name("2-Propanol")`,
    ]
    [\SYNTACTICAL]

    Args:
        name (str):
                        [BRIEF] The IUPAC name of the compound [\BRIEF]
                        [DETAILED] The IUPAC name of the compound to convert to its SMILES representation. It should be a valid IUPAC name that represents the chemical structure of the compound. [\DETAILED]
                        [SYNTACTICAL] Format: "valid IUPAC name" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "Ethanol", "Benzene", "Acetic acid" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] The SMILES representation of the compound [\BRIEF]
                        [DETAILED] The SMILES representation of the compound derived from its IUPAC name. It is a string that encodes the chemical structure of the compound in a format that can be used for computational purposes. [\DETAILED]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N" [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the remote function call or if the IUPAC name is invalid [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `get_smiles_from_name`, such as network issues, invalid IUPAC name, or if the SMILES representation cannot be found in the chemical databases. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid IUPAC name. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The IUPAC name must be valid and represent a chemical structure that can be interpreted by the remote function.
        - The remote function may not find the SMILES representation for all compounds, especially if they are not well-documented in the chemical databases.
        - The function relies on the availability of the remote service and its databases, which may change over time.
    [/LIMITATIONS]
    """
    try:
        return remote_call(function_name="get_smiles_from_name", env_name="chemenv")(
            name=compound
        )
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


@tool
def get_pka_from_smiles(smiles: str) -> str:
    r"""[BRIEF] Get the pKa value of a compound from its SMILES representation. [\BRIEF]

    [DETAILED] This function takes a SMILES representation of a compound and returns its pKa value.
    It uses the RDKit package to calculate the pKa value based on the chemical structure represented by the SMILES string. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you have a SMILES representation of a compound and need to find its pKa value.
    - When you need to calculate the acidity or basicity of a compound based on its chemical structure.
    - Recommended for tasks that require the pKa value of a compound from its SMILES representation. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have a valid SMILES representation of the compound. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the SMILES string to calculate its pKa value. [\CURRENT]
    3. [FOLLOW_UP] Use the pKa value in subsequent tasks or for further analysis. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a SMILES representation of a compound as input.
    - Calls the remote function `pka_from_smiles` in the `chemenv` environment with the SMILES string.
    - The remote function calculates the pKa value of the compound based on its chemical structure using the RDKit functionality.
    - Returns the pKa value of the compound as a string. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_pka_from_smiles("CCO")`,
        `get_pka_from_smiles("C1=CC=CC=C1")`,
        `get_pka_from_smiles("C(C(=O)O)N")`,
        `get_pka_from_smiles("C1=CC=CC=C1O")`,
        `get_pka_from_smiles("C1=CC=CC=C1C(=O)O")`,
    ]
    [\SYNTACTICAL]

    Args:
        smiles (str):
                        [BRIEF] The SMILES representation of the compound [\BRIEF]
                        [DETAILED] The SMILES representation of the compound to calculate its pKa value. It should be a valid SMILES string that represents the chemical structure of the compound. [\DETAILED]
                        [SYNTACTICAL] Format: "valid SMILES string" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] The pKa value of the compound [\BRIEF]
                        [DETAILED] The pKa value of the compound derived from its SMILES representation. It is a numerical value that indicates the acidity or basicity of the compound in solution. [\DETAILED]
                        [EXAMPLES] Examples: "4.76", "9.89", "2.15" [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the remote function call or if the SMILES string is invalid [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `pka_from_smiles`, such as network issues, invalid SMILES string, or if the pKa value cannot be calculated for the compound. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid smiles. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The SMILES string must be valid and represent a chemical structure that can be interpreted by the remote function.
        - The function relies on the availability of the remote service, which may change over time.
    [/LIMITATIONS]
    """
    try:
        return remote_call(function_name="pka_from_smiles", env_name="chemenv")(
            smiles=smiles
        )
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


@tool
def get_formula_from_smiles(smiles: str) -> str:
    r"""[BRIEF] Generate a chemical formula from a SMILES string. [\BRIEF]

    [DETAILED] This function takes a SMILES representation of a molecule and returns its chemical formula in Hill notation (C, H, then alphabetical).
    It uses RDKit to parse the SMILES string and calculate the molecular formula. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you have a SMILES representation of a molecule and need to generate its chemical formula.
    - When you need to convert a SMILES string to a chemical formula for further analysis or reporting.
    - Recommended for tasks that require the chemical formula of a molecule from its SMILES representation. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have a valid SMILES representation of the molecule. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the SMILES string to generate its chemical formula. [\CURRENT]
    3. [FOLLOW_UP] Use the chemical formula in subsequent tasks or for further analysis, such as reporting the molecular composition of the molecule. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a SMILES representation of a molecule as input.
    - Uses RDKit to parse the SMILES string and create a molecular object.
    - Calculates the molecular formula in Hill notation (C, H, then alphabetical) using RDKit's `CalcMolFormula` function.
    - Returns the chemical formula as a string. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_formula_from_smiles("CCO")`,
        `get_formula_from_smiles("C1=CC=CC=C1")`,
        `get_formula_from_smiles("C(C(=O)O)N")`,
        `get_formula_from_smiles("C1=CC=CC=C1O")`,
        `get_formula_from_smiles("C1=CC=CC=C1C(=O)O")`,
    ]
    [\SYNTACTICAL]

    Args:
        smiles (str):
                        [BRIEF] The SMILES representation of the molecule [\BRIEF]
                        [DETAILED] The SMILES representation of the molecule to generate its chemical formula. It should be a valid SMILES string that represents the chemical structure of the molecule. [\DETAILED]
                        [SYNTACTICAL] Format: "valid SMILES string" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] The chemical formula of the molecule in Hill notation [\BRIEF]
                        [DETAILED] The chemical formula of the molecule derived from its SMILES representation. It is a string that represents the molecular composition of the molecule in Hill notation (C, H, then alphabetical). [\DETAILED]
                        [EXAMPLES] Examples: "C2H6O", "C6H6", "C2H5NO2" [\EXAMPLES]

    [RAISES] Exceptions:
        Exception:
                        [ERROR_WHEN] If an error occurs during the RDKit processing [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there are issues with the RDKit library. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool. [\ERROR_RECOVERY]

        ValueError:
                        [ERROR_WHEN] If the SMILES string is invalid or cannot be parsed [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in parsing the SMILES string or calculating the molecular formula, such as network issues, invalid SMILES string, or if the compound cannot be interpreted by RDKit. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Input a valid SMILES string. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The SMILES string must be valid and represent a chemical structure that can be interpreted by RDKit.
        - The function relies on the availability of the RDKit library, which must be properly installed in the environment.
        - The generated formula is in Hill notation, which may not be suitable for all applications or reporting standards.
    [/LIMITATIONS]
    """
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return rdMolDescriptors.CalcMolFormula(mol)

    except Exception as e:
        return f"Error: {e!s}"


@tool
def get_element_info(element: str) -> str:
    r"""[BRIEF] Get information about a chemical element by its symbol. [\BRIEF]

    [DETAILED] This function retrieves detailed information about a chemical element
    given its symbol (e.g., "H" for Hydrogen). It uses a remote function call to the `get_element_info` function in the `chemenv` environment.
    The remote function uses Mendeleev's periodic table data to retrieve detailed information about the element.
    The information includes the element's name, symbol, atomic number, atomic mass,
    electronic configuration, electronegativity, group, period, and block. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need detailed information about a chemical element based on its symbol.
    - When you want to retrieve properties of an element such as atomic number, mass, and electronic configuration.
    - Recommended for tasks that require information about chemical elements. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have the symbol of the chemical element you want to query. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the element symbol to retrieve its detailed information. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved information in subsequent tasks or for further analysis, such as understanding the properties of the element or answering the taks at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes the symbol of a chemical element as input (e.g., "H" for Hydrogen).
    - Calls the remote function `get_element_info` in the `chemenv` environment with the element symbol.
    - The remote function uses Mendeleev's periodic table data to retrieve detailed information about the element.
    - Returns a string containing the element's name, symbol, atomic number, atomic mass,
    electronic configuration, electronegativity, group, period, and block. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_element_info("H")`,
        `get_element_info("O")`,
        `get_element_info("C")`,
        `get_element_info("N")`,
        `get_element_info("Na")`,
    ]
    [\SYNTACTICAL]

    Args:
        element (str):
                        [BRIEF] The symbol of the chemical element [\BRIEF]
                        [DETAILED] The symbol of the chemical element for which you want to retrieve information. It should be a valid chemical symbol, such as "H" for Hydrogen, "O" for Oxygen, etc. [\DETAILED]
                        [SYNTACTICAL] Format: "valid chemical symbol" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "H", "O", "C", "N", "Na" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Information about the chemical element [\BRIEF]
                        [DETAILED] A string containing detailed information about the chemical element, including its name, symbol, atomic number, atomic mass, electronic configuration, electronegativity, group, period, and block. [\DETAILED]
                        [EXAMPLES] Examples: "Hydrogen (H), Atomic Number: 1, Atomic Mass: 1.008, Electronic Configuration: 1s1, Electronegativity: 2.20, Group: 1, Period: 1, Block: s" [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the remote function call or if the element symbol is invalid [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `get_element_info` regarding invalid element symbol, or if the element cannot be found in the chemical database. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Input a valid element symbol. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The element symbol must be valid and represent a chemical element that can be interpreted by the remote function.
        - The remote function may not find information for all elements, especially if they are not well-documented in the chemical databases.
        - The function relies on the availability of the remote service, which may change over time.
    [/LIMITATIONS]
    """
    return remote_call(function_name="get_element_info", env_name="chemenv")(
        identifier=element
    )


@tool
def get_number_of_isomers(compound: str) -> str:
    r"""[BRIEF] Get the number of isomers for a given compound based on its empirical formula in PubChem. [\BRIEF]

    [DETAILED] This function retrieves the number of isomers for a given compound by searching for compounds with the same empirical formula in PubChem.
    It uses a remote function call to the `get_number_isomers_pubchem` function in the `chemenv` environment.
    The number of isomers is not exhaustive and may not include all possible isomers for the compound. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to find the number of isomers for a compound based on its empirical formula.
    - When you want to understand the structural diversity of a compound by knowing how many isomers exist.
    - Recommended for tasks that require information about the isomeric forms of a compound. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have the identifier of the compound you want to query (e.g., SMILES string, PubChem CID, or InChI notation). [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the compound identifier to retrieve the number of isomers. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved number of isomers in subsequent tasks or for further analysis, such as understanding the chemical behavior or properties of the compound, or answering the task at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a compound identifier (e.g., SMILES string, PubChem CID, or InChI notation) as input.
    - Calls the remote function `get_number_isomers_pubchem` in the `chemenv` environment with the compound identifier.
    - The remote function searches PubChem for compounds with the same empirical formula as the given compound.
    - Returns the number of isomers for the compound as a string. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_number_of_isomers("CCO")`,
        `get_number_of_isomers("C1=CC=CC=C1")`,
        `get_number_of_isomers("C(C(=O)O)N")`,
        `get_number_of_isomers("C1=CC=CC=C1O")`,
        `get_number_of_isomers("C1=CC=CC=C1C(=O)O")`,
    ]
    [\SYNTACTICAL]

    Args:
        compound (str):
                        [BRIEF] The compound to search for in PubChem [\BRIEF]
                        [DETAILED] The compound to search for in PubChem. It can be a SMILES string, a PubChem CID, or InChI notation. It should represent the chemical structure of the compound for which you want to find the number of isomers. [\DETAILED]
                        [SYNTACTICAL] Format: "SMILES string, PubChem CID, or InChI notation" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "2244" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] The number of isomers for the compound [\BRIEF]
                        [DETAILED] The number of isomers for the compound based on the compounds with the same empirical formula in PubChem. It is a string representation of the count of isomers found. Note that this number is not exhaustive and may not include all possible isomers. [\DETAILED]
                        [EXAMPLES] Examples: "3", "5", "10" [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the remote function call or if the compound identifier is invalid [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `get_number_isomers_pubchem`, such as network issues, invalid compound identifier, or if the number of isomers cannot be determined for the compound. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid compound. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The compound identifier must be valid and represent a chemical structure that can be interpreted by the remote function.
        - The remote function may not find all isomers for the compound, especially if they are not well-documented in PubChem.
        - The function relies on the availability of the remote service and its databases, which may change over time.
    [/LIMITATIONS]
    """
    try:
        return remote_call(
            function_name="get_number_isomers_pubchem", env_name="chemenv"
        )(compound=compound)
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


@tool
def get_ghs_classification_pubchem(compound: str) -> dict:
    r"""[BRIEF] Get the GHS classification of a compound from PubChem. [\BRIEF]

    [DETAILED] This function retrieves the GHS (Globally Harmonized System) classification of a compound from PubChem.
    It uses a remote function call to the `get_ghs_classification_pubchem` function in the `chemenv` environment.
    The GHS classification provides information about the hazards associated with the compound, such as its flammability, toxicity, and environmental impact. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to find the GHS classification of a compound based on its identifier (e.g., SMILES string, PubChem CID, or InChI notation).
    - When you want to understand the hazards associated with a compound for safety and regulatory compliance. [\PROCEDURAL]


    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have the identifier of the compound you want to query (e.g., SMILES string, PubChem CID, or InChI notation). [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the compound identifier to retrieve its GHS classification. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved GHS classification in subsequent tasks or for further analysis, such as assessing the safety and regulatory compliance of the compound, or answering the question at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a compound identifier (e.g., SMILES string, PubChem CID, or InChI notation) as input.
    - Calls the remote function `get_ghs_classification_pubchem` in the `chemenv` environment with the compound identifier.
    - The remote function searches PubChem for the GHS classification of the compound.
    - Returns the GHS classification of the compound as a dictionary containing information about its hazards. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_ghs_classification_pubchem("CCO")`,
        `get_ghs_classification_pubchem("C1=CC=CC=C1")`,
        `get_ghs_classification_pubchem("C(C(=O)O)N")`,
        `get_ghs_classification_pubchem("C1=CC=CC=C1O")`,
        `get_ghs_classification_pubchem("C1=CC=CC=C1C(=O)O")`,
    ]
    [\SYNTACTICAL]

    Args:
        compound (str):
                        [BRIEF] The compound to search for in PubChem [\BRIEF]
                        [DETAILED] The compound to search for in PubChem. It can be a SMILES string, a PubChem CID, or InChI notation. It should represent the chemical structure of the compound for which you want to find the GHS classification. [\DETAILED]
                        [SYNTACTICAL] Format: "SMILES string, PubChem CID, or InChI notation" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "2244" [\EXAMPLES]

    Returns:
        dict:
                        [BRIEF] The GHS classification of the compound [\BRIEF]
                        [DETAILED] The GHS classification of the compound derived from its identifier in PubChem. It is a dictionary containing information about the hazards associated with the compound, such as its flammability, toxicity, and environmental impact. [\DETAILED]
                        [EXAMPLES] Examples: {"flammable": True, "toxic": False, "environmental_hazard": True} [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the remote function call or if the compound identifier is invalid [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `get_ghs_classification_pubchem`, such as network issues, invalid compound identifier, or if the GHS classification cannot be determined for the compound. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid compound. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The compound identifier must be valid and represent a chemical structure that can be interpreted by the remote function.
        - The remote function may not find the GHS classification for all compounds, especially if they are not well-documented in PubChem.
        - The function relies on the availability of the remote service and its databases, which may change over time.
        - It will return an empty dictionary if the GHS classification is not available for the compound.
    [/LIMITATIONS]
    """
    try:
        return remote_call(
            function_name="get_ghs_classification_pubchem", env_name="chemenv"
        )(compound=compound)
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


@tool
def get_ms_spectra_pubchem(compound: str) -> dict:
    r"""[BRIEF] Get the MS (Mass Spectrometry) spectra of a compound from PubChem. [\BRIEF]

    [DETAILED] This function retrieves the MS spectra of a compound from PubChem.
    It uses a remote function call to the `get_ms_spectra_pubchem` function in the `chemenv` environment.
    The MS spectra provide information about the mass-to-charge ratio of ions produced from the compound, which can be useful for identifying and characterizing the compound. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to find the MS spectra of a compound based on its identifier (e.g., SMILES string, PubChem CID, or InChI notation).
    - When you want to analyze the mass spectrum of a compound for identification or characterization purposes.
    - Recommended for tasks that require information about the MS spectra of a compound. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have the identifier of the compound you want to query (e.g., SMILES string, PubChem CID, or InChI notation). [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the compound identifier to retrieve its MS spectra. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved MS spectra in subsequent tasks or for further analysis, such as identifying the compound or understanding its fragmentation pattern, or answering the task at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a compound identifier (e.g., SMILES string, PubChem CID, or InChI notation) as input.
    - Calls the remote function `get_ms_spectra_pubchem` in the `chemenv` environment with the compound identifier.
    - The remote function searches PubChem for the MS spectra of the compound.
    - Returns the MS spectra of the compound as a dictionary containing the 5 top peaks, information about the equipment, solvent, etc.  [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_ms_spectra_pubchem("CCO")`,
        `get_ms_spectra_pubchem("C1=CC=CC=C1")`,
        `get_ms_spectra_pubchem("C(C(=O)O)N")`,
        `get_ms_spectra_pubchem("C1=CC=CC=C1O")`,
        `get_ms_spectra_pubchem("C1=CC=CC=C1C(=O)O")`,
    ]
    [\SYNTACTICAL]

    Args:
        compound (str):
                        [BRIEF] The compound to search for in PubChem [\BRIEF]
                        [DETAILED] The compound to search for in PubChem. It can be a SMILES string, a PubChem CID, or InChI notation. It should represent the chemical structure of the compound for which you want to find the MS spectra. [\DETAILED]
                        [SYNTACTICAL] Format: "SMILES string, PubChem CID, or InChI notation" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "2244" [\EXAMPLES]

    Returns:
        dict:
                        [BRIEF] The MS spectra of the compound [\BRIEF]
                        [DETAILED] The MS spectra of the compound derived from its identifier in PubChem. It is a dictionary containing information about the mass-to-charge ratio of ions produced from the compound, which can be useful for identifying and characterizing the compound. [\DETAILED]
                        [EXAMPLES] Examples: {"spectrum": ["Top Peaks": {"StringWithMarkup": [{"String": "195.0 1"}, {"String": "120.0 0.81"}, {"String": "92.0 0.42"}, {"String": "135.0 0.39"}, {"String": "210.0 0.28"}]}} [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the remote function call or if the compound identifier is invalid [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `get_ms_spectra_pubchem`, such as network issues, invalid compound identifier, or if the MS spectra cannot be determined for the compound. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid compound. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The compound identifier must be valid and represent a chemical structure that can be interpreted by the remote function.
        - The remote function may not find MS spectra for all compounds, especially if they are not well-documented in PubChem.
        - The function relies on the availability of the remote service and its databases, which may change over time.
    [/LIMITATIONS]
    """
    try:
        return remote_call(function_name="get_ms_spectra_pubchem", env_name="chemenv")(
            compound=compound
        )
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


@tool
def get_h_nmr_spectra_pubchem(compound: str) -> dict:
    r"""[BRIEF] Get the 1H-NMR spectra of a compound from PubChem. [\BRIEF]

    [DETAILED] This function retrieves the 1H-NMR (proton nuclear magnetic resonance) spectra of a compound from PubChem.
    It uses a remote function call to the `get_h_nmr_spectra_pubchem` function in the `chemenv` environment.
    The 1H-NMR spectra provide information about the hydrogen atoms in the compound, which can be useful for identifying and characterizing the compound's structure. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to find the 1H-NMR spectra of a compound based on its identifier (e.g., SMILES string, PubChem CID, or InChI notation).
    - When you want to analyze the NMR spectrum of a compound for identification or characterization purposes.
    - Recommended for tasks that require information about the 1H-NMR spectra of a compound. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have the identifier of the compound you want to query (e.g., SMILES string, PubChem CID, or InChI notation). [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the compound identifier to retrieve its 1H-NMR spectra. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved 1H-NMR spectra in subsequent tasks or for further analysis, such as identifying the compound or understanding its hydrogen atom environment, or answering the task at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a compound identifier (e.g., SMILES string, PubChem CID, or InChI notation) as input.
    - Calls the remote function `get_h_nmr_spectra_pubchem` in the `chemenv` environment with the compound identifier.
    - The remote function searches PubChem for the 1H-NMR spectra of the compound.
    - Returns the 1H-NMR experimental spectra of the compound as a dictionary containing information about the hydrogen atoms in the compound, including the shifts, solvent and experimental equipment. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_h_nmr_spectra_pubchem("CCO")`,
        `get_h_nmr_spectra_pubchem("C1=CC=CC=C1")`,
        `get_h_nmr_spectra_pubchem("C(C(=O)O)N")`,
        `get_h_nmr_spectra_pubchem("C1=CC=CC=C1O")`,
        `get_h_nmr_spectra_pubchem("C1=CC=CC=C1C(=O)O")`,
    ]
    [\SYNTACTICAL]

    Args:
        compound (str):
                        [BRIEF] The compound to search for in PubChem [\BRIEF]
                        [DETAILED] The compound to search for in PubChem. It can be a SMILES string, a PubChem CID, or InChI notation. It should represent the chemical structure of the compound for which you want to find the 1H-NMR spectra. [\DETAILED]
                        [SYNTACTICAL] Format: "SMILES string, PubChem CID, or InChI notation" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "2244" [\EXAMPLES]

    Returns:
        dict:
                        [BRIEF] The 1H-NMR spectra of the compound [\BRIEF]
                        [DETAILED] The 1H-NMR spectra of the compound derived from its identifier in PubChem. It is a dictionary containing information about the hydrogen atoms in the compound, which can be useful for identifying and characterizing the compound's structure. [\DETAILED]
                        [EXAMPLES] Examples: {'1': {'instrument': 'Bruker', 'frequency': '400 MHz',...}} [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the remote function call or if the compound identifier is invalid [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `get_h_nmr_spectra_pubchem`, such as network issues, invalid compound identifier, or if the 1H-NMR spectra cannot be determined for the compound. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid compound. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The compound identifier must be valid and represent a chemical structure that can be interpreted by the remote function.
        - The remote function may not find 1H-NMR spectra for all compounds, especially if they are not well-documented in PubChem.
        - The function relies on the availability of the remote service and its databases, which may change over time.
    [/LIMITATIONS]
    """
    try:
        return remote_call(
            function_name="get_h_nmr_spectra_pubchem", env_name="chemenv"
        )(compound=compound)
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


@tool
def get_c_nmr_spectra_pubchem(compound: str) -> str:
    r"""[BRIEF] Get the 13C-NMR spectra of a compound from PubChem. [\BRIEF]

    [DETAILED] This function retrieves the 13C-NMR (carbon nuclear magnetic resonance) spectra of a compound from PubChem.
    It uses a remote function call to the `get_c_nmr_spectra_pubchem` function in the `chemenv` environment.
    The 13C-NMR spectra provide information about the carbon atoms in the compound, which can be useful for identifying and characterizing the compound's structure. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to find the 13C-NMR spectra of a compound based on its identifier (e.g., SMILES string, PubChem CID, or InChI notation).
    - When you want to analyze the NMR spectrum of a compound for identification or characterization purposes.
    - Recommended for tasks that require information about the 13C-NMR spectra of a compound. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have the identifier of the compound you want to query (e.g., SMILES string, PubChem CID, or InChI notation). [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the compound identifier to retrieve its 13C-NMR spectra. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved 13C-NMR spectra in subsequent tasks or for further analysis, such as identifying the compound or understanding its carbon atom environment, or answering the task at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a compound identifier (e.g., SMILES string, PubChem CID, or InChI notation) as input.
    - Calls the remote function `get_c_nmr_spectra_pubchem` in the `chemenv` environment with the compound identifier.
    - The remote function searches PubChem for the 13C-NMR spectra of the compound.
    - Returns the 13C-NMR spectra of the compound as a string containing information about the carbon atoms in the compound. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_c_nmr_spectra_pubchem("CCO")`,
        `get_c_nmr_spectra_pubchem("C1=CC=CC=C1")`,
        `get_c_nmr_spectra_pubchem("C(C(=O)O)N")`,
        `get_c_nmr_spectra_pubchem("C1=CC=CC=C1O")`,
        `get_c_nmr_spectra_pubchem("C1=CC=CC=C1C(=O)O")`,
    ]
    [\SYNTACTICAL]

    Args:
        compound (str):
                        [BRIEF] The compound to search for in PubChem [\BRIEF]
                        [DETAILED] The compound to search for in PubChem. It can be a SMILES string, a PubChem CID, or InChI notation. It should represent the chemical structure of the compound for which you want to find the 13C-NMR spectra. [\DETAILED]
                        [SYNTACTICAL] Format: "SMILES string, PubChem CID, or InChI notation" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "2244" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] The 13C-NMR spectra of the compound [\BRIEF]
                        [DETAILED] The 13C-NMR spectra of the compound derived from its identifier in PubChem. It is a string containing information about the carbon atoms in the compound, which can be useful for identifying and characterizing the compound's structure. [\DETAILED]
                        [EXAMPLES] Examples: "{'1': {'instrument': 'Bruker', 'frequency': '400 MHz',...}}" [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the remote function call or if the compound identifier is invalid [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `get_c_nmr_spectra_pubchem`, such as network issues, invalid compound identifier, or if the 13C-NMR spectra cannot be determined for the compound. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid compound. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The compound identifier must be valid and represent a chemical structure that can be interpreted by the remote function.
        - The remote function may not find 13C-NMR spectra for all compounds, especially if they are not well-documented in PubChem.
        - The function relies on the availability of the remote service and its databases, which may change over time.
    [/LIMITATIONS]
    """
    try:
        return remote_call(
            function_name="get_c_nmr_spectra_pubchem", env_name="chemenv"
        )(compound=compound)
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


@tool
def simulate_spectra(smiles: str) -> dict[str, str]:
    r"""[BRIEF] Simulate 1H NMR, 13C NMR, and IR spectra for a given molecule using its SMILES string. [\BRIEF]

    [DETAILED] This function simulates the 1H NMR, 13C NMR, and IR spectra for a given molecule using its SMILES string.
    It uses a remote function call to the `simulate_spectra` function in the `chemenv` environment.
    If some of the spectra are not available, the function will return None for those spectra.
    This can be used to complement the PubChem data or to provide an estimate of the spectra for a compound. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to simulate the spectra of a compound based on its SMILES representation.
    - When you want to obtain estimated spectra for a compound that may not have experimental data available.
    - Recommended for tasks that require simulated spectra for a compound, such as in cheminformatics or computational chemistry. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have the SMILES representation of the compound you want to simulate spectra for. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the SMILES string to retrieve the simulated spectra. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved spectra in subsequent tasks or for further analysis, such as comparing with experimental data, understanding the compound's structure, or answering the task at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a SMILES string as input, which represents the chemical structure of the compound.
    - Calls the remote function `simulate_spectra` in the `chemenv` environment with the SMILES string.
    - The remote function simulates the 1H NMR, 13C NMR, and IR spectra for the compound.
    - Returns a dictionary containing the simulated spectra for the compound. If some spectra are not available, they will be returned as None. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `simulate_spectra("CCO")`,
        `simulate_spectra("C1=CC=CC=C1")`,
        `simulate_spectra("C(C(=O)O)N")`,
        `simulate_spectra("C1=CC=CC=C1O")`,
        `simulate_spectra("C1=CC=CC=C1C(=O)O")`,
    ]
    [\SYNTACTICAL]

    Args:
        smiles (str):
                        [BRIEF] The SMILES representation of the compound [\BRIEF]
                        [DETAILED] The SMILES representation of the compound for which you want to simulate the spectra. It should represent the chemical structure of the compound. [\DETAILED]
                        [SYNTACTICAL] Format: "SMILES string" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N" [\EXAMPLES]

    Returns:
        dict[str, str]:
                        [BRIEF] The simulated spectra of the compound [\BRIEF]
                        [DETAILED] The simulated spectra of the compound derived from its SMILES representation. It is a dictionary containing the simulated 1H NMR, 13C NMR, and IR spectra for the compound. If some spectra are not available, they will be returned as None. [\DETAILED]
                        [EXAMPLES] Examples: {"1H NMR": "spectrum_data_1H", "13C NMR": "spectrum_data_13C", "IR": "spectrum_data_IR"} [\EXAMPLES]

    [RAISES] Exceptions:
        Exception:
                        [ERROR_WHEN] If a network error occurs during the remote function call. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `simulate_spectra`, such as network issues. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] This tool is unavailable if the remote function cannot be called. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The SMILES string must be valid and represent a chemical structure that can be interpreted by the remote function.
        - The remote function may not be able to simulate spectra for all compounds, especially if they are complex or not well-defined.
        - The function relies on the availability of the remote service and its simulation capabilities, which may change over time.
    [/LIMITATIONS]
    """
    return remote_call(function_name="simulate_spectra", env_name="chemenv")(
        smiles=smiles
    )


@tool
def get_functional_groups(smiles: str) -> list[str]:
    r"""[BRIEF] Get the functional groups of a compound from its SMILES representation. [\BRIEF]

    [DETAILED] This function retrieves the functional groups of a compound based on its SMILES representation.
    It uses a remote function call to the `get_functional_groups` function in the `chemenv` environment.
    The remote function uses the Python package `exmol` to return a list of the names of the different functional groups present in the molecule. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to identify the functional groups present in a compound based on its SMILES representation.
    - When you want to analyze the chemical structure of a compound to understand its reactivity and properties.
    - Recommended for tasks that require information about the functional groups of a compound, such as in organic chemistry or drug design. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have the SMILES representation of the compound you want to analyze. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the SMILES string to retrieve the functional groups. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved functional groups in subsequent tasks or for further analysis, such as understanding the compound's reactivity, predicting its behavior in chemical reactions, or answering the task at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a SMILES string as input, which represents the chemical structure of the compound.
    - Calls the remote function `get_functional_groups` in the `chemenv` environment with the SMILES string.
    - The remote function analyzes the SMILES representation to identify the functional groups present in the compound.
    - Returns a list of functional group names found in the compound. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_functional_groups("CCO")`,
        `get_functional_groups("C1=CC=CC=C1")`,
        `get_functional_groups("C(C(=O)O)N")`,
        `get_functional_groups("C1=CC=CC=C1O")`,
        `get_functional_groups("C1=CC=CC=C1C(=O)O")`,
    ]
    [\SYNTACTICAL]

    Args:
        smiles (str):
                        [BRIEF] The SMILES representation of the compound [\BRIEF]
                        [DETAILED] The SMILES representation of the compound for which you want to identify the functional groups. It should represent the chemical structure of the compound. [\DETAILED]
                        [SYNTACTICAL] Format: "SMILES string" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N" [\EXAMPLES]

    Returns:
        list[str]:
                        [BRIEF] The names of the functional groups of the compound [\BRIEF]
                        [DETAILED] The names of the functional groups present in the compound derived from its SMILES representation. It is a list of strings, each representing a functional group found in the compound. [\DETAILED]
                        [EXAMPLES] Examples: ["alcohol", "aromatic ring", "carboxylic acid"] [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the remote function call or if the SMILES string is invalid [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in calling the remote function `get_functional_groups`, such as network issues, invalid SMILES string, or if the functional groups cannot be determined for the compound. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid SMILES string. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The SMILES string must be valid and represent a chemical structure that can be interpreted by the remote function.
        - The remote function may not be able to identify functional groups for all compounds, especially if they are complex or not well-defined.
        - The function relies on the availability of the remote service and its databases, which may change over time.
    [/LIMITATIONS]
    """
    try:
        return remote_call(function_name="get_functional_groups", env_name="chemenv")(
            smiles=smiles
        )
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


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
        raise e

    finally:
        delete_vector_db(collection_name)


@tool
def search_clinical_trials_by_query(query: str, top_k: int = 5) -> list[dict]:
    r"""[BRIEF] Search for clinical trials based on a specific query. [\BRIEF]

    [DETAILED] This function fetches clinical trial data from ClinicalTrials.gov
    and returns the most relevant trials based on the provided query.
    It uses semantic search capabilities to find trials that match the query. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you want to find clinical trials related to a specific topic or condition.
    - When you have a specific query in mind and want to retrieve the most relevant clinical trials.
    - Recommended for tasks that require information about clinical trials, such as drug efficacy, safety, or treatment protocols. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have a specific query related to clinical trials (e.g., a drug name, condition, or treatment). [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the query to retrieve the most relevant clinical trials. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved clinical trials in subsequent tasks or for further analysis, such as understanding treatment options, evaluating drug efficacy, or answering the task at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a text query as input to search for relevant clinical trials.
    - Calls the helper function `_search_clinical_trials` to fetch clinical trial data from ClinicalTrials.gov.
    - The function retrieves all studies related to the query, parses the data, and performs semantic search.
    - Returns a list of dictionaries containing the most relevant clinical trial data based on the query. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `search_clinical_trials_by_query("COVID-19 vaccine")`,
        `search_clinical_trials_by_query("diabetes treatment")`,
        `search_clinical_trials_by_query("cancer immunotherapy")`,
        `search_clinical_trials_by_query("heart disease clinical trials")`,
        `search_clinical_trials_by_query("Alzheimer's disease research")`,
    ]
    [\SYNTACTICAL]

    Args:
        query (str):
                        [BRIEF] The text query to search for relevant clinical trials [\BRIEF]
                        [DETAILED] The text query to find relevant clinical trials. It should be a specific topic, condition, or treatment related to clinical trials. [\DETAILED]
                        [SYNTACTICAL] Format: "any valid string" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "COVID-19 vaccine", "diabetes treatment", "cancer immunotherapy" [\EXAMPLES]

        top_k (int):
                        [BRIEF] The number of top results to return (default: 5) [\BRIEF]
                        [DETAILED] The number of top results to return based on relevance to the query. It determines how many of the most relevant clinical trials will be included in the response. [\DETAILED]
                        [SYNTACTICAL] Format: "integer" [\SYNTACTICAL]
                        [EXAMPLES] Examples: 5, 10, 20 [\EXAMPLES]

    Returns:
        list[dict]:
                        [BRIEF] A list of dictionaries containing the most relevant clinical trial data [\BRIEF]
                        [DETAILED] A list of dictionaries with parsed clinical trial data, including information such as trial title, summary, status, conditions, interventions, and outcomes. The data is filtered based on relevance to the provided query. [\DETAILED]
                        [EXAMPLES] Examples: [{"NCT ID": "NCT123456", "Official Title": "Study on COVID-19 Vaccine", "Status": "Recruiting", "Conditions": ["COVID-19"], "Interventions": ["Vaccine A"], "Brief Summary": "This study evaluates the efficacy of Vaccine A against COVID-19."}] [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the search or data retrieval [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in fetching or parsing clinical trial data, such as network issues, invalid query, or if no relevant trials are found. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid query. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The function relies on the availability of ClinicalTrials.gov and its API.
        - The search results may vary based on the specificity and relevance of the query.
        - The function may return a limited number of results based on the `top_k` parameter.
        - The semantic search may not always yield the most relevant results, especially for broad or ambiguous queries.
    [/LIMITATIONS]
    """
    try:
        return _search_clinical_trials(search_term=query, query=query, top_k=top_k)
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


@tool
def search_clinical_trials_by_drug(
    drug_name: str, query: str, top_k: int = 5
) -> list[dict]:
    r"""[BRIEF] Search for clinical trials related to a specific drug with a query. [\BRIEF]

    [DETAILED] This function fetches clinical trial data for a specific drug from ClinicalTrials.gov
    and returns the most relevant trials based on the provided query.
    It uses semantic search capabilities to find trials that match the drug name and query. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you want to find clinical trials related to a specific drug and have a query in mind.
    - When you have a specific drug name and query related to clinical trials.
    - Recommended for tasks that require information about clinical trials for a specific drug, such as drug efficacy, safety, or treatment protocols. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have the name of the drug and a specific query related to clinical trials. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the drug name and query to retrieve the most relevant clinical trials. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved clinical trials in subsequent tasks or for further analysis, such as understanding treatment options, evaluating drug efficacy, or answering the task at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a drug name and a text query as input to search for relevant clinical trials.
    - Calls the helper function `_search_clinical_trials` to fetch clinical trial data from ClinicalTrials.gov.
    - The function retrieves all studies related to the drug name, parses the data, and performs semantic search based on the query.
    - Returns a list of dictionaries containing the most relevant clinical trial data based on the drug name and query. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `search_clinical_trials_by_drug("Aspirin", "pain relief")`,
        `search_clinical_trials_by_drug("Metformin", "diabetes treatment")`,
        `search_clinical_trials_by_drug("Pembrolizumab", "cancer immunotherapy")`,
        `search_clinical_trials_by_drug("Atorvastatin", "cholesterol management")`,
        `search_clinical_trials_by_drug("Adalimumab", "rheumatoid arthritis")`,
    ]
    [\SYNTACTICAL]


    Args:
        drug_name (str):
                        [BRIEF] The name of the drug to search for in clinical trials [\BRIEF]
                        [DETAILED] The name of the drug for which you want to find relevant clinical trials. It should be a specific drug name that is recognized in clinical trial databases. [\DETAILED]
                        [SYNTACTICAL] Format: "drug name" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "Aspirin", "Metformin", "Pembrolizumab" [\EXAMPLES]

        query (str):
                        [BRIEF] The text query to search for relevant clinical trials [\BRIEF]
                        [DETAILED] The text query to find relevant clinical trials related to the drug. It should be a specific topic, condition, or treatment related to the drug. [\DETAILED]
                        [SYNTACTICAL] Format: "text query" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "pain relief", "diabetes treatment", "cancer immunotherapy" [\EXAMPLES]

        top_k (int):
                        [BRIEF] The number of top results to return (default: 5) [\BRIEF]
                        [DETAILED] The number of top results to return based on relevance to the query. It determines how many of the most relevant clinical trials will be included in the response. [\DETAILED]
                        [SYNTACTICAL] Format: "integer" [\SYNTACTICAL]
                        [EXAMPLES] Examples: 5, 10, 20 [\EXAMPLES]

    Returns:
        list[dict]:
                        [BRIEF] A list of dictionaries containing the most relevant clinical trial data [\BRIEF]
                        [DETAILED] A list of dictionaries with parsed clinical trial data, including information such as trial title, summary, status, conditions, interventions, and outcomes. The data is filtered based on relevance to the provided drug name and query. [\DETAILED]
                        [EXAMPLES] Examples: [{"NCT ID": "NCT123456", "Official Title": "Study on Aspirin for Pain Relief", "Status": "Recruiting", "Conditions": ["Pain"], "Interventions": ["Aspirin"], "Brief Summary": "This study evaluates the efficacy of Aspirin for pain relief."}] [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the search or data retrieval [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in fetching or parsing clinical trial data, such as network issues, invalid drug name, or if no relevant trials are found. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying a valid drug name and query. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The function relies on the availability of ClinicalTrials.gov and its API.
        - The search results may vary based on the specificity and relevance of the drug name and query.
        - The function may return a limited number of results based on the `top_k` parameter.
        - The search might be slow if there are many studies related to the drug name, as it fetches all studies first before filtering.
    [/LIMITATIONS]
    """
    try:
        return _search_clinical_trials(search_term=drug_name, query=query, top_k=top_k)
    except Exception as e:
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e


@tool
def search_materials_compatibility(
    material: str, chemical: str, top_k: int = 10
) -> list[dict]:
    r"""[BRIEF] Search for materials compatibility data based on a material and chemical. [\BRIEF]

    [DETAILED] This function searches for materials compatibility data in a dedicated dataset based on the provided material and chemical.
    It uses a vector database to find the most relevant materials compatibility data that matches the input material and chemical. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to find compatibility information between a specific material and a chemical.
    - When you want to assess the suitability of a material for use with a particular chemical.
    - Recommended for tasks that require information about materials compatibility, such as in materials science, engineering, or chemical processing. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that you have the names of the material and chemical you want to assess compatibility for. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the material and chemical names to retrieve the most relevant materials compatibility data. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved materials compatibility data in subsequent tasks or for further analysis, such as evaluating material selection, understanding chemical interactions, or answering the task at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Takes a material name and a chemical name as input to search for compatibility data.
    - Uses a vector database to perform a semantic search based on the provided material and chemical.
    - The function queries the materials compatibility dataset to find the most relevant data that matches the input.
    - Returns a list of dictionaries containing the most relevant materials compatibility data. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `search_materials_compatibility("Polyethylene", "Acetic Acid")`,
        `search_materials_compatibility("Stainless Steel", "Hydrochloric Acid")`,
        `search_materials_compatibility("Glass", "Sodium Hydroxide")`,
        `search_materials_compatibility("Rubber", "Benzene")`,
        `search_materials_compatibility("PVC", "Ethanol")`,
    ]
    [\SYNTACTICAL]

    Args:
        material (str):
                        [BRIEF] The material to search for compatibility data [\BRIEF]
                        [DETAILED] The material for which you want to find compatibility data. It should be a specific material name that is recognized in materials compatibility databases. [\DETAILED]
                        [SYNTACTICAL] Format: "material name" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "Polyethylene", "Stainless Steel", "Glass" [\EXAMPLES]
        chemical (str):
                        [BRIEF] The chemical to search for compatibility data [\BRIEF]
                        [DETAILED] The chemical for which you want to find compatibility data. It should be a specific chemical name that is recognized in materials compatibility databases. [\DETAILED]
                        [SYNTACTICAL] Format: "chemical name" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "Acetic Acid", "Hydrochloric Acid", "Sodium Hydroxide" [\EXAMPLES]
        top_k (int):
                        [BRIEF] The number of top results to return (default: 10) [\BRIEF]
                        [DETAILED] The number of top results to return based on relevance to the material and chemical. It determines how many of the most relevant materials compatibility data will be included in the response. [\DETAILED]
                        [SYNTACTICAL] Format: "integer" [\SYNTACTICAL]
                        [EXAMPLES] Examples: 5, 10, 20 [\EXAMPLES]

    Returns:
        list[dict]:
                        [BRIEF] A list of dictionaries containing the most relevant materials compatibility data [\BRIEF]
                        [DETAILED] A list of dictionaries with materials compatibility data, including information such as material name, chemical name, compatibility status, and any relevant notes. The data is filtered based on relevance to the provided material and chemical. [\DETAILED]
                        [EXAMPLES] Examples: [{"material": "Polyethylene", "chemical": "Acetic Acid", "compatibility": "Compatible", "notes": "Good resistance at room temperature."}] [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If an error occurs during the search or data retrieval [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an error in searching materials compatibility data, such as network issues, invalid material or chemical names, or if no relevant data is found. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] The error message will include the original error message and a full traceback for debugging purposes. If it has to do with the input, try to solve the error trying valid material and chemical names. Otherwise do not try to solve the error. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The function relies on the availability of a materials compatibility dataset and its vector database.
        - The search results may vary based on the specificity and relevance of the material and chemical names.
        - The function may return a limited number of results based on the `top_k` parameter.
    [/LIMITATIONS]
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
        error_details = traceback.format_exc()
        raise ValueError(f"Error: {e}\n\nFull traceback:\n{error_details}") from e
