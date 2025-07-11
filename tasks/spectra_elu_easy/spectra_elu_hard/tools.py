import gc
import os
import traceback
import uuid
from pathlib import Path
from typing import Any

import chromadb
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from corral.base import Tool
from corral.io import (
    FSManager,
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
def online_search(query: str, num_results: int = 5) -> list[dict]:
    """[BRIEF] Perform a web search using Brave Search,
    then filter and rank results using embeddings. [/BRIEF]

    [DETAILED] This function performs a web search using Brave Search API,
    retrieves the top results for a query, and then filters and ranks them based on their
    relevance to the search query using embeddings. It returns a list of the most
    relevant search results, each containing the content and metadata of the
    search result, sorted by similarity score. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when the other specific seach tools are not available
    or when you need to perform a general web search.
    - When you need to find information that is not available
    in the local vector database or other specialized databases.
    - When some you need to retrieve some information that is not
    available with the other tools.
    - Recommended for general query seaches that do not require
    specific databases or structured data. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the other tools (`relevant_pubchem_sections` and
    `search_by_smiles`) are not suitable for the query. [/PREREQUISITE]
    2. [CURRENT] Apply this tool with a descriptive query string to perform a web search. [/CURRENT]
    3. [FOLLOW_UP] Use the information from the web search combined with retrieved from the shift
    tools (`retrieve_protons_shifts`, `retrieve_aromatic_protons_shifts` and
    `retrieve_carbon_shifts`) to solve the task validating with
    the `simulate_spectra` tool. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Checks if a BRAVE_SEARCH_API_KEY environment variable is set.
    - If not set, raises a ValueError.
    - Uses the Brave Search API to perform a web search with the provided query.
    - Retrieves the top `num_results` results.
    - Filters and ranks the results based on their relevance to the search query using embeddings.
    - Returns a list of dictionaries containing the most relevant search results,
    each with its content and metadata, sorted by similarity score that
    is also included. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `online_search("What is the 1H-NMR spectra of CCO?")`,
        `online_search("What is the chemical formula of aspirin?")`,
        `online_search("What are the chemical shifts of a carbonyl group in Carbon NMR?")`,
        `online_search("What is the chemical shift of a methyl group in 1H-NMR?")`,
        `online_search("What is the chemical shift of a methylene group in 1H-NMR?")`,
    ]
    [/SYNTACTICAL]

    Args:
        query (str):
            [BRIEF] The search query string [/BRIEF]
            [DETAILED] The query string to search for in the Brave Search API.
            It should be a descriptive string that represents the
            information you are looking for. [/DETAILED]
            [SYNTACTICAL] Format: "string with no special requirements" [/SYNTACTICAL]
            [EXAMPLES] Examples: "What is the 1H-NMR spectra of CCO?",
            "What is the chemical formula of aspirin?",
            "What are the chemical shifts of a carbonyl group in Carbon NMR?"[/EXAMPLES]

        num_results (int, optional):
            [BRIEF] Maximum number of results to return. Defaults to 5 [/BRIEF]
            [DETAILED] The maximum number of search results to return from the
            Brave Search API. It should be a positive integer. [/DETAILED]
            [SYNTACTICAL] Format: "any positive integer
            (e.g., 5, 10, 20)" [/SYNTACTICAL]
            [EXAMPLES] Examples: 5, 10, 20 [/EXAMPLES]

    Returns:
        list[dict]:
            [BRIEF] A list of dictionaries containing the most relevant
            search results with their content and metadata,
            sorted by similarity score [/BRIEF]
            [DETAILED] Each dictionary contains the content of the search result,
            its metadata, and a similarity score indicating how relevant the
            result is to the search query. The results are sorted by similarity
            score in descending order. [/DETAILED]
            [EXAMPLES] Examples: [{"content": "Result 1 content",
            "metadata": {"source": "brave"}, "similarity_score": 0.95},
            {"content": "Result 2 content", "metadata": {"source": "brave"},
            "similarity_score": 0.90}, ...] [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the BRAVE_SEARCH_API_KEY environment variable
            is not set or if an error occurs during the search. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when the
            BRAVE_SEARCH_API_KEY is not set, or if there is an
            error in making the API request to Brave Search,
            such as network issues or invalid query parameters. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try a different tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires a valid BRAVE_SEARCH_API_KEY environment variable to be set.
        - The number of results returned is limited by the
        `num_results` parameter, which defaults to 5.
        - The search results are filtered and ranked based on their
        relevance to the search query using embeddings,
        which may not always yield the most relevant results.
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
    """[BRIEF] Retrieve relevant sections from PubChem JSON
    data based on a compound and a text query. [/BRIEF]

    [DETAILED] This function retrieves the most relevant
    sections from a JSON file containing compound data
    based on a given compound and a text query.
    It processes the PubChem JSON data to extract sections,
    creates a vector database from the sections,
    and performs a search based on the query.
    It returns a list of the most relevant sections,
    each containing the section name, root path, description,
    and original data. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When other more specific tools do not work for the task at hand.
    - Use it when you need to search through PubChem records for a specific compound.
    - When you want to retrieve relevant sections from PubChem data based on a text query.
    - Recommended for tasks that require detailed information about a compound from PubChem,
    such as chemical properties, structures, or biological activities. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the specific other tools such as `search_by_smiles`
    do not work for the task at hand  [/PREREQUISITE]
    2. [CURRENT] Apply this tool with the compound and a descriptive query string
    to retrieve relevant sections from PubChem. [/CURRENT]
    3. [FOLLOW_UP] Use the information from the retrieved sections to answer the task.
    Validate the answer using the `simulate_spectra` tool. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Retrieves the full record of a compound from PubChem using the
    `get_pubchem_full_record` remote function.
    - Processes the JSON data to extract sections based on the `TOCHeading` field.
    - Creates a vector database from the extracted sections,
    with metadata including section names and paths.
    - Searches the vector database for sections that are most relevant to the provided text query.
    - Returns a list of the most relevant sections, each containing the section name,
    root path, description, and original data. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `relevant_pubchem_sections("CCO", "Mass spectrometry properties")`,
        `relevant_pubchem_sections("C1=CC=CC=C1", "NMR spectra")`,
        `relevant_pubchem_sections("C(C(=O)O)N", "1H NMR spectra")`,
        `relevant_pubchem_sections("C1=CC=C(C=C1)C(=O)O", "IR spectra")`,
        `relevant_pubchem_sections("C1=CC=CC=C1", "Carbon NMR spectra chemical shifts")`,`
    ]
    [/SYNTACTICAL]

    Args:
        compound (str):
            [BRIEF] The compound to search for in PubChem [/BRIEF]
            [DETAILED] The compound to search for in PubChem.
            It can be a SMILES string, a PubChem CID, or InChI notation. [/DETAILED]
            [SYNTACTICAL] Format: "SMILES string, PubChem CID, or InChI notation" [/SYNTACTICAL]
            [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "2244" [/EXAMPLES]

        query (str):
            [BRIEF] The text query to search for in the full record of the compound [/BRIEF]
            [DETAILED] The text query to search for in the full record of the compound.
            It should be a descriptive string that represents the information
            you are looking for in the PubChem record. [/DETAILED]
            [SYNTACTICAL] Format: "string with no special requirements" [/SYNTACTICAL]
            [EXAMPLES] Examples: "Mass spectrometry properties", "NMR spectra",
            "1H NMR spectra", "IR spectra", "Carbon NMR spectra chemical shifts" [/EXAMPLES]

        top_k (int, optional):
            [BRIEF] The number of top results to return. Defaults to 5 [/BRIEF]
            [DETAILED] The number of top results to return from the search.
            It should be a positive integer. [/DETAILED]
            [SYNTACTICAL] Format: "any positive integer (e.g., 5, 10, 20)" [/SYNTACTICAL]
            [EXAMPLES] Examples: 5, 10, 20 [/EXAMPLES]

    Returns:
        list[dict[str, Any]]:
            [BRIEF] A list of dictionaries containing the most relevant
            sections from the PubChem record [/BRIEF]
            [DETAILED] Each dictionary contains the section name, root path, description,
            and original data of the section. The sections are sorted by relevance
            to the provided text query. [/DETAILED]
            [EXAMPLES] Examples: [{"name": "Mass Spectrometry", "root_path":
            "Record.Section[0].TOCHeading", "description": "Mass spectrometry
            properties of the compound", "original": {...}}, {"name": "NMR Spectra",
            "root_path": "Record.Section[1].TOCHeading", "description":
            "NMR spectra of the compound", "original": {...}}, ...] [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If an error occurs during the retrieval or processing
            of the PubChem record. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when there is an error in
            retrieving the full record of the compound from PubChem, processing the JSON data,
            or creating the vector database. It can occur due to network issues,
            invalid compound identifiers, or other unexpected errors. [/ERROR_DETAILS]
            [ERROR_RECOVERY] If it has to do with the input, try to solve the error trying
            a valid compound. Otherwise do not try to solve the error. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The retrieval of the full record from PubChem may fail if the compound
        identifier is invalid or not found.
        - The processing of the JSON data may not extract all relevant sections
        if the structure of the PubChem record changes.
        - The vector database creation and search may fail if there are issues
        with the ChromaDB client or the underlying storage.
        - The retrieval might not retrieve all meaningful sections if the PubChem
        record does not contain the expected structure.
        - The search results are limited to the `top_k` parameter, which defaults to 5.
        - The function may return an empty list if no relevant sections are found for
        the given query.
    [/LIMITATIONS]
    """
    collection_name = f"compounds_db_{uuid.uuid4().hex}"

    try:
        full_record = remote_call(
            function_name="get_pubchem_full_record", env_name="chemenv"
        )(compound=compound)

        chunks = process_pubchem_json(full_record)

        # Create metadata for each chunk
        metadatas = [
            {
                "name": chunk.get("name", ""),
                "root_path": chunk.get("root_path", ""),
                "source": "pubchem",
            }
            for chunk in chunks
        ]

        create_vector_database(
            chunks=chunks, collection_name=collection_name, metadatas=metadatas
        )
        return vector_database_search(
            query=query, collection_name=collection_name, top_k=top_k
        )

    except Exception as e:
        raise [{"error": f"Error: {e!s}"}] from e

    finally:
        delete_vector_db(collection_name)


@tool
def get_formula_from_smiles(smiles: str) -> str:
    """[BRIEF] Generate a chemical formula from a SMILES string using RDKit. [/BRIEF]

    [DETAILED] This function takes a SMILES representation of a molecule and generates
    its chemical formula in Hill notation (C, H, then alphabetical order).
    It uses RDKit to parse the SMILES string and calculate the molecular formula.
    If the SMILES string is invalid or cannot be parsed, it returns an error message. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you have a SMILES string and need to obtain the chemical formula of the
    corresponding molecule.
    - When you need to convert a SMILES representation into a chemical formula
    for further analysis or reporting.
    - Recommended for tasks that require chemical formula generation from SMILES strings,
    such as chemical structure analysis, database searches, or reporting. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have some SMILES representations of molecules that you think
    can produce the analysis results described in the task. Use the tools `retrieve_protons_shifts`,
    `retrieve_aromatic_protons_shifts` and `retrieve_carbon_shifts` for having more info
    about the chemical shifts. [/PREREQUISITE]
    2. [CURRENT] Apply this tool with a valid SMILES string to generate the
    chemical formula of the molecule. [/CURRENT]
    3. [FOLLOW_UP] Use the generated chemical formula to compare your proposed
    molecules with the molecular analysis in the task, especially with MS results. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It uses RDKit to parse the provided SMILES string and create a molecular object.
    - If the SMILES string is valid, it calculates the molecular formula
    using `rdMolDescriptors.CalcMolFormula`.
    - The formula is returned in Hill notation, which lists carbon (C) atoms first,
    followed by hydrogen (H) atoms, and then other elements in alphabetical order.
    - If the SMILES string is invalid or cannot be parsed, it returns an
    error message indicating the issue. [/CONTEXTUAL]

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
            [DETAILED] The SMILES string representing the chemical structure of the molecule.
            It should be a valid SMILES notation that RDKit can parse. [/DETAILED]
            [SYNTACTICAL] Format: "valid SMILES string" [/SYNTACTICAL]
            [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N",
            "C1=CC=C(C=C1)C(=O)O" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] The chemical formula in Hill notation (C, H, then alphabetical) [/BRIEF]
            [DETAILED] The chemical formula of the molecule represented by the SMILES string,
            formatted in Hill notation. If the SMILES string is invalid or cannot be parsed,
            it returns an error message. [/DETAILED]
            [EXAMPLES] Examples: "C2H6O" for ethanol, "C6H6" for benzene,
            "C2H5NO" for acetic acid amide, "C7H6O3" for salicylic acid [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If the SMILES string is invalid or cannot be parsed by RDKit. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when the SMILES string cannot be parsed by
            RDKit, indicating that it is not a valid SMILES representation of a molecule. It can
            occur due to syntax errors or unsupported structures in the SMILES string.
            [/ERROR_DETAILS]]
            [ERROR_RECOVERY] If the SMILES string is invalid, try to provide a valid SMILES string.
            Otherwise do not try to solve the error. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function requires RDKit to be installed and properly configured in the environment.
        - It may return "Invalid SMILES string" if the provided SMILES cannot be parsed.
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
def simulate_spectra(smiles: str) -> dict[str, str]:
    r"""[BRIEF] Simulate 1H NMR, 13C NMR, and IR spectra for a given molecule
    using its SMILES string. [/BRIEF]

    [DETAILED] This function simulates the 1H NMR, 13C NMR, and IR spectra for a
    molecule represented by its SMILES string.
    It uses a remote function to perform the simulation, which involves structure analysis,
    neural network prediction of chemical shifts,
    prediction of J-coupling constants, and quantum-mechanical simulation to generate
    realistic multiplet patterns and effects. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it to validate the chemical structure of a proposed molecule by simulating its spectra.
    - When you want to validate some hypothetical molecule against the
    experimental data in the task description.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Analyze the analysis provided in the task and reasons
    for possible candidates molecules using the tools `retrieve_protons_shifts`,
    `retrieve_aromatic_protons_shifts` and `retrieve_carbon_shifts`
    for having more info about the chemical shifts. [/PREREQUISITE]
    2. [CURRENT] Apply this tool with the SMILES string of the proposed
    molecule to simulate its spectra and validate if can be the solution to the task. [/CURRENT]
    3. [FOLLOW_UP] Submit the answer if the simulated spectra is similar to the experimental,
    or go back to step 1 and propose new candidate molecules. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It uses a remote function `simulate_spectra` to perform the simulation.
    - The simulation process involves:
        1. Structure analysis using HOSE code descriptors to identify the
        chemical environment of atoms in the molecule.
        2. Neural network prediction of chemical shifts based on experimental data.
        3. Prediction of J-coupling constants for proton-proton interactions
        to simulate the splitting patterns in NMR spectra.
        4. Quantum-mechanical simulation to generate realistic
        multiplet patterns and effects in the spectra.
    - The function returns a dictionary containing the simulated spectra for 1H NMR, 13C NMR,
    and IR. If some of the spectra are not available, it will return None for those spectra.
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
            [BRIEF] The SMILES representation of the compound to simulate spectra for [/BRIEF]
            [DETAILED] The SMILES string representing the chemical structure of the molecule
            for which the spectra will be simulated. It should be a valid SMILES notation
            that can be processed by the remote function. [/DETAILED]
            [SYNTACTICAL] Format: "valid SMILES string" [/SYNTACTICAL]
            [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "C1=CC=C(C=C1)C(=O)O"
            [/EXAMPLES]

    Returns:
        dict[str, str]:
            [BRIEF] The simulated spectra of the compound [/BRIEF]
            [DETAILED] A dictionary containing the simulated spectra for 1H NMR, 13C NMR,
            and IR. Each key corresponds to a type of spectrum, and the value is a string
            representation of the simulated spectrum. If some spectra are not available,
            the value will be None for those keys. [/DETAILED]
            [EXAMPLES] Examples: {"1H NMR": "simulated_1H_NMR_spectrum", "13C NMR":
            "simulated_13C_NMR_spectrum", "IR": "simulated_IR_spectrum"} [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If a network error occurs during the remote function call. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when there is an error in calling the
            remote function `simulate_spectra`, such as network issues. [/ERROR_DETAILS]
            [ERROR_RECOVERY] This tool is unavailable if the remote function
            cannot be called. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The SMILES string must be valid and represent a chemical structure
        that can be interpreted by the remote function.
        - The remote function may not be able to simulate spectra for all compounds,
        especially if they are complex or not well-defined.
        - The function relies on the availability of the remote service and
        its simulation capabilities, which may change over time.
    [/LIMITATIONS]
    """
    return remote_call(function_name="simulate_spectra", env_name="chemenv")(
        smiles=smiles
    )


@tool
def search_by_smiles(smiles: str, top_k: int = 10) -> list[dict[str, Any]]:
    r"""[BRIEF] Search the NMRShift database for entries matching or chemically
    similar to the given SMILES. [/BRIEF]

    [DETAILED] This function searches the NMRShift database for entries that match
    or are chemically similar to the provided SMILES string.
    It uses a vector database search to find the top `top_k` results based on chemical similarity.
    The function returns a list of matching entries,
    each containing relevant information about the compound. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it to validate your proposed molecule against the NMRShift database.
    - Use it when you have a SMILES string and want to find related compounds
    in the NMRShift database.
    - When you need to retrieve chemical shifts or other NMR-related information
    for a specific compound.
    - Recommended for tasks that require searching for chemical compounds based
    on their SMILES representation, such as NMR spectra analysis,
    chemical structure identification, or database queries. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have a SMILES representation of the molecule you want to search for
    in the NMRShift database, or the SMILES of a structure that you think can be valid.
    [/PREREQUISITE]
    2. [CURRENT] Apply this tool with the SMILES string to search for matching or
    chemically similar entries in the NMRShift database. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved entries to validate your proposed molecule,
    compare chemical shifts, or gather additional information about the compound.
    If you have a robust candidate, use `simulate_spectra`
    to validate your candidate. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - It uses a vector database search to find entries in the NMRShift database
        that match or are chemically similar to the provided SMILES string.
        - The search is performed in the "nmrshiftdb2" collection of the NMRShift database.
        - The search is using chemical embeddings generated by the
        "ibm-research/MoLFormer-XL-both-10pct" model, which allows for chemical
        similarity searches based on the provided SMILES.
        - The function retrieves the top `top_k` results based on chemical
        similarity to the provided SMILES.
        - Each result contains relevant information about the compound, such as its SMILES,
        chemical shifts, and other properties.
        - The function returns a list of matching entries, each represented as a
        dictionary containing the relevant information. [/CONTEXTUAL]

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
            [BRIEF] The SMILES representation of the compound to search for in the
            NMRShift database [/BRIEF]
            [DETAILED] The SMILES string representing the chemical structure of
            the molecule to search for in the NMRShift database.
            It should be a valid SMILES notation that can be processed by the
            vector database search. [/DETAILED]
            [SYNTACTICAL] Format: "valid SMILES string" [/SYNTACTICAL]
            [EXAMPLES] Examples: "CCO", "C1=CC=CC=C1", "C(C(=O)O)N",
            "C1=CC=C(C=C1)C(=O)O" [/EXAMPLES]

        top_k (int, optional):
            [BRIEF] The maximum number of results to return. Defaults to 10 [/BRIEF]
            [DETAILED] The maximum number of search results to return from the NMRShift
            database. It should be a positive integer. [/DETAILED]
            [SYNTACTICAL] Format: "any positive integer (e.g., 10, 20, 50)" [/SYNTACTICAL]
            [EXAMPLES] Examples: 10, 20, 50 [/EXAMPLES]

        Returns:
            list[dict[str, Any]]:
                [BRIEF] A list of dictionaries containing the most relevant entries
                from the NMRShift database [/BRIEF]
                [DETAILED] Each dictionary contains relevant information about the compound,
                such as its SMILES, chemical shifts, and other properties.
                The results are sorted by similarity score in descending order. [/DETAILED]
                [EXAMPLES] Examples: [{"entry_id": "nmrshiftdb2:234", "compound_name":
                "Benzene", "smiles": "c1ccccc1", "spectrum": {"nucleus": "13C",
                "field_strength_mhz": 100.6, "temperature_k": 298, "solvent": "CDCl3",
                "assignment_method": "measured"}, "peaks": [{"atom_id": "a1", "xValue":
                128.5, "multiplicity": "s"}]}, {"entry_id": "nmrshiftdb2:56789",
                "compound_name": "Ethanol", "smiles": "CCO", "spectrum": {"nucleus": "1H",
                "field_strength_mhz": 400.1, "temperature_k": 298, "solvent": "D2O",
                "assignment_method": "measured"}, "peaks": [{"atom_id": "a1", "xValue": 3.65,
                "multiplicity": "q"}, {"atom_id": "a2", "xValue": 1.18, "multiplicity": "t"}]},
                {"entry_id": "nmrshiftdb2:991122", "compound_name": "Cyclohexanone", "smiles":
                "O=C1CCCCC1", "spectrum": {"nucleus": "13C", "field_strength_mhz": 125.7,
                "temperature_k": 298, "solvent": "CDCl3", "assignment_method": "measured"},
                "peaks": [{"atom_id": "a1", "xValue": 211.7, "multiplicity": "s"}, {"atom_id":
                "a2", "xValue": 42.2, "multiplicity": "s"}]}] [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If an error occurs during the search process,
            such as errors in the embedding of the query for example. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when there is an
            error in performing the vector database search, or problems with
            the vector database itself. It can also occur if the `top_k`
            parameter is not a positive integer. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try another tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function requires a valid SMILES string that can be
        processed by the vector database search.
        - The search results are limited to the `top_k` parameter, which defaults to 10.
        - The search is based on chemical embeddings, which may
        not capture all chemical similarities perfectly.
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
def retrieve_protons_shifts() -> list[dict[str, str]]:
    r"""[BRIEF] Retrieve the proton chemical shifts ranges for hydrocarbons. [/BRIEF]

    [DETAILED] This function retrieves the proton chemical shifts
    ranges for various types of hydrocarbons.
    The chemical shifts (delta) are reported in parts per million (ppm)
    relative to tetramethylsilane (TMS) as the reference standard.
    It returns a list of dictionaries, each containing the type of proton
    and its corresponding chemical shift range. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to know the typical chemical shifts of protons
    in hydrocarbons for NMR spectroscopy analysis.
    - When you want to validate the chemical shifts of protons
    in a proposed molecule against known ranges.
    - Recommended for tasks that require understanding the chemical environment of protons
    in hydrocarbons, such as NMR spectra interpretation or chemical structure elucidation.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have a proposed molecule or a set of molecules for which you want to analyze
    the proton chemical shifts. Use the tools `retrieve_aromatic_protons_shifts` and
    `retrieve_carbon_shifts` for having more info about the chemical shifts. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to retrieve the proton chemical shifts ranges for hydrocarbons.
    [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved chemical shifts to compare with the experimental NMR spectra
    of your proposed molecules, or to validate the chemical shifts of protons
    in the proposed molecules. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns a predefined list of dictionaries containing the proton
    types and their corresponding chemical shift ranges in ppm.
    - Each dictionary contains the type of proton (e.g., aldehyde, aromatic, alkene)
    and its chemical shift range relative to TMS.
    - The chemical shifts are based on typical values observed in NMR spectroscopy
    for various types of protons in hydrocarbons.
    - The function does not perform any calculations or database queries; it simply
    returns the predefined list of chemical shifts. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `retrieve_protons_shifts()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        list[dict[str, str]]:
            [BRIEF] A list of dictionaries containing the proton chemical
            shifts ranges for hydrocarbons [/BRIEF]
            [DETAILED] Each dictionary contains the type of proton and its
            corresponding chemical shift range in ppm. The ranges are based on typical
            values observed in NMR spectroscopy for various types of protons in hydrocarbons.
            [/DETAILED]
            [EXAMPLES] Examples: [{"Proton": "Aldehyde", "delta / ppm": "9.5 - 10.5"},
            {"Proton": "Aromatic", "delta / ppm": "6.5 - 8.2"}, ...] [/EXAMPLES]

    [RAISES] Exceptions:
        None
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function returns a predefined list of chemical shifts and does
        not perform any calculations or database queries.
        - The chemical shifts are based on typical values and may not be
        applicable to all compounds or conditions.
        - The ranges provided are approximate and may vary depending on the
        specific molecular environment and experimental conditions.
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
def retrieve_aromatic_protons_shifts() -> list[dict[str, str]]:
    r"""[BRIEF] Retrieve the proton shifts ranges for aromatic hydrocarbons. [/BRIEF]

    [DETAILED] This function retrieves the proton chemical shifts ranges for aromatic hydrocarbons.
    The values represent chemical shift changes (in ppm) caused by substituents on a benzene ring.
    The values show how much a substituent shifts the resonance of protons at ortho,
    meta, and para positions
    relative to unsubstituted benzene. Positive values indicate downfield shifts (deshielding),
    while negative values indicate upfield shifts (shielding).
    All shifts are relative to tetramethylsilane (TMS) as the reference standard. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to know the typical chemical shifts of protons in aromatic hydrocarbons
    for NMR spectroscopy analysis.
    - When you want to validate the chemical shifts of protons in a proposed aromatic
    molecule against known ranges.
    - Recommended for tasks that require understanding the chemical environment of protons
    in aromatic rings, such as NMR spectra interpretation or chemical structure elucidation.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have a proposed aromatic molecule or a set of molecules
    for which you want to analyze the proton chemical shifts.
    Use the tools `retrieve_protons_shifts` and `retrieve_carbon_shifts`
    for having more info about the chemical shifts. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to retrieve the proton chemical shifts ranges
    for aromatic hydrocarbons. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved chemical shifts to compare with the experimental
    NMR spectra of your proposed aromatic molecules, or to validate the chemical shifts
    of protons in the proposed aromatic molecules. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns a predefined list of dictionaries containing the substituent effects
    on proton chemical shifts in aromatic rings.
    - Each dictionary contains the substituent name and its corresponding chemical
    shift changes (in ppm) for ortho, meta, and para positions.
    - The chemical shifts are based on typical values observed in NMR spectroscopy
    for various substituents on aromatic rings.
    - The function does not perform any calculations or database queries; it simply
    returns the predefined list of chemical shifts. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `retrieve_aromatic_protons_shifts()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        list[dict[str, str]]:
            [BRIEF] A list of dictionaries containing the substituent effects on proton
            chemical shifts in aromatic rings [/BRIEF]
            [DETAILED] Each dictionary contains the substituent name and its corresponding
            chemical shift changes (in ppm) for ortho, meta, and para positions.
            The shifts are based on typical values observed in NMR spectroscopy for
            various substituents on aromatic rings. [/DETAILED]
            [EXAMPLES] Examples: [{"Substituent": "NO2", "Ortho": 0.95, "Meta": 0.17,
            "Para": 0.33}, {"Substituent": "CHO", "Ortho": 0.58, "Meta": 0.21,
            "Para": 0.27}, ...] [/EXAMPLES]

    [RAISES] Exceptions:
        None
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function returns a predefined list of chemical shifts
        and does not perform any calculations or database queries.
        - The chemical shifts are based on typical values and may
        not be applicable to all aromatic compounds or conditions.
        - The shifts provided are approximate and may vary depending
        on the specific molecular environment and experimental conditions.
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
def retrieve_carbon_shifts() -> list[dict[str, str]]:
    r"""[BRIEF] Retrieve the carbon chemical shifts ranges for various
    functional groups in organic compounds. [/BRIEF]

    [DETAILED] This function retrieves the carbon chemical shifts ranges
    for various functional groups in organic compounds.
    The chemical shifts (deltas) are reported in parts per million (ppm)
    relative to tetramethylsilane (TMS) as the reference standard.
    These values can be used to interpret 13C NMR spectra and identify carbon
    environments in unknown compounds.
    It returns a list of dictionaries, each containing the functional group
    and its corresponding chemical shift range. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to know the typical chemical shifts of carbon atoms
    in various functional groups for NMR spectroscopy analysis.
    - When you want to validate the chemical shifts of carbon atoms in a
    proposed molecule against known ranges.
    - Recommended for tasks that require understanding the chemical environment
    of carbon atoms in organic compounds, such as NMR spectra interpretation or
    chemical structure elucidation. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have a proposed molecule or a set of molecules for which
    you want to analyze the carbon chemical shifts.
    Use the tools `retrieve_protons_shifts` and `retrieve_aromatic_protons_shifts`
    for having more info about the chemical shifts. [/PREREQUISITE]
    2. [CURRENT] Apply this tool to retrieve the carbon chemical shifts ranges for
    various functional groups in organic compounds. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved chemical shifts to compare with the experimental
    NMR spectra of your proposed molecules, or to validate the chemical shifts of
    carbon atoms in the proposed molecules. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns a predefined list of dictionaries containing the carbon chemical
    shifts ranges for different functional groups.
    - Each dictionary contains the functional group and its corresponding chemical
    shift range in ppm.
    - The chemical shifts are based on typical values observed in NMR spectroscopy
    for various functional groups in organic compounds.
    - The function does not perform any calculations or database queries; it simply
    returns the predefined list of chemical shifts. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `retrieve_carbon_shifts()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        list[dict[str, str]]:
            [BRIEF] A list of dictionaries containing the carbon chemical shifts ranges
            for various functional groups in organic compounds [/BRIEF]
            [DETAILED] Each dictionary contains the functional group and its corresponding
            chemical shift range in ppm. The ranges are based on typical values observed
            in NMR spectroscopy for various functional groups in organic compounds. [/DETAILED]
            [EXAMPLES] Examples: [{"Group": "CH3-", "Shift (ppm)": "10-30 ppm"},
            {"Group": "R3C-, R₂CH, RCH₂", "Shift (ppm)": "25-50 ppm"}, ...] [/EXAMPLES]

    [RAISES] Exceptions:
        None
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The function returns a predefined list of chemical shifts and
        does not perform any calculations or database queries.
        - The chemical shifts are based on typical values and may not be
        applicable to all compounds or conditions.
        - The ranges provided are approximate and may vary depending
        on the specific molecular environment and experimental conditions.
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


def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)
    fs_manager = FSManager("file", base_path=BASE_WORK_DIR)  # noqa: F841
    return {
        # "list_files": ListFilesTool(fs_manager),
        # "read_file": ReadFileTool(fs_manager),
        # "write_file": WriteFileTool(fs_manager),
        # "file_info": FileInfoTool(fs_manager),
        # "cat_files": CatFilesTool(fs_manager),
        # "copy_file": CopyFileTool(fs_manager),
        # "online_search": online_search,
        # "relevant_pubchem_sections": relevant_pubchem_sections,
        "get_formula_from_smiles": get_formula_from_smiles,
        "simulate_spectra": simulate_spectra,
        "retrieve_protons_shifts": retrieve_protons_shifts,
        "retrieve_aromatic_protons_shifts": retrieve_aromatic_protons_shifts,
        "retrieve_carbon_shifts": retrieve_carbon_shifts,
        "search_by_smiles": search_by_smiles,
    }
