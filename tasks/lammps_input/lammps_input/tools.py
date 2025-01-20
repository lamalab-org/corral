from __future__ import annotations

import os
from pathlib import Path

from ase import Atoms
from dotenv import load_dotenv
from langchain_community.tools.brave_search.tool import BraveSearch
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

from corral.utils import tool

load_dotenv("../.env")


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a summary of the topic.

    Args:
        query: The search term or topic to look up on Wikipedia

    Returns:
        A string containing the Wikipedia article summary
    """
    wikipedia = WikipediaAPIWrapper()
    return wikipedia.run(query)


@tool
def brave_search(query: str) -> list[dict]:
    """Perform a web search using Brave Search API.

    Args:
        query: The search query string

    Returns:
        A list of dictionaries containing search results with titles and snippets

    Raises:
        ValueError: If BRAVE_SEARCH_API_KEY environment variable is not set
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError(
            "BRAVE_SEARCH_API_KEY environment variable is required but not set"
        )

    brave_search_tool = BraveSearch.from_api_key(api_key=api_key)

    return brave_search_tool._run(query)


@tool
def pickle_to_lammps(input_path: str, output_path: str) -> str:
    """Read ASE Atoms from pickle file and write to LAMMPS data format.

    Args:
        input_path: Path to input pickle file containing ASE Atoms object
        output_path: Path to write LAMMPS data file

    Returns:
        Status message
    """
    import pickle

    from ase.io import write

    with Path(input_path).open("rb") as f:
        atoms = pickle.load(f)
    write(output_path, atoms, format="lammps-data")
    return f"Converted {input_path} to {output_path}"


@tool
def ase_lammps(
    output_path: str, elements: list, positions: list, cell: list, pbc: list
) -> str:
    """Create ASE Atoms from parameters and write to LAMMPS data format.

    Args:
        output_path: Path to write LAMMPS data file
        elements: List of element symbols
        positions: Array of atomic positions (N x 3)
        cell: Unit cell parameters (3 x 3)
        pbc: Periodic boundary conditions

    Returns:
        Status message
    """
    from ase.io import write

    atoms = Atoms(symbols=elements, positions=positions, cell=cell, pbc=pbc)
    write(output_path, atoms, format="lammps-data")
    return f"Created LAMMPS data file at {output_path}"


@tool
def read_lammps_to_string(input_path: str) -> str:
    """Read LAMMPS data file and return contents as string.

    Args:
        input_path: Path to LAMMPS data file

    Returns:
        File contents as string
    """
    with Path(input_path).open() as f:
        return f.read()
