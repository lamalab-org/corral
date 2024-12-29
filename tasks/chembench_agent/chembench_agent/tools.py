from __future__ import annotations

import os
from time import sleep
from typing import Optional

import pubchempy as pcp
import requests
from dotenv import load_dotenv
from langchain_community.tools.brave_search.tool import BraveSearch
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper

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
    """
    brave_search_tool = BraveSearch.from_api_key(
        api_key=os.getenv("BRAVE_SEARCH_API_KEY")
    )
    return brave_search_tool._run(query)


@tool
def wolfram_alpha(query: str) -> str:
    """Perform scientific calculations and queries using Wolfram Alpha.

    Args:
        query: Mathematical expression or scientific query (e.g., "What is 2x+5 = -3x + 7?")
    Returns:
        The result of the calculation or query as a string
    """
    wolfram = WolframAlphaAPIWrapper()
    return wolfram.run(query)


@tool
def smiles_to_iupac_name(smiles: str) -> Optional[str]:
    """Convert SMILES chemical notation to IUPAC name.

    Args:
        smiles: SMILES representation of a chemical compound
    Returns:
        IUPAC name of the compound or None if conversion fails
    """
    try:
        sleep(0.1)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name"
        response = requests.get(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        name = response.text
        if "html" in name:
            return None
        return name
    except Exception:
        try:
            compound = pcp.get_compounds(smiles, "smiles")
            return compound[0].iupac_name
        except Exception:
            return None


@tool
def iupac_to_smiles_name(iupac_name: str) -> Optional[str]:
    """Convert IUPAC chemical name to SMILES notation.

    Args:
        iupac_name: IUPAC name of a chemical compound
    Returns:
        SMILES representation of the compound or None if conversion fails
    """
    try:
        sleep(0.1)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{iupac_name}/smiles"
        response = requests.get(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        name = response.text
        if "html" in name:
            return None
        return name
    except Exception:
        try:
            compound = pcp.get_compounds(iupac_name, "name")
            return compound[0].isomeric_smiles
        except Exception:
            return None
