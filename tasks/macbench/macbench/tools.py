from __future__ import annotations

import os

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
def wolfram_alpha(query: str) -> str:
    """Perform scientific calculations and queries using Wolfram Alpha.

    Args:
        query: Mathematical expression or scientific query (e.g., "What is 2x+5 = -3x + 7?")
    Returns:
        The result of the calculation or query as a string
    """
    wolfram = WolframAlphaAPIWrapper()
    return wolfram.run(query)
