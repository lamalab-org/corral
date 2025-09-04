import os
from typing import Any

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from corral.utils.tool_helpers import embed_text, make_api_request


def make_brave_search_request(query: str, api_key: str) -> list[dict[str, Any]]:
    """
    Make a request to the Brave Search API using the general API request function.

    Args:
        query (str): The search query string
        api_key (str): The Brave Search API key

    Returns:
        list[dict[str, Any]]: List of search results

    Raises:
        requests.exceptions.RequestException: If the request fails after retries
    """
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}

    params = {"q": query}

    logger.info(f"Making Brave Search API request for query: '{query}'")

    data = make_api_request(
        url="https://api.search.brave.com/res/v1/web/search",
        method="GET",
        headers=headers,
        params=params,
        verbose=True,
    )

    return [
        {
            "title": web_result.get("title", ""),
            "snippet": web_result.get("description", ""),
            "link": web_result.get("url", ""),
        }
        for web_result in data.get("web", {}).get("results", [])
    ]


def web_search(query: str, num_results: int = 5) -> list[dict[str, Any]]:
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
    logger.info(
        f"Starting web search for query: '{query}' with parameters: num_results={num_results}"
    )

    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        logger.error("BRAVE_SEARCH_API_KEY environment variable is not set")
        raise ValueError(
            "BRAVE_SEARCH_API_KEY environment variable is required but not set"
        )

    try:
        logger.info("Making Brave Search API request")
        initial_results = make_brave_search_request(query, api_key)

        if not initial_results:
            logger.warning("No results returned from Brave Search API")
            return []

        logger.info(
            f"Retrieved {len(initial_results)} initial results from Brave Search"
        )

        logger.info("Generating query embedding")
        query_embedding = np.array(embed_text(chunks=[query])[0]).reshape(1, -1)
        result_texts = [f"{r['title']}: {r['snippet']}" for r in initial_results]
        logger.info("Generating result embeddings")
        result_embeddings = np.array(embed_text(chunks=result_texts))

        logger.info("Calculating similarity scores")
        similarity_scores = sklearn_cosine_similarity(
            query_embedding, result_embeddings
        ).flatten()

        logger.info("Processing results with similarity scores")
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

        logger.info("Sorting results by similarity score")
        sorted_results = sorted(
            results_with_scores, key=lambda x: x["similarity_score"], reverse=True
        )

        final_results = sorted_results[:num_results]
        logger.info(f"Returning {len(final_results)} final results")
        return final_results

    except Exception as e:
        logger.error(f"Error while performing search: {e!s}", exc_info=True)
        return []
