from typing import Any
from urllib.parse import quote

import requests
from litellm import embedding
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (requests.exceptions.RequestException, requests.exceptions.HTTPError)
    ),
)
def make_api_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Make an API request with retry capabilities.

    Args:
        url (str): The API endpoint URL
        method (str, optional): HTTP method (GET, POST, PUT, etc.). Defaults to "GET".
        headers (dict[str, str], optional): Request headers. Defaults to None.
        params (dict[str, Any], optional): URL parameters. Defaults to None.
        json_data (dict[str, Any], optional): JSON data for POST/PUT requests. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        dict[str, Any]: JSON response from the API

    Raises:
        requests.exceptions.RequestException: If the request fails after retries
    """
    method = method.upper()
    url = quote(url, safe=":/?&=")
    logger.info(f"Making {method} request to {url}")

    if verbose:
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Params: {params}")
        if json_data:
            logger.debug(f"JSON data: {json_data}")

    response = requests.request(
        method=method, url=url, headers=headers, params=params, json=json_data
    )

    if verbose:
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response content: {response.text[:500]}...")

    response.raise_for_status()
    return response.json()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
def embed_text(chunks: list, model: str, chemical=False) -> list[list[float]]:
    """
    Embed a list of text chunks using the specified model with automatic retries.
    If the list is large (>2048 chunks), it will process them in smaller batches.
    When chemical=True, uses MoLFormer for chemical embeddings.

    Args:
        chunks (list): List of text chunks to embed
        model (str): Model to use for embeddings
        chemical (bool, optional): Whether to use chemical embedding model. Default is False.

    Returns:
        list[list[float]]: List of embeddings for each chunk

    Raises:
        ValueError: If chunks is not a non-empty list of strings
        RuntimeError: If embeddings fail after multiple retries
    """
    if (
        not chunks
        or not isinstance(chunks, list)
        or not all(isinstance(chunk, str) for chunk in chunks)
    ):
        logger.error("Invalid input: chunks must be a non-empty list of strings")
        raise ValueError("Input must be a non-empty list of strings")

    logger.info(
        f"Embedding {len(chunks)} {'chemical' if chemical else 'text'} chunks using model: {model}"
    )

    BATCH_SIZE = 2048  # This is a configuration from LiteLLM:
    # https://docs.litellm.ai/docs/embedding/supported_embedding#required-fields

    # Process in batches if the input is large
    if len(chunks) > BATCH_SIZE:
        logger.info(f"Input size exceeds {BATCH_SIZE} chunks, processing in batches")
        all_embeddings = []

        # Process chunks in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            logger.info("Processing batched chunks")

            try:
                batch_embeddings = embed_text(batch, model=model, chemical=chemical)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error in batch {i//BATCH_SIZE + 1}: {e!s}")
                raise

        logger.info(
            f"Successfully generated {len(all_embeddings)} embeddings across all batches"
        )
        return all_embeddings

    # For chemical embeddings, use MoLFormer
    if chemical:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            logger.info("Using MoLFormer model for chemical embeddings")

            # Load model & tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            molformer_model = AutoModel.from_pretrained(
                model, deterministic_eval=True, trust_remote_code=True
            )

            # Tokenization
            inputs = tokenizer(
                chunks, padding=True, truncation=True, return_tensors="pt"
            )

            # Inference
            with torch.no_grad():
                outputs = molformer_model(**inputs)

            # Extract embeddings
            embeddings = outputs.pooler_output.tolist()  # Convert to list format

            logger.info(f"Successfully generated {len(embeddings)} chemical embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating chemical embeddings: {e!s}", exc_info=True)
            raise RuntimeError(f"Failed to generate chemical embeddings: {e!s}") from e

    # For text embeddings, use litellm as before
    try:
        result_embeddings = embedding(
            model=model,
            input=chunks,
        )
        logger.info(
            f"Successfully generated {len(result_embeddings['data'])} embeddings"
        )
        return [item["embedding"] for item in result_embeddings["data"]]
    except Exception as e:
        logger.error(f"Error generating embeddings: {e!s}", exc_info=True)
        raise RuntimeError(f"Failed to generate embeddings: {e!s}") from e
