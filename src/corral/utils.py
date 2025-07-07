import gc
import inspect
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints
from urllib.parse import quote

import chromadb
import modal
import numpy as np
import requests
import tiktoken
from litellm import embedding
from loguru import logger
from modal import App, Image, Mount, Secret, Volume
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from corral.base import ModalTool, Tool, ToolArgument

MODAL_TOOL_REGISTRY = {}


def format_type_annotation(annotation):
    """Formats type annotations to readable strings."""

    # Handle basic types
    if isinstance(annotation, type):
        return "None" if annotation is type(None) else annotation.__name__
    # Handle new-style union (str | int)
    if isinstance(annotation, type | type(None)):
        return annotation.__name__

    # Handle new-style unions using '|'
    if get_origin(annotation) is Union:
        args = [format_type_annotation(arg) for arg in get_args(annotation)]
        return " | ".join(args).replace("NoneType", "None")

    # Handle old-style unions (Union[str, int])
    if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
        args = [format_type_annotation(arg) for arg in annotation.__args__]
        return " | ".join(args).replace("NoneType", "None")

    # Handle Optional (which is Union[T, None])
    if annotation is Optional:
        return f"{format_type_annotation(annotation.__args__[0])} | None"

    # Handle generic types like list, dict, etc.
    if hasattr(annotation, "__origin__"):
        origin = format_type_annotation(annotation.__origin__)
        args = ", ".join(format_type_annotation(arg) for arg in annotation.__args__)
        return f"{origin}[{args}]"

    # Fallback to string representation for unknown types
    return str(annotation)


keywords = [
    "BRIEF",
    "DETAILED",
    "PROCEDURAL",
    "WORKFLOW_INTEGRATION",
    "CONTEXTUAL",
    "SYNTACTICAL",
    "RAISES",
    "LIMITATIONS",
    "EXAMPLES",
]


def regex_parsing_docstring_sections(doc_without_sections: str):
    """Extracts sections from a docstring based on predefined keywords."""
    keyword_pattern = "|".join(re.escape(keyword) for keyword in keywords)
    keyword_regex = re.compile(rf"\[({keyword_pattern})\](.*?)\[/\1\]", re.DOTALL)
    matches = keyword_regex.findall(doc_without_sections)
    return [(match[0], match[1].strip()) for match in matches]


def parse_arguments(args_content: str) -> list[ToolArgument]:
    """Parse the Args section of a docstring to extract arguments."""
    arguments = []
    lines = args_content.strip().splitlines()
    current_arg = None
    arg_pattern = re.compile(r"^\s*\w+\s*\([^)]+\)\s*:")
    for _i, line in enumerate(lines):
        if current_arg is None or arg_pattern.match(line):
            if current_arg is not None:
                arguments.append(current_arg)
            current_arg = line.strip()
        else:
            current_arg += " " + line.strip()

    if current_arg is not None:
        arguments.append(current_arg)
    return arguments


def parse_complex_docstring(doc: str) -> tuple[dict[str, str], list[ToolArgument]]:
    """Parse a complex docstring with multiple sections including Args, Returns, and RAISES."""
    sections = {}
    # Extract Args section (between "Args:" and "Returns:")
    args_match = re.search(r"Args:(.*?)(?=Returns:|$)", doc, re.DOTALL)
    if args_match:
        args_content = args_match.group(1).strip()
        arguments = parse_arguments(args_content)

    # Extract Returns section (between "Returns:" and "[RAISES]")
    returns_match = re.search(r"Returns:(.*?)(?=\[RAISES\]|$)", doc, re.DOTALL)
    if returns_match:
        returns_content = returns_match.group(1).strip()
        sections["RETURNS"] = returns_content if returns_match else ""

    # Extract RAISES section (between "[RAISES]" and "[/RAISES]")
    raises_match = re.search(r"\[RAISES\](.*?)\[/RAISES\]", doc, re.DOTALL)
    if raises_match:
        raises_content = raises_match.group(1).strip()
        sections["RAISES"] = raises_content if raises_match else ""

    # Remove these sections from the original document
    doc_without_sections = doc
    if args_match:
        doc_without_sections = re.sub(
            r"Args:.*?(?=Returns:|$)", "", doc_without_sections, flags=re.DOTALL
        )
    if returns_match:
        doc_without_sections = re.sub(
            r"Returns:.*?(?=\[RAISES\]|$)", "", doc_without_sections, flags=re.DOTALL
        )
    if raises_match:
        doc_without_sections = re.sub(
            r"\[RAISES\].*?\[/RAISES\]", "", doc_without_sections, flags=re.DOTALL
        )

    # Now extract the remaining keyword sections
    matches = regex_parsing_docstring_sections(doc_without_sections)

    for match in matches:
        sections[match[0]] = match[1]

    return sections, arguments


def parse_docstring(func: Callable) -> tuple[dict[str, str], list[ToolArgument]]:
    """Parse function docstring to get description and arguments.

    This function extracts the description and arguments from a function's docstring.
    It expects a docstring with a description section and an Args section.

    Args:
        func (Callable): The function to parse docstring from

    Returns:
        tuple[dict[str, str], list[ToolArgument]]: (sections, arguments) where sections is a dict
               mapping section names to content and arguments is a list of ToolArgument objects

    Raises:
        ValueError: If the docstring is missing or doesn't have an Args section
    """
    doc = inspect.getdoc(func)
    if not doc:
        raise ValueError(f"Function {func.__name__} must have a docstring")

    if "[BRIEF]" in str(doc):
        sections, args_lines = parse_complex_docstring(doc)

    else:
        # Split docstring into sections
        doc_sections = doc.split("\n\n")
        description = doc_sections[0].strip()
        sections = {"BRIEF": description}

        # Find Args section
        args_section = None
        for section in doc_sections:
            if section.strip().startswith("Args:"):
                args_section = section.strip()
                break

        if not args_section:
            raise ValueError("Docstring must have an 'Args:' section")

        # Parse arguments section, skip the "Args:" line
        args_lines = [
            line.strip() for line in args_section.splitlines()[1:] if line.strip()
        ]
    arguments = []

    # Get type hints from function
    type_hints = get_type_hints(func)

    # Parse each argument line
    for line in args_lines:
        if ":" not in line:
            continue
        arg_name, arg_desc = line.split(":", 1)
        arg_name = arg_name.strip()
        arg_desc = arg_desc.strip()

        # Extract bare parameter name without the type annotation in parentheses
        # This handles formats like "query (str): Description"
        if "(" in arg_name and ")" in arg_name:
            arg_name = arg_name.split("(")[0].strip()

        # Parse choices if specified in format (choices: [val1, val2, ...])
        choices = None
        if "(choices:" in arg_desc:
            desc_parts = arg_desc.split("(choices:", 1)
            arg_desc = desc_parts[0].strip()
            choices_str = desc_parts[1].split(")", 1)[0].strip()
            try:
                choices = eval(choices_str)  # Convert string representation to list
            except ValueError:
                raise ValueError(
                    f"Invalid choices format for argument {arg_name}"
                ) from None

        # Get type from type hints
        if arg_name not in type_hints:
            continue  # Skip non-argument sections like Returns

        arg_annotation = type_hints[arg_name]
        arg_type = format_type_annotation(arg_annotation)

        # Check if argument has default value
        signature = inspect.signature(func)
        param = signature.parameters.get(arg_name)
        has_default = param.default != inspect.Parameter.empty if param else False
        default_value = param.default if has_default else None

        arguments.append(
            ToolArgument(
                name=arg_name,
                type=arg_type,
                description=arg_desc,
                required=not has_default,
                default=default_value,
                choices=choices,
            )
        )

    return sections, arguments


def tool(func: Callable) -> Tool:
    """
    Decorator to convert a function into a Tool. The decorated function must have:
    1. A complete docstring with description and Args section.
    2. Type hints for all parameters.
    3. A return type hint.

    The docstring must follow this format:
    ```
                Brief description of what the tool does.

                Args:
                    param1: Description of first parameter (choices: ["optional", "list", "of", "choices"])
                    param2: Description of second parameter
                    ...

                Returns:
                    Description of what the function returns
    ```

    Args:
        func (Callable): The function to convert into a tool

    Returns:
        Tool: A Tool instance wrapping the function

    Raises:
        ValueError: If the function lacks proper docstring, type hints, or has invalid format
        TypeError: If the function signature is incompatible with Tool requirements
    """
    # Validate function has a docstring
    if not func.__doc__:
        raise ValueError(
            f"Function {func.__name__} must have a docstring describing its purpose and arguments. "
            "See the decorator documentation for the required format."
        )

    # Validate function has type hints
    type_hints = get_type_hints(func)
    if not type_hints:
        raise TypeError(
            f"Function {func.__name__} must have type hints for all parameters and return type. "
            "Example: def func(param1: str, param2: int) -> str"
        )

    # Validate return type is specified
    if "return" not in type_hints:
        raise TypeError(
            f"Function {func.__name__} must specify a return type hint. "
            "Example: def func(param: str) -> str"
        )

    # Validate docstring format
    doc = inspect.getdoc(func)
    if doc is None or "Args:" not in doc:
        raise ValueError(
            f"Function {func.__name__}'s docstring must have an 'Args:' section. "
            "See the decorator documentation for the required format."
        )

    try:
        sections, arguments = parse_docstring(func)
    except Exception as e:
        raise ValueError(
            f"Error parsing docstring for function {func.__name__}: {e!s}. "
            "Please ensure it follows the required format shown in the decorator documentation."
        ) from e

    # Validate all parameters have documentation
    signature_params = set(inspect.signature(func).parameters.keys())
    documented_params = {arg.name for arg in arguments}
    if missing_docs := signature_params - documented_params:
        raise ValueError(
            f"Missing documentation for parameters: {', '.join(missing_docs)}. "
            "All parameters must be documented in the Args section of the docstring."
        )

    class FunctionTool(Tool):
        def __init__(self):
            super().__init__(
                name=func.__name__, description=sections, arguments=arguments
            )

        def execute(self, **kwargs):
            return str(func(**kwargs))

    return FunctionTool()


def create_modal_function(func: Callable, app: App, **modal_kwargs) -> Callable:
    """Create a Modal function with the given configuration."""
    modal_kwargs = {k: v for k, v in modal_kwargs.items() if v is not None}
    return app.function(**modal_kwargs)(func)


def modal_tool(
    app: App,
    image: Image | None = None,
    secrets: Sequence[Secret] | None = None,
    mounts: Sequence[Mount] | None = None,
    volumes: dict[str, Volume] | None = None,
    memory: int | None = None,
    timeout: int | None = None,
    cpu: float | None = None,
    retries: int | None = None,
    gpu: str | None = None,
    keep_warm: int | None = None,
    block_network: bool = False,
    **kwargs,
):
    """
    Decorator that returns a ModalTool instance (which inherits from Tool)
    that can be used both as a Tool and as a Modal function.
    """
    if app is None:
        raise ValueError(
            "The 'app' argument is required. This is the Modal App instance."
        )

    modal_kwargs = {
        "image": image,
        "secrets": secrets,
        "mounts": mounts,
        "volumes": volumes,
        "memory": memory,
        "timeout": timeout,
        "cpu": cpu,
        "retries": retries,
        "gpu": gpu,
        "keep_warm": keep_warm,
        "block_network": block_network,
        **kwargs,
    }

    def decorator(func: Callable):
        modal_func = create_modal_function(func, app, **modal_kwargs)
        sections, arguments = parse_docstring(func)

        tool_instance = ModalTool(
            modal_func=modal_func,
            name=func.__name__,
            description=sections,
            arguments=arguments,
        )

        # Register the tool for later use
        MODAL_TOOL_REGISTRY[func.__name__] = tool_instance

        # Return the modal function
        return modal_func

    return decorator


def remote_call(function_name: str, env_name: str = "chemenv"):
    """
    Decorator to call a function in a remote environment.
    This decorator is used to call a function in a remote environment
    using the Modal library.

    Args:
        function_name (str): The name of the function to call
        env_name (str): The name of the environment to use

    Returns:
        Callable: A wrapper function that calls the remote function
    """

    def wrapper(**kwargs) -> str:
        remote = modal.Function.from_name(env_name, function_name)
        return remote.remote(**kwargs)

    return wrapper


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


def vector_database_search(
    query: str,
    collection_name: str = "default_collection",
    path: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve the top 5 most similar instructions from a vector database based on the query.

    Args:
        query (str): The search query to find similar instructions
        collection_name (str, optional): The name of the collection in the vector database. Default is "default_collection".
        path (str, optional): The path to the vector database directory. Defaults to None, which uses "vector_db" in the current directory.
        top_k (int, optional): The number of similar instructions to retrieve. Default is 5.

    Returns:
        list[dict]: A list of dictionaries containing the top 5 most similar instructions with their content and metadata

    Raises:
        RuntimeError: If the specified collection doesn't exist
    """
    query_embedding = embed_text(
        chunks=[query],
    )[0]

    persist_directory = Path(Path.cwd()) / "vector_db" if path is None else Path(path)

    if not persist_directory.exists():
        raise RuntimeError(
            f"Vector database directory '{persist_directory}' does not exist. Please create the vector database first."
        )

    try:
        client = chromadb.PersistentClient(path=str(persist_directory))

        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{collection_name}' does not exist: {e!s}"
            ) from e

        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        formatted_results = []
        for _i, (doc, doc_id, distance) in enumerate(
            zip(
                results["documents"][0],
                results["ids"][0],
                results["distances"][0],
                strict=False,
            )
        ):
            similarity_score = 1 - distance

            formatted_results.append(
                {
                    "content": doc,
                    "metadata": {"id": doc_id},
                    "similarity_score": similarity_score,
                }
            )

        return formatted_results

    except Exception as e:
        raise RuntimeError(f"Error querying vector database: {e!s}") from e


def _tokenize_and_split_chunks(
    chunks: list[str], chunk_size: int, batch_size: int = 2
) -> list[str]:
    """
    Tokenize each chunk and split any chunks that exceed the specified token limit.
    Process chunks in batches to reduce memory usage.
    The chunking is done with a 20% chunk size overlap to ensure no information is lost.
    In addition the chunking is done naively, i.e. it does not look for sentence boundaries.

    Args:
        chunks (list[str]): List of text chunks to process
        chunk_size (int): Maximum number of tokens per chunk
        batch_size (batch_size): Number of chunks to process in each batch. Default is 2.

    Returns:
        list[str]: List of processed chunks, each within the specified token limit

    Raises:
        ValueError: If chunk_size is less than 1
    """

    logger.info(
        f"Tokenizing and splitting {len(chunks)} chunks with max size {chunk_size} tokens"
    )
    chunks = [str(chunk) for chunk in chunks]

    encoding = tiktoken.get_encoding("o200k_base")
    target_size = int(0.8 * chunk_size)
    processed_chunks = []

    try:
        for batch_idx in range(0, len(chunks), batch_size):
            gc.collect()

            batch = chunks[batch_idx : batch_idx + batch_size]
            logger.debug(
                f"Processing batch {batch_idx // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}"
            )

            batch_results = []
            for chunk_idx, chunk in enumerate(batch):
                tokens = encoding.encode(chunk)

                if len(tokens) <= chunk_size:
                    batch_results.append(chunk)
                    logger.debug(
                        f"Chunk {batch_idx + chunk_idx + 1} is within token limit ({len(tokens)}/{chunk_size})"
                    )
                    del tokens
                    continue

                logger.debug(
                    f"Chunk {batch_idx + chunk_idx + 1} exceeds token limit ({len(tokens)}/{chunk_size}), splitting..."
                )

                start_idx = 0
                sub_chunk_count = 0

                while start_idx < len(tokens):
                    end_idx = min(start_idx + target_size, len(tokens))

                    logger.debug(
                        f"Sub-chunk {sub_chunk_count + 1}: Processing from token {start_idx} to {end_idx} ({end_idx - start_idx} tokens)"
                    )

                    sub_chunk = encoding.decode(tokens[start_idx:end_idx])
                    batch_results.append(sub_chunk)
                    sub_chunk_count += 1

                    if end_idx >= len(tokens):
                        break

                    old_start_idx = start_idx
                    overlap = min(100, end_idx - start_idx)  # Fixed small overlap
                    start_idx = end_idx - overlap

                    logger.debug(
                        f"Sub-chunk {sub_chunk_count}: Added {len(sub_chunk)} chars, moved start_idx from {old_start_idx} to {start_idx} (overlap: {overlap} tokens)"
                    )

                    if start_idx <= old_start_idx:
                        logger.error(
                            f"Loop not progressing! start_idx={start_idx}, old_start_idx={old_start_idx}, end_idx={end_idx}"
                        )
                        logger.error(
                            "Breaking infinite loop, please check the algorithm logic"
                        )
                        break

                logger.debug(
                    f"Split chunk {batch_idx + chunk_idx + 1} into {sub_chunk_count} smaller chunks"
                )
                del tokens
            processed_chunks.extend(batch_results)
            del batch_results
            del batch
            gc.collect()
    except Exception as e:
        logger.error(f"Error during tokenization and splitting: {e}")
        raise
    finally:
        gc.collect()

    logger.info(
        f"Completed tokenization and splitting: {len(chunks)} input chunks → {len(processed_chunks)} output chunks"
    )
    return processed_chunks


def create_vector_database(
    chunks: list[str],
    collection_name: str = "default_collection",
    path: str | None = None,
    chunk_size: int = 8192,
    update_mode: str = "recreate",
) -> str:
    """Create or update a vector database from text instructions.

    Args:
        chunks (list[str]): The text instructions to be stored in the vector database
        collection_name (str, optional): The name of the collection in the vector database. Default is "default_collection".
        path (str, optional): Path to store the vector database. Default is "vector_db" in the current directory.
        chunk_size (int, optional): Maximum number of tokens per chunk. Default is 8192.
        update_mode (str, optional): How to handle existing collections: "recreate" deletes and recreates the collection,
                    "append" adds new chunks to existing collection, "upsert" updates existing chunks and
                    adds new ones. Default is "recreate".

    Returns:
        str: A message indicating the success of the operation

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set or update_mode is invalid
    """
    logger.info(
        f"Creating vector database with {len(chunks)} chunks in collection '{collection_name}'"
    )

    if update_mode not in ["recreate", "append", "upsert"]:
        raise ValueError("update_mode must be one of: 'recreate', 'append', 'upsert'")

    persist_directory = Path(Path.cwd()) / "vector_db" if path is None else Path(path)

    persist_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using persist directory: {persist_directory}")

    processed_chunks = _tokenize_and_split_chunks(chunks, chunk_size)
    logger.info(
        f"Processed {len(chunks)} chunks into {len(processed_chunks)} chunks after tokenization and splitting"
    )

    client = chromadb.PersistentClient(path=str(persist_directory))

    try:
        collection_list = client.list_collections()
        collection_exists = collection_name in collection_list

        logger.debug(f"Available collections: {collection_list}")
        logger.debug(f"Collection '{collection_name}' exists: {collection_exists}")

        if update_mode == "recreate" and collection_exists:
            logger.info(
                f"Collection '{collection_name}' already exists, deleting before recreation"
            )
            client.delete_collection(name=collection_name)
            collection = client.create_collection(name=collection_name)
            logger.info(f"Created collection '{collection_name}'")
        elif not collection_exists:
            logger.info(
                f"Collection '{collection_name}' does not exist, creating new collection"
            )
            collection = client.create_collection(name=collection_name)
            logger.info(f"Created new collection '{collection_name}'")
        else:
            logger.info(
                f"Using existing collection '{collection_name}' for {update_mode} operation"
            )
            collection = client.get_collection(name=collection_name)

        logger.info(f"Generating embeddings for {len(processed_chunks)} chunks")
        embeddings = embed_text(
            chunks=processed_chunks,
        )

        # For recreate or new collection, use sequential IDs
        if update_mode == "recreate" or not collection_exists:
            ids = [f"id_{i}" for i in range(len(processed_chunks))]
            logger.info(
                f"Adding {len(processed_chunks)} documents with embeddings to collection"
            )
            collection.add(
                embeddings=embeddings,
                documents=processed_chunks,
                ids=ids,
            )
            operation = "created"
        # For append mode, get existing IDs to avoid conflicts
        else:
            try:
                existing_count = collection.count()
                start_id = existing_count
                ids = [
                    f"id_{i}" for i in range(start_id, start_id + len(processed_chunks))
                ]

                if update_mode == "append":
                    logger.info(
                        f"Appending {len(processed_chunks)} documents with embeddings to collection"
                    )
                    collection.add(
                        embeddings=embeddings,
                        documents=processed_chunks,
                        ids=ids,
                    )
                    operation = "updated (appended)"
                elif update_mode == "upsert":
                    logger.info(
                        f"Upserting {len(processed_chunks)} documents with embeddings to collection"
                    )
                    collection.upsert(
                        embeddings=embeddings,
                        documents=processed_chunks,
                        ids=ids,
                    )
                    operation = "updated (upserted)"
            except Exception as e:
                logger.error(f"Error during {update_mode} operation: {e!s}")
                raise RuntimeError(
                    f"Error during {update_mode} operation: {e!s}"
                ) from e

        return f"Successfully {operation} vector database with {len(processed_chunks)} instructions in collection '{collection_name}'."

    except Exception as e:
        logger.error(f"Error creating/updating vector database: {e!s}", exc_info=True)
        raise RuntimeError(f"Error creating/updating vector database: {e!s}") from e


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
def embed_text(
    chunks: list, model: str = "openai/text-embedding-3-large"
) -> list[list[float]]:
    """
    Embed a list of text chunks using the specified model with automatic retries.
    If the list is large (>2048 chunks), it will process them in smaller batches.

    Args:
        chunks (list): List of text chunks to embed
        model (str, optional): Model to use for embeddings. Default: "openai/text-embedding-3-large"

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

    logger.info(f"Embedding {len(chunks)} text chunks using model: {model}")

    BATCH_SIZE = 2048  # This is a configuration from LiteLLM:
    # https://docs.litellm.ai/docs/embedding/supported_embedding#required-fields

    # Process in batches if the input is large
    if len(chunks) > BATCH_SIZE:
        logger.info(f"Input size exceeds {BATCH_SIZE} chunks, processing in batches")
        all_embeddings = []

        # Process chunks in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            logger.info("Processing batched chunks)")

            try:
                batch_embeddings = embed_text(batch, model=model)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error in batch {i // BATCH_SIZE + 1}: {e!s}")
                raise

        logger.info(
            f"Successfully generated {len(all_embeddings)} embeddings across all batches"
        )
        return all_embeddings

    # For smaller inputs, process normally
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


def chunk_text(text: str) -> list[str]:
    """
    Split a long text into smaller chunks based on the number of lines.
    Args:
        text (str): The text to be split into chunks

    Returns:
        list[str]: A list of text chunks, each containing a single line
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string")

    return [chunk.strip() for chunk in text.split("\n")]


if __name__ == "__main__":
    # Example usage
    example_text = """[BRIEF] Returns metadata from a known LAMMPS potential file given the file path. [/BRIEF]

[DETAILED] This tool provides a quick and reliable way to identify the type and
supported elements of a LAMMPS potential file based solely on its filename.
It eliminates the need to parse the often large and complex contents of the potential files,
which can often exceed the processing limits of many systems or applications.
By returning a structured description, this tool enables the user to determine whether
a given potential file is appropriate for a specific molecular dynamics (MD) simulation.
This is especially useful when selecting the correct interatomic potential for a system
involving specific elements, without having to inspect the file manually or load it entirely.
[/DETAILED]

[PROCEDURAL] When to use this tool:
- Use when you need to quickly determine the type and supported elements
of a LAMMPS potential file based on its filename.
- Best suited for selecting an appropriate potential file for a specific
molecular dynamics (MD) simulation without reading or parsing the full file contents.
- Recommended for gaining a fast, structured overview of a potential file's
applicability to specific element combinations or simulation scenarios.
[/PROCEDURAL]

[WORKFLOW_INTEGRATION] Typical workflow integration:
1. [PREREQUISITE] Select the potential files that might be relevant to
your simulation task. [/PREREQUISITE]
2. [CURRENT] Use this tool to retrieve metadata for each potential file based
on its filename. This will help you quickly identify which potentials are suitable
for your simulation needs. [/CURRENT]
3. [FOLLOW_UP] Based on the metadata returned, choose the appropriate potential file
for your simulation setup, and run the simulation using the potential file
and the tool `run_lammps`. [/FOLLOW_UP]
[/WORKFLOW_INTEGRATION]

[CONTEXTUAL] How this tool works:
- Extracts the file name from the provided file path
- Matches it against a set of known file names
- Returns a structured metadata string for recognized files
- Raises a ValueError if the file name is unrecognized
[/CONTEXTUAL]

[SYNTACTICAL] Usage examples:
[
`get_potential_metadata("sim_data/ffield.reax")`,
`get_potential_metadata("/path/to/potentials/Al99.eam.alloy")`,
`get_potential_metadata("Mg_Zhou04.eam.alloy`,
`get_potential_metadata("/data/Fe-C_Hepburn_Ackland.eam.fs")`,
`get_potential_metadata("Cu_Zhou04.eam.alloy`
]
[/SYNTACTICAL]

Args:
file_path (str):
[BRIEF] Absolute path to the potential file. [/BRIEF]
[DETAILED] This is the absolute path to a LAMMPS-compatible potential file
(e.g., ReaxFF or EAM formats). The file name is used to determine metadata,
so it must match one of the known patterns. [/DETAILED]
[SYNTACTICAL] Format: "string ending in a recognized potential filename". [/SYNTACTICAL]
[EXAMPLES] Examples: "/path/to/file/ffield.reax", "ffield_UTA1.ITT" [/EXAMPLES]

Returns:
str :
[BRIEF] Structured metadata string describing the potential file. [/BRIEF]
[DETAILED] The returned string includes the type of interatomic potential
and a list of chemical elements that it supports. This helps in choosing
suitable potentials for simulations involving specific atoms. [/DETAILED]
[EXAMPLES] Example outputs: "{potential type : reax, elements supported :
Carbon (C), Hydrogen (H), Oxygen (O), Calcium (Ca), Silicon (Si),
pair_style : reaxff}" [/EXAMPLES]

[RAISES] Exceptions:
ValueError:
[ERROR_WHEN] If the file name is not recognized. [/ERROR_WHEN]
[ERROR_DETAILS] Raised when the filename does not match any known potential files.
This helps prevent silent failures and makes debugging easier
in automated workflows. [/ERROR_DETAILS]
[ERROR_RECOVERY] To resolve this, ensure the file name matches one of the known
potential files or update the tool to include new potential file
names as needed. [/ERROR_RECOVERY]
[/RAISES]

[LIMITATIONS] Limitations:
- This tool only recognizes a predefined set of potential file names.
If the file name does not match any of the known patterns, it will raise a ValueError.
- The metadata returned is static and does not include dynamic information
from the file contents, such as specific parameters or coefficients used in the potential.
- The tool does not validate the actual contents of the potential file;
it relies solely on the file name for metadata extraction.
[/LIMITATIONS]
"""
    # This is invalid - parse_docstring expects a function, not a string
    # chunks = parse_docstring(example_text)
