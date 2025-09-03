import gc
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints
from urllib.parse import quote

import chromadb
import modal
import more_itertools
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


def format_type_annotation(annotation) -> str:
    """Formats type annotations to readable strings with proper error handling."""
    try:
        # Handle None type
        if annotation is type(None):
            return "None"

        # Handle basic types
        if isinstance(annotation, type):
            return annotation.__name__

        # Handle typing constructs
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Union types (including Optional which is Union[T, None])
        if origin is Union:
            # Handle Optional[T] which is Union[T, None]
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return f"{format_type_annotation(non_none_type)} | None"
            else:
                # Handle regular Union[T, U, ...]
                formatted_args = [format_type_annotation(arg) for arg in args]
                return " | ".join(formatted_args)

        # Handle Python 3.10+ union syntax (str | int) using types.UnionType
        try:
            is_union_type = isinstance(annotation, types.UnionType)
        except (ImportError, AttributeError):
            is_union_type = False

        if is_union_type:
            # Recursively format all types in the union
            formatted_args = [
                format_type_annotation(arg) for arg in annotation.__args__
            ]
            # Handle Optional[T] (T | None)
            if len(formatted_args) == 2 and "None" in formatted_args:
                non_none_type = next(a for a in formatted_args if a != "None")
                return f"{non_none_type} | None"
            return " | ".join(formatted_args)

        # Handle generic types like List[str], Dict[str, int], etc.
        if origin is not None:
            origin_name = getattr(origin, "__name__", str(origin))
            if args:
                formatted_args = [format_type_annotation(arg) for arg in args]
                return f"{origin_name}[{', '.join(formatted_args)}]"
            return origin_name

        # Fallback to string representation for unknown types
        return str(annotation).replace("typing.", "")

    except Exception as e:
        logger.warning(f"Could not format type annotation {annotation}: {e}")
        return str(annotation)


def extract_tagged_content(text: str, tag: str) -> str | None:
    """Extract content from a specific tagged section with error handling"""
    try:
        pattern = rf"\[{re.escape(tag)}\](.*?)\[/{re.escape(tag)}\]"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    except Exception as e:
        logger.warning(f"Error extracting tagged content for tag '{tag}': {e}")
        return None


def extract_main_description_from_complex_docstring(docstring: str) -> str:
    """Extract main description from complex docstring with tagged sections"""
    try:
        # First try to get BRIEF section
        brief = extract_tagged_content(docstring, "BRIEF")
        if brief:
            return brief

        # If no BRIEF, get content before first tagged section or Args section
        lines = docstring.strip().split("\n")
        description_lines = []

        for line in lines:
            line_text = line.strip()
            # Stop at first tagged section or Args section
            if line_text.startswith(("[", "Args:")):
                break
            if line_text:
                description_lines.append(line_text)

        return (
            " ".join(description_lines)
            if description_lines
            else "No description available"
        )

    except Exception as e:
        logger.warning(f"Error extracting main description: {e}")
        return "No description available"


def parse_argument_from_lines(arg_lines: list[str]) -> dict[str, Any]:
    """Parse a single argument from its lines with improved error handling"""
    if not arg_lines:
        return {}

    try:
        # First line should contain "arg_name: description"
        first_line = arg_lines[0]
        if ":" not in first_line:
            return {}

        # Extract argument name and start of description
        colon_pos = first_line.find(":")
        arg_name_part = first_line[:colon_pos].strip()
        first_desc_part = first_line[colon_pos + 1 :].strip()

        # Extract bare argument name (remove type annotation if present)
        if "(" in arg_name_part and ")" in arg_name_part:
            arg_name = arg_name_part.split("(")[0].strip()
        else:
            arg_name = arg_name_part.strip()

        # Validate argument name
        if not arg_name or not arg_name.isidentifier():
            logger.warning(f"Invalid argument name: {arg_name}")
            return {}

        # Combine all description lines
        all_desc_parts = [first_desc_part] + [line.strip() for line in arg_lines[1:]]
        full_description = " ".join(part for part in all_desc_parts if part)

        # Parse choices if specified
        choices = None
        if "(choices:" in full_description:
            try:
                desc_parts = full_description.split("(choices:", 1)
                full_description = desc_parts[0].strip()
                choices_str = desc_parts[1].split(")", 1)[0].strip()
                # Safely evaluate choices
                choices = eval(
                    choices_str
                )  # This should be replaced with ast.literal_eval for safety
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Invalid choices format for argument {arg_name}: {e}")

        # Extract tagged sections from argument description
        raises_info = extract_tagged_content(full_description, "RAISES")
        limitations_info = extract_tagged_content(full_description, "LIMITATIONS")

        return {
            "name": arg_name,
            "description": full_description,
            "choices": choices,
            "raises": raises_info,
            "limitations": limitations_info,
        }

    except Exception as e:
        logger.error(f"Error parsing argument from lines: {e}")
        return {}


def parse_args_section(args_section: str) -> list[dict[str, Any]]:
    """Parse the Args section to extract individual arguments with their descriptions"""
    if not args_section:
        return []

    try:
        # Split into lines and remove "Args:" header
        lines = args_section.splitlines()
        if lines and lines[0].strip().startswith("Args:"):
            lines = lines[1:]

        arguments = []
        current_arg_lines = []

        # Pattern to match argument start: "arg_name:" or "arg_name (type):"
        arg_pattern = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:")

        for line in lines:
            if arg_pattern.match(line):
                # New argument found, process previous one
                if current_arg_lines:
                    arg_data = parse_argument_from_lines(current_arg_lines)
                    if arg_data.get("name"):
                        arguments.append(arg_data)
                current_arg_lines = [line]
            elif current_arg_lines:
                # Continuation of current argument
                current_arg_lines.append(line)

        # Process the last argument
        if current_arg_lines:
            arg_data = parse_argument_from_lines(current_arg_lines)
            if arg_data.get("name"):
                arguments.append(arg_data)

        return arguments

    except Exception as e:
        logger.error(f"Error parsing args section: {e}")
        return []


def is_complex_docstring(docstring: str) -> bool:
    """Check if docstring has tagged sections like [BRIEF], [DETAILED], etc."""
    if not docstring:
        return False

    tagged_keywords = [
        "BRIEF",
        "DETAILED",
        "PROCEDURAL",
        "CONTEXTUAL",
        "WORKFLOW_INTEGRATION",
        "SYNTACTICAL",
        "RAISES",
        "LIMITATIONS",
    ]
    return any(f"[{keyword}]" in docstring for keyword in tagged_keywords)


def parse_docstring(func: Callable) -> tuple[str, list[ToolArgument]]:
    """Parse function docstring to get description and arguments with comprehensive error handling."""

    # Validate function
    if not callable(func):
        raise ValueError(f"Expected callable function, got {type(func)}")

    # Get docstring
    doc = inspect.getdoc(func)
    if not doc:
        raise ValueError(f"Function {func.__name__} must have a docstring")

    try:
        # Determine if this is a complex docstring with tagged sections
        if is_complex_docstring(doc):
            # Extract main description from complex docstring
            # description = extract_main_description_from_complex_docstring(doc)
            doc_without_args_returns = re.sub(
                r"Args:.*?(?=Returns:|$)", "", doc, flags=re.DOTALL
            )
            description = doc_without_args_returns.strip()
        else:
            # Handle simple docstring - use first section as description
            sections = doc.split("\n\n")
            description = sections[0].strip()

        # Validate description
        if not description or description == "No description available":
            logger.warning(f"Function {func.__name__} has empty or invalid description")

        # Find Args section (works for both simple and complex docstrings)
        args_section = None

        # Look for Args section in the docstring
        args_match = re.search(
            r"Args:(.*?)(?=Returns:|$)", doc, re.DOTALL | re.IGNORECASE
        )
        if args_match:
            args_section = "Args:" + args_match.group(1)
        else:
            # Fallback: look for Args section in split sections (for simple docstrings)
            sections = doc.split("\n\n")
            for section in sections:
                if section.strip().startswith("Args:"):
                    args_section = section.strip()
                    break

        if not args_section:
            raise ValueError(
                f"Function {func.__name__} docstring must have an 'Args:' section"
            )

        # Parse arguments from the Args section
        parsed_args = parse_args_section(args_section)

        # Get type hints and signature from function
        try:
            type_hints = get_type_hints(func)
            signature = inspect.signature(func)
        except Exception as e:
            raise ValueError(
                f"Error getting type hints for {func.__name__}: {e}"
            ) from e

        arguments = []

        # Convert parsed arguments to ToolArgument objects
        for arg_data in parsed_args:
            try:
                arg_name = arg_data["name"]

                # Skip if not a real parameter
                if arg_name not in type_hints:
                    logger.warning(
                        f"Argument {arg_name} not found in type hints for {func.__name__}"
                    )
                    continue

                # Get type and default from function signature
                arg_annotation = type_hints[arg_name]
                arg_type = format_type_annotation(arg_annotation)

                param = signature.parameters.get(arg_name)
                has_default = param and param.default != inspect.Parameter.empty
                default_value = param.default if has_default else None

                arguments.append(
                    ToolArgument(
                        name=arg_name,
                        type=arg_type,
                        description=arg_data["description"],
                        required=not has_default,
                        default=default_value,
                        choices=arg_data["choices"],
                    )
                )

            except Exception as e:
                logger.error(
                    f"Error processing argument {arg_data.get('name', 'unknown')} for {func.__name__}: {e}"
                )
                continue

        return description, arguments

    except Exception as e:
        logger.error(f"Error parsing docstring for {func.__name__}: {e}")
        raise ValueError(
            f"Error parsing docstring for function {func.__name__}: {e}"
        ) from e


def tool(func: Callable | None = None, *, hidden_args: list[str] | None = None) -> Tool:
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
        hidden_args (list[str], optional): List of parameter names that exist in the function
            signature but should not be documented in the docstring Args section.
            These parameters will be excluded from docstring validation. This allows for defining
            tools with fixed arguements such that the agent does not need to provide them.

    Returns:
        Tool: A Tool instance wrapping the function

    Raises:
        ValueError: If the function lacks proper docstring, type hints, or has invalid format
        TypeError: If the function signature is incompatible with Tool requirements
    """

    def decorator(func: Callable) -> Tool:
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
            description, arguments = parse_docstring(func)
        except Exception as e:
            raise ValueError(
                f"Error parsing docstring for function {func.__name__}: {e!s}. "
                "Please ensure it follows the required format shown in the decorator documentation."
            ) from e

        # Validate all non-hidden parameters have documentation
        signature_params = set(inspect.signature(func).parameters.keys())
        hidden_params = set(hidden_args or [])
        documented_params = {arg.name for arg in arguments}

        # Only require documentation for parameters that are not hidden
        required_docs = signature_params - hidden_params
        if missing_docs := required_docs - documented_params:
            raise ValueError(
                f"Missing documentation for parameters: {', '.join(missing_docs)}. "
                "All non-hidden parameters must be documented in the Args section of the docstring."
            )

        # Check for invalid hidden args (args that don't exist in signature)
        if invalid_hidden := hidden_params - signature_params:
            raise ValueError(
                f"Hidden args {', '.join(invalid_hidden)} do not exist in function signature."
            )

        # Prepare hidden_args dict for Tool instance
        hidden_args_dict = {}
        if hidden_args:
            sig = inspect.signature(func)
            for arg in hidden_args:
                param = sig.parameters[arg]
                # Use default value if available, else None
                hidden_args_dict[arg] = (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else None
                )

        class FunctionTool(Tool):
            def __init__(self):
                super().__init__(
                    name=func.__name__,
                    description=description,
                    arguments=arguments,
                    hidden_args=hidden_args_dict if hidden_args_dict else None,
                )

            def execute(self, **kwargs):
                return str(func(**kwargs))

        return FunctionTool()

    # Handle both @tool and @tool(hidden_args=["arg1", "arg2"]) syntax
    if func is None:
        # Called with arguments: @tool(hidden_args=["arg1"])
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)


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
        description, arguments = parse_docstring(func)

        tool_instance = ModalTool(
            modal_func=modal_func,
            name=func.__name__,
            description=description,
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
    model: str = "openai/text-embedding-3-large",
    chemical_model: str | None = None,
) -> list[dict]:
    """Retrieve the top 5 most similar instructions from a vector database based on the query.
    If a chemical model is provided, it will use that model to generate embeddings instead of the text embedding model.

    Args:
        query (str): The search query to find similar instructions
        collection_name (str, optional): The name of the collection in the vector database. Default is "default_collection".
        path (str, optional): The path to the vector database directory. Defaults to None, which uses "vector_db" in the current directory.
        top_k (int, optional): The number of similar instructions to retrieve. Default is 5.
        model (str, optional): The model to use for embedding. Default is "openai/text-embedding-3-large".
        chemical_model (str, optional): The model to use for chemical embeddings. If provided, it will use this model instead of the text embedding model. Default is None.

    Returns:
        list[dict]: A list of dictionaries containing the top similar instructions with their content and metadata

    Raises:
        RuntimeError: If the specified collection doesn't exist
    """
    chemical = False
    if chemical_model is not None:
        model = chemical_model
        chemical = True

    query_embedding = embed_text(chunks=[query], model=model, chemical=chemical)[0]

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

        include = ["documents", "metadatas", "distances"]

        results = collection.query(
            query_embeddings=[query_embedding], n_results=top_k, include=include
        )

        formatted_results = []

        for i in range(len(results["ids"][0])):
            result_item = {
                "id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": 1 - results["distances"][0][i],
                "content": results["documents"][0][i],
            }

            formatted_results.append(result_item)

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
                f"Processing batch {batch_idx//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}"
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
                        f"Sub-chunk {sub_chunk_count+1}: Processing from token {start_idx} to {end_idx} ({end_idx-start_idx} tokens)"
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


def _setup_collection(
    client: chromadb.PersistentClient, collection_name: str, update_mode: str
) -> tuple[Any, str, bool]:
    """Setup the collection based on the update mode and return it with operation status."""
    operation = "created"  # Default operation status
    collection_exists = False

    # Get list of collection names instead of collection objects
    collection_names = [col.name for col in client.list_collections()]
    collection_exists = collection_name in collection_names

    logger.debug(f"Available collections: {collection_names}")
    logger.debug(f"Collection '{collection_name}' exists: {collection_exists}")

    try:
        if update_mode == "recreate" and collection_exists:
            logger.info(
                f"Collection '{collection_name}' already exists, deleting before recreation"
            )
            client.delete_collection(name=collection_name)
            # Create a small delay to ensure deletion completes
            import time

            time.sleep(0.5)
            collection = client.create_collection(name=collection_name)
            logger.info(f"Recreated collection '{collection_name}'")
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
            if update_mode == "append":
                operation = "updated (appended)"
            elif update_mode == "upsert":
                operation = "updated (upserted)"
    except Exception as e:
        logger.error(f"Error setting up collection: {e!s}")
        raise RuntimeError(
            f"Error setting up collection '{collection_name}': {e!s}"
        ) from e

    return collection, operation, collection_exists


def _validate_inputs(
    chunks: list[str],
    update_mode: str,
    metadatas: list[dict] | None,
    chemical: list[str] | None,
) -> None:
    """Validate inputs for the create_vector_database function."""
    logger.info(f"Creating vector database with {len(chunks)} chunks")

    if metadatas is not None and len(metadatas) != len(chunks):
        raise ValueError(
            f"Length of metadata ({len(metadatas)}) must match length of chunks ({len(chunks)})"
        )

    if isinstance(chemical, list) and len(chemical) != len(chunks):
        raise ValueError(
            f"Length of chemical flags ({len(chemical)}) must match length of chunks ({len(chunks)})"
        )

    if update_mode not in ["recreate", "append", "upsert"]:
        raise ValueError("update_mode must be one of: 'recreate', 'append', 'upsert'")


def _setup_database_environment(
    path: str | None,
) -> tuple[Path, chromadb.PersistentClient]:
    """Setup the vector database environment and return the directory and client."""
    persist_directory = Path(Path.cwd()) / "vector_db" if path is None else Path(path)
    persist_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using persist directory: {persist_directory}")

    return persist_directory, chromadb.PersistentClient(path=str(persist_directory))


def _add_documents_to_collection(
    collection,
    update_mode: str,
    batch_embeddings: list[list[float]],
    batch_chunks: list[str],
    batch_ids: list[str],
    batch_metadatas: list[dict] | None,
) -> None:
    """Add documents to the collection using the appropriate method based on update mode."""
    try:
        if update_mode == "upsert":
            collection.upsert(
                embeddings=batch_embeddings,
                documents=batch_chunks,
                ids=batch_ids,
                metadatas=batch_metadatas,
            )
        else:  # For both "recreate" and "append" modes
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_chunks,
                ids=batch_ids,
                metadatas=batch_metadatas,
            )
    except Exception as e:
        logger.error(f"Error during {update_mode} operation: {e!s}")
        raise RuntimeError(f"Error during {update_mode} operation: {e!s}") from e


def _process_chunks_in_batches(
    collection,
    processed_chunks: list[str],
    embeddings: list[list[float]],
    start_id: int,
    update_mode: str,
    metadatas: list[dict] | None,
) -> int:
    """Process chunks in batches to add them to the collection."""
    BATCH_SIZE = 1000
    total_processed = 0

    logger.info(
        f"Processing {len(processed_chunks)} documents with embeddings in batches of {BATCH_SIZE}"
    )

    total_batches = (len(processed_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    metadata_chunks = (
        more_itertools.chunked(metadatas, BATCH_SIZE)
        if metadatas is not None
        else [None] * total_batches
    )

    for batch_idx, (chunk_batch, embedding_batch, metadata_batch) in enumerate(
        zip(
            more_itertools.chunked(processed_chunks, BATCH_SIZE),
            more_itertools.chunked(embeddings, BATCH_SIZE),
            metadata_chunks,
            strict=False,
        )
    ):
        batch_ids = [
            f"id_{start_id + total_processed + i}" for i in range(len(chunk_batch))
        ]

        logger.info(
            f"Processing batch {batch_idx + 1}/{total_batches} "
            f"({len(chunk_batch)} documents)"
        )

        _add_documents_to_collection(
            collection,
            update_mode,
            embedding_batch,
            chunk_batch,
            batch_ids,
            metadata_batch,
        )

        total_processed += len(chunk_batch)
        logger.info(f"Processed {total_processed}/{len(processed_chunks)} documents")

        # Force garbage collection between batches
        gc.collect()

    return total_processed


def create_vector_database(
    chunks: list[str],
    collection_name: str = "default_collection",
    path: str | None = None,
    chunk_size: int = 8192,
    update_mode: str = "recreate",
    metadatas: list[dict] | None = None,
    model: str = "openai/text-embedding-3-large",
    chemical: list[str] | None = None,
    chemical_model: str = "ibm-research/MoLFormer-XL-both-10pct",
) -> str:
    """Create or update a vector database from text instructions. If chemical data is provided,
    it will be used to generate embeddings instead of the text chunks. The database will be build
    with the chemical embeddings as the primary vectors, and the original text will be stored in metadata.

    Args:
        chunks (list[str]): The text instructions to be stored in the vector database
        collection_name (str, optional): The name of the collection in the vector database. Default is "default_collection".
        path (str, optional): Path to store the vector database. Default is "vector_db" in the current directory.
        chunk_size (int, optional): Maximum number of tokens per chunk. Default is 8192.
        update_mode (str, optional): How to handle existing collections: "recreate" deletes and recreates the collection,
                    "append" adds new chunks to existing collection, "upsert" updates existing chunks and
                    adds new ones. Default is "recreate".
        metadatas (list[dict], optional): Metadata for each chunk. Default is None.
        model (str, optional): The model to use for embedding. Default is "openai/text-embedding-3-large".
        chemical (list[str], optional): List of chemical data to embed separately. Default is None.
        chemical_model (str, optional): The model to use for chemical embeddings. Default is "ibm-research/MoLFormer-XL-both-10pct".

    Returns:
        str: A message indicating the success of the operation
    """
    try:
        # Validate inputs
        _validate_inputs(chunks, update_mode, metadatas, chemical)

        # Setup database environment
        persist_directory, client = _setup_database_environment(path)

        # Setup collection based on update mode
        collection, operation, collection_exists = _setup_collection(
            client, collection_name, update_mode
        )

        # Initialize metadata if none provided
        if metadatas is None:
            metadatas = [{} for _ in range(len(chunks))]

        # If chemical is provided, invert the storage approach
        if chemical is not None:
            processed_chunks = chunks
            # Generate chemical embeddings as the primary vectors
            logger.info(f"Generating chemical embeddings for {len(chemical)} items")
            primary_embeddings = embed_text(
                chunks=chemical, model=chemical_model, chemical=True
            )

            # Store original text in metadata for reference
            for i, chunk in enumerate(chunks):
                if i < len(metadatas):
                    metadatas[i]["text"] = chunk

            logger.info(
                "Added original text to metadata, using chemical embeddings as primary"
            )
        else:
            # Process chunks for embedding
            processed_chunks = _tokenize_and_split_chunks(chunks, chunk_size)
            logger.info(
                f"Processed {len(chunks)} chunks into {len(processed_chunks)} chunks after tokenization and splitting"
            )

            # Standard approach - text embeddings are primary
            logger.info(
                f"Generating text embeddings for {len(processed_chunks)} chunks"
            )
            primary_embeddings = embed_text(chunks=processed_chunks, model=model)

        # Process and add documents in batches
        start_id = (
            0
            if update_mode == "recreate" or not collection_exists
            else collection.count()
        )

        # Process in batches and add to collection
        _process_chunks_in_batches(
            collection,
            processed_chunks,
            primary_embeddings,
            start_id,
            update_mode,
            metadatas,
        )

        # Log which embeddings are primary
        embedded_type = "text" if chemical is None else "chemical"

        return f"Successfully {operation} vector database with {len(processed_chunks)} instructions in collection '{collection_name}' using {embedded_type} as primary embeddings."

    except Exception as e:
        logger.error(f"Error creating/updating vector database: {e!s}", exc_info=True)
        raise RuntimeError(f"Error creating/updating vector database: {e!s}") from e


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


#################################
## coding tool
#################################
def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure the parent directory of a file path exists.

    Args:
        file_path: Path to a file
    """
    from pathlib import Path

    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def generate_output_capture_code() -> str:
    """
    Generate the Python code string for capturing execution results.

    This utility function creates the code that gets appended to user code
    to capture variables and prepare them for JSON serialization.

    Returns:
        str: Python code string for variable capture
    """
    return """

import json
import types
import sys
from collections.abc import Iterable

result_ = {}

def is_json_serializable(obj, max_depth=10, current_depth=0):
    '''Check if object is JSON serializable with depth limit'''
    if current_depth > max_depth:
        return False

    try:
        # Handle basic types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return True

        # Handle sequences (but not strings)
        if isinstance(obj, (list, tuple)):
            return all(is_json_serializable(item, max_depth, current_depth + 1) for item in obj)

        # Handle dictionaries
        if isinstance(obj, dict):
            return all(
                isinstance(k, str) and is_json_serializable(v, max_depth, current_depth + 1)
                for k, v in obj.items()
            )

        # Quick test with actual JSON serialization for edge cases
        json.dumps(obj)
        return True
    except (TypeError, ValueError, RecursionError, OverflowError):
        return False

def safe_convert_to_serializable(obj):
    '''Convert common non-serializable types to serializable ones'''
    try:
        import numpy as np
        # Handle numpy arrays
        if hasattr(obj, '__module__') and obj.__module__ == 'numpy':
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
    except ImportError:
        pass

    # Handle sets
    if isinstance(obj, set):
        return list(obj)

    # Handle other iterables (but not strings/bytes)
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
        try:
            return list(obj)
        except:
            pass

    return obj

# Try to capture from preferred variable names first
preferred_vars = ['output', 'result', 'filtered_data', 'processed_data', 'dataset']
captured = False

for var_name in preferred_vars:
    if var_name in locals():
        var_value = locals()[var_name]
        converted_value = safe_convert_to_serializable(var_value)
        if is_json_serializable(converted_value):
            result_[var_name] = converted_value
            captured = True
            break

# Fallback: capture any user-defined variables
if not captured:
    excluded = {
        '__builtins__', '__name__', '__doc__', '__package__', '__loader__',
        '__spec__', '__annotations__', '__cached__', '__file__',
        'json', 'types', 'sys', 'Iterable', 'result_', 'preferred_vars',
        'captured', 'excluded', 'local_vars', 'is_json_serializable',
        'safe_convert_to_serializable', 'var_name', 'var_value', 'converted_value'
    }

    local_vars = dict(locals())  # Create snapshot

    # Collect all suitable variables
    suitable_vars = {}
    for var_name, var_value in local_vars.items():
        if (not var_name.startswith('_') and
            var_name not in excluded and
            not isinstance(var_value, types.ModuleType) and
            not callable(var_value)):

            converted_value = safe_convert_to_serializable(var_value)
            if is_json_serializable(converted_value):
                suitable_vars[var_name] = converted_value

    # If we have suitable variables, include them all
    if suitable_vars:
        result_.update(suitable_vars)

print('EXECUTION_RESULT:', json.dumps(result_))
"""


def safe_convert_timeout(timeout) -> int:
    """
    Safely convert timeout parameter to integer.

    Args:
        timeout: Timeout value (could be string, int, float, or None)

    Returns:
        int: Valid timeout value
    """
    if timeout is None:
        return 300

    try:
        return int(float(timeout))  # Handle both "60" and "60.5"
    except (ValueError, TypeError):
        return 300  # Default fallback


def parse_execution_output(stdout: str) -> tuple[dict, list[str]]:
    """
    Parse subprocess output to extract execution results and regular output.

    Args:
        stdout: Raw stdout from subprocess

    Returns:
        tuple: (execution_result dict, output_lines list)
    """
    import json
    from contextlib import suppress

    stdout_lines = stdout.strip().split("\n") if stdout.strip() else []
    execution_result = {}
    output_lines = []

    for line in stdout_lines:
        if line.startswith("EXECUTION_RESULT:"):
            with suppress(json.JSONDecodeError):
                execution_result = json.loads(line[17:])
        else:
            output_lines.append(line)

    return execution_result, output_lines


def execute_python_code_given_code(code: str) -> str:
    """
    Executes a given Python code string.
    This is a placeholder and should be replaced with your actual implementation.
    """
    try:
        # Create a dictionary to hold local variables during execution
        exec_globals = {}
        exec_locals = {}
        exec(code, exec_globals, exec_locals)
        # Assuming the filtering code will produce a 'output' variable
        return json.dumps(
            {"success": True, "execution_result": {"output": exec_locals.get("output")}}
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


@tool
def execute_python_code(
    python_code: str,
    input_data: str | None = None,
    save_output_to: str | None = None,
    timeout: int = 300,
) -> str:
    """[BRIEF] Execute Python code in a secure environment with data input/output capabilities. [/BRIEF]

    [DETAILED] This tool provides a secure execution environment for custom Python code, essential for data analysis, custom calculations, and algorithm development in materials science workflows.
    It supports data injection, output capture, and file saving capabilities while maintaining security through process isolation and timeout controls.
    This enables flexible custom analysis. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to execute custom Python analysis or calculations
    - Best suited for data processing and custom algorithm development
    - Essential for implementing custom filtering, analysis, or transformation logic
    - Recommended for prototyping and testing analysis workflows
    - Avoid for simple operations that can be done with existing tools
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Creates isolated subprocess environment for secure code execution
    - Injects input data as JSON-parsed variable if provided
    - Captures standard output, error streams, and execution results
    - Implements timeout protection to prevent infinite loops
    - Extracts variables from executed code for result capture
    - Saves results to file if requested for persistence
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Prepare input data and ensure code is syntactically correct [/PREREQUISITE]
    2. [CURRENT] Execute custom Python code with data processing or analysis [/CURRENT]
    3. [FOLLOW_UP] Use captured results for further analysis or save to files [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `execute_python_code("result = sum([1, 2, 3, 4, 5])", None, None, 30)`,
    `execute_python_code("filtered_data = [x for x in input_data if x > 0.5]", json_data, "output.json")`,
    `execute_python_code("import numpy as np; result = np.mean(input_data)", array_data, None, 60)`,
    [/SYNTACTICAL]

    Args:
        python_code: [BRIEF] Python code string to be executed. [/BRIEF]
                    [DETAILED] A string containing valid Python code to be executed in this environment.
                    For best results, assign your main output to a variable named 'result' or 'output'.
                    The code can import standard libraries and perform complex calculations.
                    The tool will attempt to capture user-defined variables as execution results. [/DETAILED]
                    [SYNTACTIC] Valid Python code string [/SYNTACTIC]
                    [EXAMPLES] "result = 2 + 2", "import json; result = json.loads(data)", "filtered = [x for x in data if x > threshold]" [/EXAMPLES]
        input_data: [BRIEF] Optional JSON string to inject as input_data variable. [/BRIEF]
                   [DETAILED] An optional JSON string that will be loaded into a Python variable named 'input_data' within the executed script.
                   This allows the script to process external data.
                   The JSON will be parsed and made available as a Python object (dict, list, etc.) depending on the JSON structure. [/DETAILED]
                   [SYNTACTIC] Valid JSON string or None [/SYNTACTIC]
                   [EXAMPLES] "{"data": [1, 2, 3]}", '[1, 2, 3, 4, 5]', "{"threshold": 0.5, "values": [...]}" [/EXAMPLES]
        save_output_to: [BRIEF] Optional file path to save execution results. [/BRIEF]
                       [DETAILED] An optional file path where the captured execution results will be saved as a JSON file.
                       If provided and execution is successful, the results will be written to this file for persistence and later use.
                       The directory will be created if it doesn't exist. [/DETAILED]
                       [SYNTACTIC] "Valid file path or None" [/SYNTACTIC]
                       [EXAMPLES] "results.json", "output/analysis_results.json", "data/processed_output.json" [/EXAMPLES]
        timeout: [BRIEF] Maximum execution time in seconds. Defaults to 300. [/BRIEF]
                [DETAILED] The maximum time in seconds the subprocess is allowed to run before being terminated.
                This prevents infinite loops and runaway processes from consuming system resources.
                If the execution exceeds this limit, a timeout error will be returned.
                Choose appropriate values based on expected computation time. [/DETAILED]
                [SYNTACTIC] positive integer representing seconds [/SYNTACTIC]
                [EXAMPLES] 30 (quick calculations), 300 (standard), 1800 (long processing) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with detailed execution results and captured output. [/BRIEF]
             [DETAILED] A comprehensive JSON string containing execution status, standard output, error messages, return code, captured execution results, and file save status.
             The execution_result field contains variables captured from the executed code.
             This enables full visibility into the execution process and results. [/DETAILED]
             [EXAMPLES] "{"success": true, "execution_result": {"result": 10}, "stdout": "...", "stderr": "", "return_code": 0}" [/EXAMPLES]

    [RAISES] Exceptions:
        TimeoutExpired: [ERROR_WHEN] When code execution exceeds the specified timeout [/ERROR_WHEN]
                       [ERROR_DETAILS] Process terminated due to timeout limit [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Increase timeout value or optimize code for faster execution [/ERROR_RECOVERY]
        SyntaxError: [ERROR_WHEN] When the Python code contains syntax errors [/ERROR_WHEN]
                    [ERROR_DETAILS] Invalid Python syntax in the code string [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Check code syntax and fix any errors [/ERROR_RECOVERY]
        RuntimeError: [ERROR_WHEN] When code execution fails due to runtime errors [/ERROR_WHEN]
                     [ERROR_DETAILS] Errors during code execution such as undefined variables [/ERROR_DETAILS]
                     [ERROR_RECOVERY] Debug code logic and ensure all required variables are defined [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Limited to Python standard library and commonly available packages
    - Cannot access external network resources or file system outside working directory
    - Cannot install new packages during execution
    - Does not persist state between executions
    [/LIMITATIONS]
    """
    import json
    import subprocess
    import sys
    import traceback
    from pathlib import Path

    try:
        # Ensure timeout is valid
        timeout = safe_convert_timeout(timeout)

        # Create directory if needed
        if save_output_to:
            ensure_directory_exists(save_output_to)

        # Create temporary file for the code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Prepare the code with input data if provided
            full_code = ""
            if input_data:
                full_code += "import json\n"
                full_code += f"input_data = json.loads({input_data!r})\n"

            full_code += python_code
            full_code += generate_output_capture_code()

            f.write(full_code)
            temp_file = f.name

        # Execute the code
        process = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
            check=False,
        )

        # Clean up
        Path(temp_file).unlink()

        # Parse output
        execution_result, output_lines = parse_execution_output(process.stdout)

        # Save output if requested
        saved_path = None
        if save_output_to and execution_result:
            try:
                with Path(save_output_to).open("w") as f:
                    json.dump(execution_result, f, indent=2)
                saved_path = save_output_to
            except Exception as e:
                logger.warning(
                    f"Warning: Failed to save output to {save_output_to}: {e}"
                )

        result = {
            "success": process.returncode == 0,
            "stdout": "\n".join(output_lines),
            "stderr": process.stderr,
            "return_code": process.returncode,
            "execution_result": execution_result,
            "saved_to": saved_path,
        }
        if process.returncode != 0:
            result["error"] = process.stderr or "Script execution failed"

        return json.dumps(result, indent=2)

    except subprocess.TimeoutExpired:
        from pathlib import Path

        if "temp_file" in locals() and Path(temp_file).exists():
            Path(temp_file).unlink()
        return json.dumps(
            {
                "success": False,
                "error": f"Code execution timed out after {timeout} seconds",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_result": {},
            }
        )
    except Exception as e:
        if "temp_file" in locals() and Path(temp_file).exists():
            Path(temp_file).unlink()
        return json.dumps(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_result": {},
            }
        )


@tool
def execute_python_script(
    script_path: str,
    args: list | None = None,
    timeout: int = 600,
    working_dir: str | None = None,
) -> str:
    """[BRIEF] Execute a Python script file with arguments in a controlled environment. [/BRIEF]

    [DETAILED] This tool executes existing Python script files with command-line arguments, providing a controlled environment for running complex analysis workflows, data processing pipelines, or computational simulations.
    It captures all output streams and provides comprehensive execution monitoring with timeout protection.
    This is essential for integrating existing Python scripts into automated workflows and materials analysis pipelines. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to execute existing Python scripts with specific arguments. You can also use io tool to write a script and then execute it.
    - Best suited for running complex analysis workflows or simulations
    - Essential for integrating external Python tools into automated pipelines
    - Recommended for batch processing and computational workflows
    - Avoid for simple code execution
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Validates script file existence and accessibility
    - Constructs command with script path and provided arguments
    - Executes script in subprocess with timeout protection
    - Captures standard output, error streams, and return codes
    - Provides comprehensive execution monitoring and error reporting
    - Supports custom working directory for script execution
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] Ensure script file exists and is executable with proper dependencies [/PREREQUISITE]
    2. [CURRENT] Execute script with appropriate arguments and timeout [/CURRENT]
    3. [FOLLOW_UP] Process script output and results for further analysis. Can be used to process json script as required [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `execute_python_script("analysis.py", ["--input", "data.json", "--output", "results.json"], 300)`,
    `execute_python_script("simulation.py", ["--steps", "1000", "--temp", "300"], 1800, "/path/to/workdir")`,
    `execute_python_script("processing.py", None, 600, None)`,
    [/SYNTACTICAL]

    Args:
        script_path: [BRIEF] Path to the Python script file to execute. [/BRIEF]
                    [DETAILED] Complete file path to the Python script that should be executed.
                    The script must exist and be readable.
                    The path can be relative to the current working directory or absolute.
                    The script should be a valid Python file with appropriate shebang or run using the Python interpreter. [/DETAILED]
                    [SYNTACTIC] "Valid file path to Python script" [/SYNTACTIC]
                    [EXAMPLES] "scripts/analysis.py", "/home/user/simulations/run_sim.py", "data_processing.py" [/EXAMPLES]
        args: [BRIEF] Optional list of command-line arguments for the script. [/BRIEF]
             [DETAILED] A list of strings representing command-line arguments to pass to the script.
             These arguments will be passed to the script in the order provided.
             Common arguments include input files, output paths, configuration parameters, and processing options.
             If None, the script will be executed without arguments. [/DETAILED]
             [SYNTACTIC] ["arg1", "arg2", "arg3", ...] or None [/SYNTACTIC]
             [EXAMPLES] ["--input", "data.json"], ["--verbose", "--output", "results.csv"], None [/EXAMPLES]
        timeout: [BRIEF] Maximum execution time in seconds. Defaults to 600. [/BRIEF]
                [DETAILED] The maximum time in seconds the script is allowed to run before being terminated.
                This prevents runaway processes and ensures resource management.
                Choose appropriate values based on expected script execution time.
                For computational simulations, longer timeouts may be necessary. [/DETAILED]
                [SYNTACTIC] positive integer representing seconds [/SYNTACTIC]
                [EXAMPLES] 300 (5 minutes), 600 (10 minutes), 3600 (1 hour) [/EXAMPLES]
        working_dir: [BRIEF] Optional working directory for script execution. [/BRIEF]
                    [DETAILED] The directory from which the script should be executed.
                    This affects relative path resolution and file I/O operations within the script.
                    If None, the current working directory will be used.
                    This is useful when scripts expect to run from specific directories or access relative files. [/DETAILED]
                    [SYNTACTIC] Valid directory path or None [/SYNTACTIC]
                    [EXAMPLES] "/path/to/project", "data/analysis", None [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with comprehensive execution results and monitoring data. [/BRIEF]
             [DETAILED] A JSON-formatted string containing execution status, captured output streams, error messages, return code, and the complete command that was executed.
             This provides full visibility into the script execution process and enables debugging and monitoring of automated workflows. [/DETAILED]
             [EXAMPLES] "{"success": true, "stdout": "Processing complete", "stderr": "", "return_code": 0, "command": "python script.py --input data.json"}" [/EXAMPLES]

    [RAISES] Exceptions:
        FileNotFoundError: [ERROR_WHEN] When the specified script file doesn't exist [/ERROR_WHEN]
                          [ERROR_DETAILS] Script path is invalid or file is not accessible [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Verify script path exists and is readable [/ERROR_RECOVERY]
        TimeoutExpired: [ERROR_WHEN] When script execution exceeds the specified timeout [/ERROR_WHEN]
                       [ERROR_DETAILS] Script terminated due to timeout limit [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Increase timeout value or optimize script performance [/ERROR_RECOVERY]
        PermissionError: [ERROR_WHEN] When script file lacks execute permissions [/ERROR_WHEN]
                        [ERROR_DETAILS] Insufficient permissions to execute the script [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Check file permissions and ensure script is executable [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Cannot modify script execution environment beyond working directory
    - Limited to Python scripts and available system Python installation
    - No real-time output streaming during execution
    - Cannot interact with scripts requiring user input
    [/LIMITATIONS]
    """
    try:
        if not Path(script_path).exists():
            return json.dumps(
                {"success": False, "error": f"Script file not found: {script_path}"}
            )

        # Prepare command
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(str(arg) for arg in args)

        # Execute
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir or Path.cwd(),
            check=False,
        )

        result = {
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "return_code": process.returncode,
            "command": " ".join(cmd),
        }

        return json.dumps(result, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "success": False,
                "error": f"Script execution timed out after {timeout} seconds",
            }
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
