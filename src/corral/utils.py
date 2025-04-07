import gc
import inspect
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union, get_args, get_origin, get_type_hints

import chromadb
import modal
import tiktoken
from chembench.baseline import Generation, Generations
from litellm import embedding
from loguru import logger
from modal import App, Image, Mount, Secret, Volume
from tenacity import (
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from corral.agents.utils import LiteLLMMessage
from corral.base import ModalTool, Tool, ToolArgument

MODAL_TOOL_REGISTRY = {}


class Model:
    def __init__(self, name: str = "Dummy Model"):
        self.name = name

    def generate(self, prompts: list[str], **_kwargs):
        generations = []
        for _prompt in prompts:
            generation = None
            generations.append([Generation(text=generation)])

        return Generations(generations=generations)


def vector_database_search(
    query: str, collection_name: str = "default_collection", path: str | None = None
) -> list[dict]:
    """Retrieve the top 5 most similar instructions from a vector database based on the query.

    Args:
        query (str): The search query to find similar instructions
        collection_name (str, optional): The name of the collection in the vector database. Default is "default_collection".
        path (str): The path to the vector database directory. Defaults to None, which uses "vector_db" in the current directory.

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

        results = collection.query(query_embeddings=[query_embedding], n_results=5)

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
        # Get list of collections and check if our collection exists
        collection_list = [col.name for col in client.list_collections()]
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
    before=before_log(logger, "INFO"),
    after=before_log(logger, "INFO"),
)
def embed_text(
    chunks: list, model: str = "openai/text-embedding-3-large"
) -> list[list[float]]:
    """
    Embed a list of text chunks using the specified model with automatic retries.
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


def parse_docstring(func: Callable) -> tuple[str, list[ToolArgument]]:
    """Parse function docstring to get description and arguments.

    This function extracts the description and arguments from a function's docstring.
    It expects a docstring with a description section and an Args section.

    Args:
        func (Callable): The function to parse docstring from

    Returns:
        tuple[str, list[ToolArgument]]: (description, arguments) where description is a string and
               arguments is a list of ToolArgument objects

    Raises:
        ValueError: If the docstring is missing or doesn't have an Args section
    """
    doc = inspect.getdoc(func)
    if not doc:
        raise ValueError(f"Function {func.__name__} must have a docstring")

    # Split docstring into sections
    sections = doc.split("\n\n")
    description = sections[0].strip()

    # Find Args section
    args_section = None
    for section in sections:
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

    return description, arguments


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
        description, arguments = parse_docstring(func)
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
                name=func.__name__, description=description, arguments=arguments
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


def serialize_messages(messages: list[LiteLLMMessage]) -> list[dict]:
    """
    Serialize LiteLLMMessage objects to a format that can be saved to a JSON file.

    Args:
        messages (List[LiteLLMMessage]): The messages to serialize.

    Returns:
        List[Dict]: The serialized messages.
    """
    serializable_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            message_dict = msg.copy()
        else:
            message_dict = {"role": msg.role, "content": msg.content}

            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, "name") and msg.name:
                message_dict["name"] = msg.name
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

        serializable_messages.append(message_dict)

    return serializable_messages


def save_agent_messages(
    messages: list[LiteLLMMessage],
    task_id: str,
    agent_name: str,
    output_dir: str = "agent_logs",
) -> str:
    """Save agent conversation to a JSON file for logging and analysis purposes.

    This function handles both regular dictionaries and LiteLLMMessage objects,
    properly serializing them for storage.

    Args:
        messages (list[LiteLLMMessage]): List of message objects (LiteLLMMessages or dictionaries)
        task_id (str): The ID of the task being solved
        agent_name (str): The name of the agent that generated the messages
        output_dir (str, optional): Directory to save the logs (will be created if it doesn't exist). Default is "agent_logs".

    Returns:
        str: Path to the saved file
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{task_id}_{timestamp}.json"
    file_path = Path(output_dir) / filename

    # Convert messages to serializable format
    serializable_messages = serialize_messages(messages)

    # Write to file with metadata and pretty formatting
    with Path(file_path).open("w") as f:
        json.dump(
            {
                "task_id": task_id,
                "agent": agent_name,
                "timestamp": timestamp,
                "messages": serializable_messages,
            },
            f,
            indent=2,
        )

    return file_path


def remote_call(function_name: str, env_name: str = "chemenv"):
    def wrapper(arg: str) -> str:
        remote = modal.Function.from_name(env_name, function_name)
        return remote.remote(arg)

    return wrapper
