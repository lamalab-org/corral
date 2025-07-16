import gc
from pathlib import Path
from typing import Any

import chromadb
import more_itertools
import tiktoken
from loguru import logger

from corral.utils.tool_utils import embed_text


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
