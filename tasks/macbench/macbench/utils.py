from pathlib import Path

import chromadb

from corral.utils import embed_text


def vector_database_search(
    query: str, collection_name: str = "default_collection"
) -> list[dict]:
    """Retrieve the top 5 most similar instructions from a vector database based on the query.

    Args:
        query: The search query to find similar instructions
        collection_name: The name of the collection in the vector database (default: "default_collection")

    Returns:
        A list of dictionaries containing the top 5 most similar instructions with their content and metadata

    Raises:
        RuntimeError: If the specified collection doesn't exist
    """
    query_embedding = embed_text(
        chunks=[query],
    )[0]

    persist_directory = Path(Path.cwd()) / "vector_db"
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


def create_vector_database(
    chunks: list[str], collection_name: str = "default_collection"
) -> str:
    """Create a vector database from text instructions, splitting by newlines.

    Args:
        chunks: The text instructions to be stored in the vector database
        collection_name: The name of the collection in the vector database (default: "default_collection")

    Returns:
        A confirmation message indicating the number of chunks stored

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """

    persist_directory = Path(Path.cwd()) / "vector_db"
    persist_directory.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_directory))

    try:
        if collection_name in client.list_collections():
            client.delete_collection(name=collection_name)

        collection = client.create_collection(name=collection_name)

        embeddings = embed_text(
            chunks=chunks,
        )

        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=[f"id_{i}" for i in range(len(chunks))],
        )

        return f"Successfully created vector database with {len(chunks)} instructions in collection '{collection_name}'."

    except Exception as e:
        raise RuntimeError(f"Error creating vector database: {e!s}") from e
