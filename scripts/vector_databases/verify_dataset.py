from pathlib import Path

import chromadb
from loguru import logger

from corral.utils.tool_helpers import embed_text


def verify_vector_database(db_path: str, collection_name: str):
    """
    Verify that a vector database exists and can be queried.

    Args:
        db_path: Path to the vector database
        collection_name: Name of the collection to check
    """
    # Check if database path exists
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        logger.error(f"Database path does not exist: {db_path}")
        return False

    logger.info(f"Database directory exists at: {db_path}")

    try:
        # Initialize ChromaDB client with updated configuration
        client = chromadb.PersistentClient(path=str(db_path))

        # Get list of collections - in v0.6.0+ this returns just the names
        collections = client.list_collections()

        logger.info(f"Found collections: {collections}")

        if collection_name not in collections:
            logger.error(f"Collection '{collection_name}' not found in the database")
            return False

        # Get the collection
        collection = client.get_collection(name=collection_name)

        # Check collection count
        count = collection.count()
        logger.info(f"Collection '{collection_name}' contains {count} documents")

        if count == 0:
            logger.warning(f"Collection '{collection_name}' exists but is empty")
            return False

        # Try a simple query to verify functionality using the same embedding function
        query = "Copper"
        query_embedding = embed_text(
            chunks=[query], model="openai/text-embedding-3-large"
        )[0]

        logger.info(f"Generated embedding for test query '{query}'")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
        )

        logger.info(
            f"Test query '{query}' retrieved {len(results['documents'][0])} results"
        )

        # Display a snippet of the first result
        if results["documents"][0]:
            first_doc = results["documents"][0][0]
            snippet = first_doc[:200] + "..." if len(first_doc) > 200 else first_doc
            logger.info(f"First result snippet: {snippet}")

        return True

    except Exception as e:
        logger.error(f"Error verifying database: {e}")
        return False


if __name__ == "__main__":
    # Use the same path as in the embedding script
    db_path = "../vector_databases/materials_compatibility"

    success = verify_vector_database(db_path, "materials_compatibility")

    if success:
        logger.success("Vector database verification successful!")
    else:
        logger.error("Vector database verification failed!")
