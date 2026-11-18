"""Export annotation documents from MongoDB into a local JSON snapshot.

This script reads the MongoDB connection settings from the same environment
variables used by the application, fetches every document from the configured
collection, and writes the result to
`analysis/results/data/corral_annotations_dump.json` using MongoDB Extended
JSON so BSON values such as `ObjectId` and datetimes are preserved.
"""

import os
import sys
from pathlib import Path

from bson.json_util import dumps as bson_dumps
from dotenv import load_dotenv
from loguru import logger
from pymongo import MongoClient

load_dotenv("../.env", override=True)


def main() -> None:
    """Export the configured MongoDB collection to disk.

    Returns:
        None.
    """
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        logger.error("MONGODB_URI is not set.")
        sys.exit(1)

    db_name = os.environ.get("MONGODB_DB", "Corral")
    coll_name = os.environ.get("MONGODB_COLLECTION", "First-traces")

    output_path = Path("results/data/corral_annotations_dump.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to MongoDB...")
    client = MongoClient(uri)
    coll = client[db_name][coll_name]

    logger.info(
        f"Dumping all documents from {db_name}.{coll_name} to {output_path} ..."
    )
    cursor = coll.find({})
    docs = list(cursor)

    json_str = bson_dumps(docs, indent=4)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(json_str)

    logger.info(f"Done. Wrote {len(docs)} documents to {output_path}.")


if __name__ == "__main__":
    main()
