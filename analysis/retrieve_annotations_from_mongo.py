"""
Dump all documents from the MongoDB collection defined in your Modal app
to a local JSON file.

It uses the same env vars as your server:
- MONGODB_URI          (required)
- MONGODB_DB           (default: "Corral")
- MONGODB_COLLECTION   (default: "First-traces")

Output: corral_dump.json (JSON array of documents, MongoDB Extended JSON)
"""

import os
import sys
from pathlib import Path

from bson.json_util import dumps as bson_dumps  # handles ObjectId, datetime, etc.
from dotenv import load_dotenv
from loguru import logger
from pymongo import MongoClient

load_dotenv("../.env", override=True)


def main():
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

    # If the collection is not massive, collecting into a list is fine.
    # This will produce a single JSON array in the output file.
    docs = list(cursor)

    json_str = bson_dumps(docs, indent=4)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(json_str)

    logger.info(f"Done. Wrote {len(docs)} documents to {output_path}.")


if __name__ == "__main__":
    main()
