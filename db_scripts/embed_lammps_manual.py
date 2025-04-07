import gc
import shutil
from collections.abc import Generator
from pathlib import Path

import git
from dotenv import load_dotenv
from loguru import logger

from corral.utils import create_vector_database

load_dotenv("../.env", override=True)


def clone_lammps_repo(target_dir: str = "lammps_repo") -> str:
    """
    Clone the LAMMPS repository if it doesn't exist already.

    Args:
        target_dir (str): Directory where the repo will be cloned. Default is "lammps_repo".

    Returns:
        str: Path to the cloned repository
    """
    repo_url = "https://github.com/lammps/lammps"
    repo_path = Path(target_dir).resolve()

    if not Path(repo_path).exists():
        logger.info(f"Cloning LAMMPS repository to {repo_path}...")
        git.Repo.clone_from(repo_url, repo_path)
        logger.info("Repository cloned successfully!")
    else:
        logger.info(f"Repository already exists at {repo_path}, skipping clone.")

    return str(repo_path)


def get_rst_files(repo_path: str) -> list[str]:
    """
    Get all RST files in the doc/src directory.

    Args:
        repo_path (str): Path to the LAMMPS repository

    Returns:
        list[str]: List of paths to RST files
    """
    doc_src_path = Path(repo_path) / "doc" / "src"
    rst_files = list(doc_src_path.glob("*.rst"))

    logger.info(f"Found {len(rst_files)} RST files in {doc_src_path}")

    return [str(file) for file in rst_files]


def read_rst_files_in_batches(
    rst_files: list[str], batch_size: int = 5
) -> Generator[list[str], None, None]:
    """
    Read content from RST files in batches.

    Args:
        rst_files (list[str]): List of paths to RST files
        batch_size (int, optional): Number of files to process in each batch. Default is 5.

    Returns:
        Generator[list[str], None, None]: A generator yielding batches of file contents
    """
    total_files = len(rst_files)
    logger.info(f"Processing {total_files} files in batches of {batch_size}")

    for i in range(0, total_files, batch_size):
        batch_files = rst_files[i : i + batch_size]
        docs = []

        for file_path in batch_files:
            path = Path(file_path)
            file_name = path.name

            try:
                with path.open(encoding="utf-8") as f:
                    content = f.read()

                docs.append(str({"filename": file_name, "content": content}))
                logger.info(f"Read {file_name} ({len(content)} characters)")

            except Exception as e:
                logger.error(f"Error reading {file_name}: {e}")

        logger.info(
            f"Yielding batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size} with {len(docs)} documents"
        )
        yield docs


def cleanup_repo(repo_path: str) -> None:
    """
    Remove the cloned repository to free up disk space.

    Args:
        repo_path (str): Path to the repository to be removed
    """
    path = Path(repo_path)
    if path.exists():
        logger.info(f"Cleaning up: removing repository at {repo_path}")
        shutil.rmtree(repo_path)
        logger.info("Repository removed successfully")
    else:
        logger.info(f"Repository at {repo_path} doesn't exist, nothing to clean up")


def create_vector_database_incrementally(
    rst_files: list[str], collection_name: str, db_path: str, batch_size: int = 5
) -> None:
    """
    Create a vector database incrementally by processing batches of files.

    Args:
        rst_files (list[str]): List of paths to RST files
        collection_name (str): Name for the vector database collection
        db_path (str): Path where to store the vector database
        batch_size (int, optional): Size of each batch of files to process. Default is 5.
    """
    total_docs = 0

    for i, docs_batch in enumerate(read_rst_files_in_batches(rst_files, batch_size)):
        if not docs_batch:
            continue

        try:
            update_mode = "recreate" if i == 0 else "append"

            with logger.contextualize(batch=i + 1):
                logger.info(
                    f"Processing batch {i+1} with {len(docs_batch)} documents using update_mode='{update_mode}'"
                )

                create_vector_database(
                    chunks=docs_batch,
                    collection_name=collection_name,
                    path=db_path,
                    update_mode=update_mode,
                )

            total_docs += len(docs_batch)
            logger.info(
                f"Processed batch {i+1} with {len(docs_batch)} documents. Total processed: {total_docs}"
            )

            gc.collect()

        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {e}")

    logger.info(f"Finished creating vector database with {total_docs} total documents")


def main():
    """Main function to execute the script."""
    repo_path = clone_lammps_repo()
    rst_files = get_rst_files(repo_path)

    create_vector_database_incrementally(
        rst_files, "lammps_manual", "../vector_databases/lammps_manual", batch_size=10
    )

    cleanup_repo(repo_path)


if __name__ == "__main__":
    main()
