from pathlib import Path

import git
from loguru import logger

from corral.utils import create_vector_database


def clone_lammps_repo(target_dir: str = "lammps_repo") -> str:
    """
    Clone the LAMMPS repository if it doesn't exist already.

    Args:
        target_dir: Directory where the repo will be cloned

    Returns:
        Path to the cloned repository
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
        repo_path: Path to the LAMMPS repository

    Returns:
        List of paths to RST files
    """
    doc_src_path = Path(repo_path) / "doc" / "src"
    rst_files = list(doc_src_path.glob("*.rst"))

    logger.info(f"Found {len(rst_files)} RST files in {doc_src_path}")

    return [str(file) for file in rst_files]


def read_rst_files(rst_files: list[str]) -> list[dict[str, str]]:
    """
    Read content from RST files.

    Args:
        rst_files: List of paths to RST files

    Returns:
        List of dictionaries containing filename and content
    """
    docs = []

    for file_path in rst_files:
        path = Path(file_path)
        file_name = path.name

        try:
            with path.open(encoding="utf-8") as f:
                content = f.read()

            docs.append({"filename": file_name, "content": content})

            logger.info(f"Read {file_name} ({len(content)} characters)")

        except Exception as e:
            logger.info(f"Error reading {file_name}: {e}")

    return docs


def main():
    """Main function to execute the script."""
    repo_path = clone_lammps_repo()
    rst_files = get_rst_files(repo_path)
    docs = read_rst_files(rst_files)

    create_vector_database(docs, "lammps_manual", "../vector_databases/lammps_manual")

    return docs


if __name__ == "__main__":
    docs = main()
