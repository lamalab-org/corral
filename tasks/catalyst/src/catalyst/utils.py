import os
import re
from pathlib import Path


def extract_path_from_answer(answer: str) -> str:
    """Extract file path from agent answers"""
    if not isinstance(answer, str):
        return str(answer)

    answer = answer.strip()

    # Remove prefixes
    if answer.startswith("answer:"):
        answer = answer.replace("answer:", "", 1).strip()

    # Extract from markdown backticks: `path/file.json`
    markdown_match = re.search(r"`([^`]+\.[a-zA-Z0-9]+)`", answer)
    if markdown_match:
        return markdown_match.group(1)

    # Extract from quotes: "path/file.json" or 'path/file.json'
    quote_match = re.search(r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']', answer)
    if quote_match:
        return quote_match.group(1)

    # Extract absolute paths: /path/to/file.json
    abs_match = re.search(r"(/[^\s]+\.[a-zA-Z0-9]+)", answer)
    if abs_match:
        return abs_match.group(1)

    # Extract any file pattern: filename.json
    file_match = re.search(r"([^\s]+\.[a-zA-Z0-9]+)", answer)
    if file_match:
        return file_match.group(1)

    return answer


def find_file_by_name(filename: str, base_dir: str | None = None) -> str:
    """Find file by name in workspace"""
    if not base_dir:
        base_dir = os.environ.get("CORRAL_WORK_DIR", "")

    if not base_dir or not Path(base_dir).exists():
        return filename

    # Search for the file recursively
    base_path = Path(base_dir)
    matches = list(base_path.rglob(filename))

    if matches:
        # Return the most recent file
        return str(max(matches, key=lambda x: x.stat().st_mtime))

    return filename


def smart_resolve_path(input_path: str) -> str:
    """Resolve path intelligently - use absolute path if exists, search only as fallback"""
    extracted_path = extract_path_from_answer(input_path)

    if Path(extracted_path).exists():
        return extracted_path  # Use the extracted path directly
    else:
        return find_file_by_name(Path(extracted_path).name)  # Search as fallback
