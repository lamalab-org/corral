import os
import re
from pathlib import Path

from litellm import embedding
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from corral.report.logging import logger


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

    # Search for the file recursively. A symlink is never a valid workspace
    # file, even if its current target happens to remain below `base_dir`.
    from corral.workspace import confine_workspace_path

    base_path = Path(base_dir)
    matches = []
    for match in base_path.rglob(filename):
        try:
            confined = confine_workspace_path(base_path, match)
        except ValueError:
            continue
        if confined.is_file():
            matches.append(confined)

    if matches:
        # Return the most recent file
        return str(max(matches, key=lambda x: x.stat().st_mtime))

    return filename


def smart_resolve_path(input_path: str, base_dir: str | None = None) -> str:
    """Resolve a path, confining task submissions to `base_dir` when supplied.

    With an explicit `base_dir`, both direct paths and fallback searches are
    restricted to that one task workspace. Existing absolute sibling paths,
    `..` traversal, and symlink aliases are rejected instead of being passed
    to a scorer. Without `base_dir` the legacy process-global lookup remains
    available for non-runtime callers.
    """
    extracted_path = extract_path_from_answer(input_path)

    if base_dir is not None:
        from corral.workspace import confine_workspace_path

        candidate = confine_workspace_path(base_dir, extracted_path)
        if candidate.is_file():
            return str(candidate)

        # Preserve the historical basename fallback, but search this execution
        # root only. A missing result remains rooted inside the workspace so a
        # relative process cwd can never redirect the scorer elsewhere.
        found = find_file_by_name(Path(extracted_path).name, base_dir)
        if Path(found).is_absolute():
            return found
        return str(confine_workspace_path(base_dir, Path(extracted_path).name))

    if Path(extracted_path).exists():
        return extracted_path  # Use the extracted path directly
    else:
        return find_file_by_name(Path(extracted_path).name)


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
        raise ValueError("Input must be a non-empty list of strings")

    logger.debug(
        f"Embedding {len(chunks)} {'chemical' if chemical else 'text'} chunks using model: {model}"
    )

    BATCH_SIZE = 2048  # This is a configuration from LiteLLM:
    # https://docs.litellm.ai/docs/embedding/supported_embedding#required-fields

    # Process in batches if the input is large
    if len(chunks) > BATCH_SIZE:
        logger.debug(f"Input size exceeds {BATCH_SIZE} chunks, processing in batches")
        all_embeddings = []

        # Process chunks in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            logger.debug("Processing batched chunks")

            try:
                batch_embeddings = embed_text(batch, model=model, chemical=chemical)
                all_embeddings.extend(batch_embeddings)
            except Exception:
                raise

        logger.debug(
            f"Successfully generated {len(all_embeddings)} embeddings across all batches"
        )
        return all_embeddings

    # For chemical embeddings, use MoLFormer
    if chemical:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            logger.debug("Using MoLFormer model for chemical embeddings")

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

            logger.debug(
                f"Successfully generated {len(embeddings)} chemical embeddings"
            )
            return embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to generate chemical embeddings: {e!s}") from e

    # For text embeddings, use litellm as before
    try:
        result_embeddings = embedding(
            model=model,
            input=chunks,
        )
        logger.debug(
            f"Successfully generated {len(result_embeddings['data'])} embeddings"
        )
        return [item["embedding"] for item in result_embeddings["data"]]
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {e!s}") from e
