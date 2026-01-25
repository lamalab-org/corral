import asyncio
import hashlib
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

from corral.backend.tool import tool

# Context7 remote MCP server URL
CONTEXT7_URL = "https://mcp.context7.com/mcp"

# Cache directory for storing fetched documentation
CACHE_DIR = Path(
    os.getenv("CONTEXT7_CACHE_DIR", Path.home() / ".cache" / "context7_mcp")
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(*parts: str) -> str:
    """
    Generate a cache key from multiple parts.

    Args:
        *parts (str): Variable number of strings to combine into cache key

    Returns:
        str: SHA256 hash of combined parts (truncated to 24 chars)
    """
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    """
    Get the file path for a cache key.

    Args:
        key (str): Cache key string

    Returns:
        Path: Full path to cache file
    """
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> dict[str, Any] | None:
    """
    Load cached data from disk.

    Args:
        key (str): Cache key to load

    Returns:
        dict | None: Cached data if exists, None otherwise
    """
    p = _cache_path(key)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load cache for key {key}: {e}")
            return None
    return None


def _save_cache(key: str, payload: dict[str, Any]) -> None:
    """
    Save data to cache on disk.

    Args:
        key (str): Cache key to save under
        payload (dict[str, Any]): Data to cache
    """
    try:
        with _cache_path(key).open("w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError as e:
        logger.warning(f"Failed to save cache for key {key}: {e}")


def _parse_text_results(text: str) -> list[dict]:
    results = []
    for block in text.split("----------"):
        lines = block.strip().splitlines()
        entry = {}
        for line in lines:
            if line.startswith("- Title:"):
                entry["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Context7-compatible library ID:"):
                entry["libraryId"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Code Snippets:"):
                val = line.split(":", 1)[1].strip()
                if val.isdigit():  # only parse numeric values
                    entry["codeSnippets"] = int(val)
            elif line.startswith("- Trust Score:"):
                val = line.split(":", 1)[1].strip()
                with suppress(ValueError):
                    entry["trustScore"] = float(val)
        if entry:
            results.append(entry)
    return results


async def _resolve_library_id(session: ClientSession, package_name: str) -> dict:
    """
    Call Context7's resolve-library-id tool and choose the best match.

    Selection policy mirrors Context7 guidance:
    - Exact name match preferred
    - Otherwise highest trust score and code snippet coverage

    Args:
        session (ClientSession): Active MCP client session
        package_name (str): Package name to resolve

    Returns:
        dict: Chosen library entry with id, versions, etc.

    Raises:
        RuntimeError: If no libraries found or resolution fails
    """
    logger.info(f"Resolving library ID for package: {package_name}")

    result = await session.call_tool(
        "resolve-library-id", {"libraryName": package_name}
    )

    # Handle structured content or text-based JSON responses
    data = None

    if getattr(result, "structuredContent", None):
        data = result.structuredContent
    elif result.content:
        for block in result.content:
            if isinstance(block, types.TextContent):
                data = {"results": _parse_text_results(block.text)}
                break

    if not data or not data.get("results"):
        raise RuntimeError(f"No libraries found for '{package_name}'")

    # Selection heuristic: exact name > relevance > snippets > trust
    results = data["results"]

    # 1) Prefer exact name match (case-insensitive)
    exact = [r for r in results if r.get("name", "").lower() == package_name.lower()]
    candidates = exact or results

    # 2) Sort by (snippets desc, trust desc) as tiebreakers
    def score(r):
        return (int(r.get("codeSnippets", 0)), float(r.get("trustScore", 0)))

    chosen = sorted(candidates, key=score, reverse=True)[0]

    logger.info(
        f"Selected library: {chosen.get('name')} (ID: {chosen.get('libraryId')})"
    )
    return chosen


async def _get_docs_text(
    session: ClientSession, library_id: str, topic: str, tokens: int
) -> str:
    """
    Call Context7's get-library-docs tool to fetch documentation text.

    Args:
        session (ClientSession): Active MCP client session
        library_id (str): Context7-compatible library ID (e.g., /org/project)
        topic (str): Optional topic to focus on
        tokens (int): Maximum tokens to retrieve (server enforces min/defaults)

    Returns:
        str: Documentation text

    Raises:
        RuntimeError: If no documentation content returned
    """
    logger.info(
        f"Fetching docs for {library_id} (topic: {topic or 'general'}, tokens: {tokens})"
    )

    args = {
        "context7CompatibleLibraryID": library_id,
        "topic": topic or "",
        "tokens": int(tokens),
    }
    result = await session.call_tool("get-library-docs", args)

    # Prefer structured content with text key
    if getattr(result, "structuredContent", None):
        sc = result.structuredContent
        if isinstance(sc, dict) and "text" in sc:
            return sc["text"]

    # Fallback to text blocks
    for block in result.content:
        if isinstance(block, types.TextContent):
            return block.text

    raise RuntimeError("No textual documentation content returned.")


@tool
def get_library_documentation(
    package_name: str,
    topic: str | None = None,
    tokens: int = 5000,
    library_id: str | None = None,
) -> str:
    """[BRIEF] Fetch up-to-date documentation for a library from Context7 MCP server. [/BRIEF]

    [DETAILED] This tool retrieves comprehensive, current documentation for software libraries and frameworks via Context7's MCP service.
    It automatically resolves package names to library IDs, caches documentation to avoid repeated network calls, and supports focused topic queries.
    Essential for accessing the latest API documentation, usage examples, and best practices for libraries not in the agent's training data.
    The tool maintains a local cache to improve performance and reduce API calls. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need current documentation for a library or framework
    - Best suited for looking up API references, usage patterns, and examples
    - Avoid if you already have sufficient knowledge from training data
    - Avoid for extremely obscure packages not indexed by Context7
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] You want to use some package but you are not 100% sure about its API or usage [/PREREQUISITE]
    2. [CURRENT] Fetch documentation for required library with optional topic focus [/CURRENT]
    3. [FOLLOW_UP] Use retrieved documentation to inform implementation decisions [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]


    [CONTEXTUAL] How this tool works:
    - Connects to Context7's public MCP server at https://mcp.context7.com/mcp
    - Resolves package names to Context7-compatible library IDs automatically
    - Caches fetched documentation locally to avoid redundant network requests
    - Supports optional topic filtering for focused documentation retrieval
    - Respects server-side token limits (minimum 1000, default 5000)
    - Returns comprehensive text documentation with usage examples
    - Uses optional API key from CONTEXT7_API_KEY environment variable for higher limits
    [/CONTEXTUAL]


    [SYNTACTICAL] Usage examples:
    [
        `get_library_documentation("nextjs", "routing", 5000, None)`,
        `get_library_documentation("react", "hooks", 3000, None)`,
        `get_library_documentation("", None, 5000, "/vercel/next.js/v14.2.5")`,
        `get_library_documentation("fastapi", "websockets", 8000, None)`,
        `get_library_documentation("pandas", "dataframe operations", 10000, None)`,
    ]
    [/SYNTACTICAL]

    Args:
        package_name (str):
            [BRIEF] Name of the package/library to fetch documentation for. [/BRIEF]
            [DETAILED] The name of the software library, framework, or package to retrieve documentation for.
            This will be resolved to a Context7-compatible library ID automatically using fuzzy matching and relevance scoring.
            If library_id is provided, this parameter is optional but still used for cache key generation. [/DETAILED]
            [SYNTACTIC] "valid package name string" [/SYNTACTIC]
            [EXAMPLES] "react", "nextjs", "fastapi", "pandas", "tensorflow", "django" [/EXAMPLES]

        topic (str):
            [BRIEF] Optional topic to focus documentation retrieval. Defaults to None. [/BRIEF]
            [DETAILED] An optional string specifying what aspect or feature of the library to focus on.
            When provided, the documentation returned will be more relevant to this specific topic.
            Leave as None for general comprehensive documentation. [/DETAILED]
            [SYNTACTIC] "descriptive topic string or None" [/SYNTACTIC]
            [EXAMPLES] "routing", "hooks", "authentication", "websockets", "dataframe operations", None [/EXAMPLES]

        tokens (int):
            [BRIEF] Maximum number of tokens to retrieve. Defaults to 5000. [/BRIEF]
            [DETAILED] The maximum number of tokens (roughly words) of documentation to retrieve from the server.
            Context7 enforces a minimum of 1000 and defaults to 5000.
            Higher values provide more comprehensive documentation but consume more tokens and bandwidth.
            Adjust based on how much context you need: 3000 for quick reference, 5000 for standard usage, 10000+ for deep dives. [/DETAILED]
            [SYNTACTIC] positive integer between 1000 and server maximum [/SYNTACTIC]
            [EXAMPLES] 3000, 5000 (default), 8000, 10000, 15000 [/EXAMPLES]

        library_id (str):
            [BRIEF] Optional Context7-compatible library ID to skip resolution. [/BRIEF]
            [DETAILED] An optional pre-determined Context7 library ID in the format '/org/project' or '/org/project/version'.
            When provided, skips the package name resolution step and fetches documentation directly.
            Useful when you already know the exact library ID or want to retrieve a specific version.
            If None, the tool will resolve package_name automatically. [/DETAILED]
            [SYNTACTIC] "/org/project[/version] format string or None" [/SYNTACTIC]
            [EXAMPLES] "/vercel/next.js", "/vercel/next.js/v14.2.5", "/mongodb/docs", None [/EXAMPLES]

    Returns:
        str:
            [BRIEF] JSON string containing documentation text and metadata. [/BRIEF]
            [DETAILED] A comprehensive JSON-formatted string containing the retrieved documentation text, cache status, and retrieval metadata.
            The documentation includes API references, usage examples, and best practices for the requested library.
            Includes 'cached' field indicating whether docs were retrieved from cache or freshly fetched.
            The 'text' field contains the actual documentation content ready for use. [/DETAILED]
            [EXAMPLES] '{"success": true, "text": "Next.js routing documentation...", "cached": false, "library_id": "/vercel/next.js", "tokens": 5000}' [/EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
            [ERROR_WHEN] When library cannot be resolved or documentation cannot be retrieved [/ERROR_WHEN]
            [ERROR_DETAILS] Network errors, invalid library names, or server unavailability [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify package name spelling [/ERROR_RECOVERY]

        ConnectionError:
            [ERROR_WHEN] When unable to connect to Context7 MCP server [/ERROR_WHEN]
            [ERROR_DETAILS] Network connectivity issues or server downtime [/ERROR_DETAILS]
            [ERROR_RECOVERY] You have to try another tool [/ERROR_RECOVERY]

        ValueError:
            [ERROR_WHEN] When invalid parameters are provided (e.g., negative tokens) [/ERROR_WHEN]
            [ERROR_DETAILS] Invalid parameter values or types [/ERROR_DETAILS]
            [ERROR_RECOVERY] Validate parameter values and types according to specifications [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Requires active internet connection to Context7 MCP server
    - Documentation quality depends on Context7's indexing of the library
    - Very new or obscure packages may not be indexed
    - Token limits are enforced server-side (minimum 1000, defaults enforced)
    - Cache is stored locally and not shared across different machines
    - Rate limits may apply without API key authentication
    - Specific version selection requires knowing the exact library ID format
    [/LIMITATIONS]
    """
    logger.info(
        f"Getting documentation for package: {package_name}, topic: {topic}, tokens: {tokens}"
    )

    try:
        # Generate cache key based on inputs
        cache_key = _cache_key(
            package_name or "", topic or "", library_id or "", str(tokens)
        )

        # Check cache first
        cached = _load_cache(cache_key)
        if cached and "text" in cached:
            logger.info(f"Documentation found in cache (key: {cache_key})")
            return json.dumps(
                {
                    "success": True,
                    "text": cached["text"],
                    "cached": True,
                    "library_id": cached.get("library_id"),
                    "tokens": tokens,
                }
            )

        # Fetch from Context7 MCP server
        api_key = os.getenv("CONTEXT7_API_KEY", "")
        headers = {}
        if api_key:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "CONTEXT7_API_KEY": api_key,
            }

        async def _fetch() -> tuple[str, str]:
            """Async helper to fetch documentation."""
            async with (
                streamablehttp_client(CONTEXT7_URL, headers=headers) as (
                    read,
                    write,
                    _,
                ),
                ClientSession(read, write) as session,
            ):
                await session.initialize()

                # Resolve library ID if not provided
                lib_id = library_id
                if not lib_id:
                    chosen = await _resolve_library_id(session, package_name)
                    lib_id = (
                        chosen.get("libraryId")
                        or chosen.get("id")
                        or chosen.get("libraryID")
                    )
                    if not lib_id:
                        raise RuntimeError(
                            "Could not determine Context7 library ID from resolve results."
                        )

                # Fetch documentation text
                text = await _get_docs_text(session, lib_id, topic or "", tokens)
                return text, lib_id

        # Execute async fetch
        # text, resolved_lib_id = asyncio.run(_fetch())
        try:
            text, resolved_lib_id = asyncio.run(_fetch())
        except Exception as e:
            # Handle both ExceptionGroup (Python 3.11+) and regular exceptions
            if hasattr(e, "exceptions"):
                for sub in e.exceptions:
                    logger.exception(f"Sub-exception in TaskGroup: {sub}")
            else:
                logger.exception(f"Regular exception: {e}")
            raise

        # Cache the result
        _save_cache(cache_key, {"text": text, "library_id": resolved_lib_id})

        logger.info(
            f"Successfully fetched documentation (length: {len(text)} chars, cached with key: {cache_key})"
        )

        return json.dumps(
            {
                "success": True,
                "text": text,
                "cached": False,
                "library_id": resolved_lib_id,
                "tokens": tokens,
            }
        )

    except Exception as e:
        logger.error(f"Failed to get library documentation: {e}")
        return json.dumps(
            {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "package_name": package_name,
            }
        )
