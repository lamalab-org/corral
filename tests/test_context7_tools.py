"""Tests for Context7 documentation tools."""

import json
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from exceptiongroup import ExceptionGroup
from mcp import types

from corral.utils import context7_tools
from corral.utils.context7_tools import (
    _cache_key,
    _load_cache,
    _save_cache,
    get_library_documentation,
)


def test_cache_key_generation():
    """Test that cache keys are generated consistently."""
    key1 = _cache_key("react", "hooks", "", "5000")
    key2 = _cache_key("react", "hooks", "", "5000")
    key3 = _cache_key("react", "routing", "", "5000")

    # Same inputs should produce same key
    assert key1 == key2

    # Different inputs should produce different keys
    assert key1 != key3

    # Keys should be 24 characters (truncated SHA256)
    assert len(key1) == 24


def test_cache_operations(tmp_path):
    """Test cache save and load operations."""
    # Use a temporary cache directory
    with patch("corral.utils.context7_tools.CACHE_DIR", tmp_path):
        test_key = "test_key_123"
        test_data = {"text": "Sample documentation", "library_id": "/test/lib"}

        # Save to cache
        _save_cache(test_key, test_data)

        # Load from cache
        loaded_data = _load_cache(test_key)

        assert loaded_data is not None
        assert loaded_data["text"] == test_data["text"]
        assert loaded_data["library_id"] == test_data["library_id"]


def test_cache_load_nonexistent():
    """Test loading from cache when file doesn't exist."""
    result = _load_cache("nonexistent_key_xyz")
    assert result is None


def test_get_library_documentation_cached(tmp_path):
    """Test that cached documentation is returned without API calls."""
    with patch("corral.utils.context7_tools.CACHE_DIR", tmp_path):
        # Pre-populate cache
        test_text = "Cached React hooks documentation"
        cache_data = {"text": test_text, "library_id": "/facebook/react"}
        test_key = _cache_key("react", "hooks", "", "5000")
        _save_cache(test_key, cache_data)

        # Call the tool's execute method
        result = get_library_documentation.execute(
            package_name="react", topic="hooks", tokens=5000
        )

        # Parse result
        data = json.loads(result)

        assert data["success"] is True
        assert data["text"] == test_text
        assert data["cached"] is True
        assert data["library_id"] == "/facebook/react"


@pytest.mark.parametrize("wrapped", [False, True])
def test_get_library_documentation_error_handling(wrapped):
    """Test error handling when Context7 API fails."""

    def fail_fetch(coroutine):
        coroutine.close()
        error = RuntimeError("API connection failed")
        if wrapped:
            raise ExceptionGroup("unhandled errors in a TaskGroup", [error])
        raise error

    with patch("corral.utils.context7_tools.asyncio.run", side_effect=fail_fetch):
        result = get_library_documentation.execute(
            package_name="invalid-package", tokens=5000
        )

        data = json.loads(result)

        assert data["success"] is False
        assert "error" in data
        assert "API connection failed" in data["error"]
        assert data["error_type"] == "RuntimeError"


@pytest.mark.parametrize("library_id", [None, "/acesuit/mace"])
@pytest.mark.parametrize("topic", [None, "ASE MACECalculator stress"])
def test_current_context7_api(tmp_path, monkeypatch, library_id, topic):
    calls = []

    @asynccontextmanager
    async def transport(*args, **kwargs):
        yield None, None, None

    class Session:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            assert isinstance(arguments.get("query"), str)
            assert arguments["query"]
            if topic:
                assert arguments["query"] == topic
            if name == "resolve-library-id":
                assert set(arguments) == {"libraryName", "query"}
                assert arguments["libraryName"] == "mace-torch"
                text = (
                    "- Title: MACE\n"
                    "- Context7-compatible library ID: /acesuit/mace\n"
                    "- Code Snippets: 123\n"
                )
            else:
                assert name == "query-docs"
                assert set(arguments) == {"libraryId", "query"}
                assert arguments["libraryId"] == "/acesuit/mace"
                text = "MACECalculator documentation"
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)]
            )

    monkeypatch.setattr(context7_tools, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(context7_tools, "streamablehttp_client", transport)
    monkeypatch.setattr(context7_tools, "ClientSession", Session)
    result = json.loads(
        get_library_documentation.execute(
            package_name="mace-torch",
            topic=topic,
            tokens=3000,
            library_id=library_id,
        )
    )
    assert result == {
        "success": True,
        "text": "MACECalculator documentation",
        "cached": False,
        "library_id": "/acesuit/mace",
        "tokens": 3000,
    }
    assert [name for name, _ in calls] == (
        ["query-docs"] if library_id else ["resolve-library-id", "query-docs"]
    )


@pytest.mark.parametrize("operation", ["resolve", "docs"])
def test_context7_error_is_not_documentation(operation):
    import asyncio

    class Session:
        async def call_tool(self, name, arguments):
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text="service unavailable")],
            )

    async def fetch():
        if operation == "resolve":
            return await context7_tools._resolve_library_id(Session(), "mace-torch")
        return await context7_tools._get_docs_text(
            Session(), "/acesuit/mace", "stress", 3000
        )

    with pytest.raises(RuntimeError, match="service unavailable"):
        asyncio.run(fetch())


@pytest.mark.integration
def test_get_library_documentation_integration():
    """
    Integration test - actually calls Context7 API.
    Requires internet connection and working Context7 service.
    """
    result = get_library_documentation.execute(
        package_name="react", topic="hooks", tokens=3000
    )

    data = json.loads(result)

    # Should succeed (unless API is down)
    if data["success"]:
        assert "text" in data
        assert len(data["text"]) > 0
        assert "library_id" in data
        assert data["tokens"] == 3000
    else:
        # If it fails, should have error info
        assert "error" in data


@pytest.mark.integration
def test_get_library_documentation_with_library_id():
    """Test using a known library ID directly."""
    result = get_library_documentation.execute(
        library_id="/vercel/next.js", topic="routing", tokens=3000, package_name=""
    )

    data = json.loads(result)

    if data["success"]:
        assert "text" in data
        assert data["library_id"] == "/vercel/next.js"
