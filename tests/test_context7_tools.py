"""Tests for Context7 documentation tools."""

import json
from unittest.mock import patch

import pytest

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


def test_get_library_documentation_error_handling():
    """Test error handling when Context7 API fails."""
    with patch("corral.utils.context7_tools.asyncio.run") as mock_run:
        mock_run.side_effect = RuntimeError("API connection failed")

        result = get_library_documentation.execute(
            package_name="invalid-package", tokens=5000
        )

        data = json.loads(result)

        assert data["success"] is False
        assert "error" in data
        assert "API connection failed" in data["error"]
        assert data["error_type"] == "RuntimeError"


@pytest.mark.integration()
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


@pytest.mark.integration()
def test_get_library_documentation_with_library_id():
    """Test using a known library ID directly."""
    result = get_library_documentation.execute(
        library_id="/vercel/next.js", topic="routing", tokens=3000, package_name=""
    )

    data = json.loads(result)

    if data["success"]:
        assert "text" in data
        assert data["library_id"] == "/vercel/next.js"
