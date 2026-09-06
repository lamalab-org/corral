"""Tests for authoritative State-owned tool catalog snapshots."""

import pytest
from pydantic import ValidationError

from corral.core.tool_catalog import (
    ToolCatalogSnapshot,
    tool_catalog_fingerprint,
)


def _tool(name: str, description: str = "description") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }


def test_catalog_fingerprint_ignores_presentation_order_only():
    first = _tool("first")
    second = _tool("second")

    assert tool_catalog_fingerprint([first, second]) == tool_catalog_fingerprint(
        [second, first]
    )
    assert tool_catalog_fingerprint([first, second]) != tool_catalog_fingerprint(
        [first, _tool("second", "changed")]
    )


def test_snapshot_rejects_a_fingerprint_that_does_not_match_its_tools():
    snapshot = ToolCatalogSnapshot.capture([_tool("first")])
    payload = snapshot.model_dump(mode="json")
    payload["tools"][0]["function"]["description"] = "tampered"

    with pytest.raises(ValidationError, match="does not match"):
        ToolCatalogSnapshot.model_validate(payload)


def test_catalog_rejects_duplicate_tool_names():
    with pytest.raises(ValueError, match="duplicate tool"):
        ToolCatalogSnapshot.capture([_tool("duplicate"), _tool("duplicate")])


def test_mcp_description_preserves_the_catalog_schema_and_is_detached():
    definition = _tool("measure", "Measure a sample")
    definition["function"]["parameters"] = {
        "type": "object",
        "properties": {"sample": {"type": "string", "enum": ["a", "b"]}},
        "required": ["sample"],
        "additionalProperties": False,
    }
    catalog = ToolCatalogSnapshot.capture([definition])
    converted = catalog.mcp_tools()
    assert converted == (
        {
            "name": "measure",
            "description": "Measure a sample",
            "inputSchema": definition["function"]["parameters"],
        },
    )
    converted[0]["inputSchema"]["properties"]["sample"]["enum"].append("c")
    assert catalog.detached_tools() == (definition,)
