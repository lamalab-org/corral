"""Disconnects must not abandon accepted MCP work or hang incomplete requests."""

import asyncio
import json
from unittest.mock import AsyncMock

import anyio
import pytest

from corral.backend.mcp import _Binding
from corral.core.tool import ToolResponse
from corral.core.tool_catalog import ToolCatalogSnapshot


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def catalog():
    return ToolCatalogSnapshot.capture(
        [
            {
                "type": "function",
                "function": {
                    "name": "measure",
                    "description": "Measure a sample.",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )


@pytest.fixture
def request_scope():
    return {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"host", b"localhost"),
            (b"content-type", b"application/json"),
            (b"accept", b"application/json, text/event-stream"),
        ],
    }


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_owner", [False, True])
async def test_disconnect_preserves_accepted_work(catalog, request_scope, cancel_owner):
    started, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    disconnected, disconnect_received = asyncio.Event(), asyncio.Event()
    completed = False

    async def execute(action):
        nonlocal completed
        started.set()
        try:
            await release.wait()
            completed = True
            return ToolResponse(success=True, result="finished", error=None)
        finally:
            with anyio.CancelScope(shield=True):
                await asyncio.sleep(0)
                cleaned.set()

    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "measure",
                "arguments": {},
                "_meta": {"progressToken": 0},
            },
        }
    ).encode()
    body_received = False

    async def receive():
        nonlocal body_received
        if not body_received:
            body_received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        disconnect_received.set()
        return {"type": "http.disconnect"}

    # Uvicorn ignores outgoing data once a client has disconnected.
    send = AsyncMock()
    binding = _Binding(catalog, execute, "disconnect")
    request = asyncio.create_task(binding(request_scope, receive, send))
    try:
        with anyio.fail_after(5):
            await started.wait()
            disconnected.set()
            await disconnect_received.wait()
            with anyio.move_on_after(0.1) as disconnect_wait:
                await asyncio.shield(request)
            assert disconnect_wait.cancel_called
            assert not request.done()
            assert not cleaned.is_set()
            if cancel_owner:
                await binding.close(cancel=True)
            else:
                release.set()
            await request
            assert cleaned.is_set()
            assert completed is not cancel_owner
    finally:
        release.set()
        await binding.close(cancel=True)
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.anyio
async def test_incomplete_body_disconnect_aborts_request(catalog, request_scope):
    execute = AsyncMock()
    binding = _Binding(catalog, execute, "incomplete-body")
    receive = AsyncMock(
        side_effect=[
            {"type": "http.request", "body": b'{"jsonrpc":', "more_body": True},
            {"type": "http.disconnect"},
        ]
    )
    try:
        with anyio.fail_after(5):
            await binding(request_scope, receive, AsyncMock())
        execute.assert_not_awaited()
    finally:
        await binding.close(cancel=True)


@pytest.mark.anyio
async def test_standalone_get_stream_closes_on_disconnect(catalog, request_scope):
    execute = AsyncMock()
    binding = _Binding(catalog, execute, "standalone-stream")
    receive = AsyncMock(
        side_effect=[
            {"type": "http.request", "body": b"", "more_body": False},
            {"type": "http.disconnect"},
        ]
    )
    send = AsyncMock()
    try:
        with anyio.fail_after(5):
            await binding({**request_scope, "method": "GET"}, receive, send)
        execute.assert_not_awaited()
        response = send.await_args_list[0].args[0]
        assert response["type"] == "http.response.start"
        assert response["status"] == 200
        assert any(
            name == b"content-type" and b"text/event-stream" in value
            for name, value in response["headers"]
        )
    finally:
        await binding.close(cancel=True)
