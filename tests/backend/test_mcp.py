"""Exercise bound tool catalogs and MCP backend cleanup over real HTTP."""

import asyncio
import signal
import threading
from contextlib import asynccontextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from corral.backend import mcp as transport_module
from corral.backend.mcp import open_mcp_host
from corral.core.tool import ToolResponse
from corral.core.tool_catalog import ToolCatalogSnapshot


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def make_binding(name):
    return SimpleNamespace(
        execution_id=name,
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "measure",
                    "description": "Measure a sample.",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample": {"type": "string"}},
                        "required": ["sample"],
                    },
                },
            },
        ),
        execute=AsyncMock(
            return_value=ToolResponse(success=True, result=name, error=None)
        ),
    )


def bind_tools(host, binding):
    return host.bind(
        catalog=ToolCatalogSnapshot.capture(binding.tools),
        execute=binding.execute,
        name=binding.execution_id,
    )


@asynccontextmanager
async def serve_tools(binding):
    async with open_mcp_host() as host, bind_tools(host, binding) as connection:
        yield connection


@pytest.mark.anyio()
async def test_endpoints_route_tools_to_their_own_sessions():
    first, second = make_binding("first"), make_binding("second")
    async with (
        open_mcp_host() as host,
        bind_tools(host, first) as first_endpoint,
        bind_tools(host, second) as second_endpoint,
    ):
        assert first_endpoint.url != second_endpoint.url
        assert httpx.URL(first_endpoint.url).port == httpx.URL(second_endpoint.url).port
        for session, endpoint in ((first, first_endpoint), (second, second_endpoint)):
            async with (
                streamablehttp_client(endpoint.url) as streams,
                ClientSession(streams[0], streams[1]) as client,
            ):
                await client.initialize()
                tools = await client.list_tools()
                assert [tool.name for tool in tools.tools] == ["measure"]
                assert (
                    tools.tools[0].inputSchema
                    == session.tools[0]["function"]["parameters"]
                )
                result = await client.call_tool("measure", {"sample": "a"})
                assert result.isError is False
                assert result.content[0].text == session.execution_id
                action = session.execute.await_args.args[0]
                assert action.name == "measure"
                assert action.arguments == {"sample": "a"}

        first.execute.assert_awaited_once()
        second.execute.assert_awaited_once()
        async with httpx.AsyncClient() as client:
            wrong_url = first_endpoint.url.rsplit("/", 2)[0] + "/unknown/mcp"
            assert (await client.post(wrong_url, json={})).status_code == 404

    async with httpx.AsyncClient() as client:
        for endpoint in (first_endpoint, second_endpoint):
            with pytest.raises(httpx.ConnectError):
                await client.post(endpoint.url, json={})


@pytest.mark.anyio()
async def test_endpoint_preserves_tool_errors():
    session = make_binding("failed-tool")
    session.execute.return_value = ToolResponse(
        success=False, result=None, error="sample is missing"
    )
    async with (
        serve_tools(session) as endpoint,
        streamablehttp_client(endpoint.url) as streams,
        ClientSession(streams[0], streams[1]) as client,
    ):
        await client.initialize()
        result = await client.call_tool("measure", {"sample": "missing"})
        assert result.isError is True
        assert result.content[0].text == "sample is missing"


@pytest.mark.anyio()
async def test_endpoint_closes_when_the_harness_is_cancelled():
    with anyio.CancelScope() as scope:
        async with serve_tools(make_binding("cancelled")) as endpoint:
            scope.cancel()
            await anyio.lowlevel.checkpoint()

    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.ConnectError):
            await client.post(endpoint.url, json={})


@pytest.mark.anyio()
async def test_failed_startup_releases_the_listener(monkeypatch):
    listeners = []

    async def fail_start(server, sockets=None):
        listeners.extend(sockets)
        raise RuntimeError("startup failed")

    monkeypatch.setattr(transport_module._EmbeddedServer, "startup", fail_start)
    with pytest.raises(RuntimeError, match="startup failed"):
        async with serve_tools(make_binding("failed-startup")):
            pytest.fail("a failed endpoint must not reach the harness")

    assert len(listeners) == 1
    assert listeners[0].fileno() == -1


async def post_tool(client, url):
    return await client.post(
        url,
        headers={"Accept": "application/json, text/event-stream"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "measure", "arguments": {"sample": "a"}},
        },
    )


@pytest.mark.anyio()
async def test_concurrent_first_bindings_start_one_listener(monkeypatch):
    listeners = []
    startup = transport_module._EmbeddedServer.startup

    async def track_start(server, sockets=None):
        listeners.extend(sockets)
        await startup(server, sockets)

    monkeypatch.setattr(transport_module._EmbeddedServer, "startup", track_start)
    async with open_mcp_host() as host, httpx.AsyncClient() as client:
        assert not listeners

        async def invoke(name):
            async with bind_tools(host, make_binding(name)) as connection:
                response = await post_tool(client, connection.url)
                assert response.json()["result"]["content"][0]["text"] == name
                return connection.url

        urls = await asyncio.gather(invoke("first"), invoke("second"))
        assert len(listeners) == 1
        assert urls[0] != urls[1]
        assert {httpx.URL(url).port for url in urls} == {host.port}

    assert listeners[0].fileno() == -1


@pytest.mark.anyio()
@pytest.mark.parametrize("cancel_startup", [False, True])
async def test_first_binding_can_retry_after_startup_failure(
    monkeypatch, cancel_startup
):
    started = asyncio.Event()
    listeners = []
    startup = transport_module._EmbeddedServer.startup

    async def fail_start(server, sockets=None):
        listeners.extend(sockets)
        started.set()
        if cancel_startup:
            await anyio.sleep_forever()
        raise RuntimeError("startup failed")

    monkeypatch.setattr(transport_module._EmbeddedServer, "startup", fail_start)
    async with open_mcp_host() as host:

        async def first_binding():
            async with bind_tools(host, make_binding("first")):
                pytest.fail("startup did not succeed")

        owner = asyncio.create_task(first_binding())
        try:
            with anyio.fail_after(5):
                await started.wait()
            if cancel_startup:
                owner.cancel()
            expected_error = asyncio.CancelledError if cancel_startup else RuntimeError
            with pytest.raises(expected_error):
                await owner
        finally:
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

        assert listeners[0].fileno() == -1
        assert host.port is None
        monkeypatch.setattr(transport_module._EmbeddedServer, "startup", startup)
        async with (
            bind_tools(host, make_binding("retry")) as connection,
            httpx.AsyncClient() as client,
        ):
            response = await post_tool(client, connection.url)
            assert response.json()["result"]["content"][0]["text"] == "retry"

    with pytest.raises(RuntimeError, match="host is closed"):
        async with bind_tools(host, make_binding("too-late")):
            pytest.fail("a closed host cannot accept bindings")


@pytest.mark.anyio()
async def test_host_preserves_signal_handlers_and_invocation_context():
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    context = ContextVar("invocation", default="outside")
    loop = asyncio.get_running_loop()
    binding = make_binding("context")

    async def execute(action):
        assert asyncio.get_running_loop() is loop
        return ToolResponse(success=True, result=context.get(), error=None)

    binding.execute = execute
    async with open_mcp_host() as host:
        assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
        token = context.set("delegate")
        async with bind_tools(host, binding) as endpoint:
            context.reset(token)
            async with httpx.AsyncClient() as client:
                response = await post_tool(client, endpoint.url)
                assert response.json()["result"]["content"][0]["text"] == "delegate"
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers


@pytest.mark.anyio()
async def test_binding_revokes_new_requests_and_drains_accepted_responses():
    started, release, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    endpoint_ready = asyncio.get_running_loop().create_future()
    binding = make_binding("draining")

    async def execute(action):
        started.set()
        await release.wait()
        return ToolResponse(success=True, result="finished", error=None)

    binding.execute = execute
    async with open_mcp_host() as host, httpx.AsyncClient() as client:

        async def invocation():
            async with bind_tools(host, binding) as endpoint:
                endpoint_ready.set_result(endpoint)
                await finish.wait()

        owner = asyncio.create_task(invocation())
        endpoint = await endpoint_ready
        request = asyncio.create_task(post_tool(client, endpoint.url))
        try:
            with anyio.fail_after(5):
                await started.wait()
                finish.set()
                # The owner revokes the route before waiting for the tool.
                await asyncio.sleep(0)
                assert (await client.post(endpoint.url, json={})).status_code == 404
                assert not owner.done()
                release.set()
                response = await request
                assert response.json()["result"]["content"][0]["text"] == "finished"
                await owner
        finally:
            release.set()
            finish.set()
            await asyncio.gather(owner, request, return_exceptions=True)


@pytest.mark.anyio()
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("cancel_while_draining", [False, True])
async def test_cancellation_waits_for_accepted_tool_cleanup(
    synchronous, cancel_while_draining
):
    started, cleaned, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release = threading.Event()
    endpoint_ready = asyncio.get_running_loop().create_future()
    loop = asyncio.get_running_loop()
    binding = make_binding("cancel-tool")

    def sync_work():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        loop.call_soon_threadsafe(cleaned.set)

    async def execute(action):
        if synchronous:
            await anyio.to_thread.run_sync(sync_work)
        else:
            started.set()
            try:
                await anyio.sleep_forever()
            finally:
                with anyio.CancelScope(shield=True):
                    await asyncio.sleep(0)
                    cleaned.set()
        return ToolResponse(success=True, result="finished", error=None)

    binding.execute = execute
    async with open_mcp_host() as host, httpx.AsyncClient() as client:

        async def invocation():
            async with bind_tools(host, binding) as endpoint:
                endpoint_ready.set_result(endpoint)
                await finish.wait()

        owner = asyncio.create_task(invocation())
        endpoint = await endpoint_ready
        request = asyncio.create_task(post_tool(client, endpoint.url))
        try:
            with anyio.fail_after(5):
                await started.wait()
                if cancel_while_draining:
                    finish.set()
                    await asyncio.sleep(0)
                owner.cancel()
                await asyncio.sleep(0)
                assert (await client.post(endpoint.url, json={})).status_code == 404
                if synchronous:
                    assert not cleaned.is_set()
                    assert not owner.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await owner
                assert cleaned.is_set()
                await request
        finally:
            release.set()
            owner.cancel()
            await asyncio.gather(owner, request, return_exceptions=True)


@pytest.mark.anyio()
async def test_partial_startup_failure_leaves_no_server_or_listener(monkeypatch):
    servers, listeners = [], []
    startup = transport_module._EmbeddedServer.startup

    async def fail_after_start(server, sockets=None):
        servers.append(server)
        listeners.extend(sockets)
        await startup(server, sockets)
        raise RuntimeError("failed after listening")

    monkeypatch.setattr(transport_module._EmbeddedServer, "startup", fail_after_start)
    with pytest.raises(RuntimeError, match="failed after listening"):
        async with serve_tools(make_binding("partial-startup")):
            pytest.fail("startup failed")
    assert all(sock.fileno() == -1 for sock in listeners)
    assert all(not server.is_serving() for server in servers[0].servers)
    assert not any(
        task.get_name().startswith("corral-mcp-host-") for task in asyncio.all_tasks()
    )


@pytest.mark.anyio()
async def test_startup_cancellation_leaves_no_task_or_socket(monkeypatch):
    listeners = []
    started = asyncio.Event()

    async def stalled_start(server, sockets=None):
        listeners.extend(sockets)
        started.set()
        await anyio.sleep_forever()

    async def invocation():
        async with serve_tools(make_binding("cancelled-startup")):
            pytest.fail("startup was cancelled")

    monkeypatch.setattr(transport_module._EmbeddedServer, "startup", stalled_start)
    owner = asyncio.create_task(invocation())
    try:
        with anyio.fail_after(5):
            await started.wait()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
    finally:
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
    assert listeners[0].fileno() == -1
    assert not any(
        task.get_name().startswith("corral-mcp-host-") for task in asyncio.all_tasks()
    )
