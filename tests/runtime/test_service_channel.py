"""The worker-to-controller service channel, over a real socket pair."""

from __future__ import annotations

import socket
import threading
import time

import pytest

from corral.runtime import service_channel
from corral.runtime.service_channel import (
    ServiceClient,
    ServiceError,
    ServiceServer,
    socket_pair,
)


@pytest.fixture
def channel():
    servers = []

    def connect(handler, **kwargs):
        controller, worker = socket_pair()
        server = ServiceServer(controller, handler, **kwargs).start()
        servers.append(server)
        return server, ServiceClient(worker)

    yield connect
    for server in servers:
        server.close()


def test_a_request_gets_its_own_answer(channel):
    _, client = channel(lambda body: {"echo": body["value"] * 2})
    assert client.call({"value": 21}) == {"echo": 42}


def test_concurrent_requests_are_answered_side_by_side(channel):
    inside = []
    gate = threading.Barrier(4, timeout=5)

    def handler(body):
        inside.append(body)
        gate.wait()  # every request must be in flight at once to pass
        return body

    _, client = channel(handler, workers=4)
    results = [None] * 4
    threads = [
        threading.Thread(target=lambda i=i: results.__setitem__(i, client.call(i)))
        for i in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert results == [0, 1, 2, 3]


def test_a_handler_error_reaches_the_worker_with_its_kind(channel):
    class BudgetExhausted(RuntimeError):
        pass

    def handler(_body):
        raise BudgetExhausted("no calls left")

    _, client = channel(handler)
    with pytest.raises(ServiceError, match="no calls left") as raised:
        client.call({})
    assert raised.value.kind == "BudgetExhausted"


def test_the_channel_keeps_working_after_an_error(channel):
    def handler(body):
        if body == "bad":
            raise ValueError("refused")
        return "fine"

    _, client = channel(handler)
    with pytest.raises(ServiceError):
        client.call("bad")
    assert client.call("good") == "fine"


def test_a_closed_controller_fails_pending_calls(channel):
    release = threading.Event()

    def handler(_body):
        release.wait(5)
        return "late"

    server, client = channel(handler)
    errors = []

    def call():
        try:
            client.call({})
        except ServiceError as exc:
            errors.append(exc)

    caller = threading.Thread(target=call)
    caller.start()
    time.sleep(0.2)
    server._sock.shutdown(socket.SHUT_RDWR)
    caller.join(timeout=5)
    release.set()
    assert errors
    assert "closed" in str(errors[0])
    with pytest.raises(ServiceError):
        client.call({})


def test_an_oversize_request_is_refused_before_sending(channel):
    _, client = channel(lambda body: body, max_message_bytes=1024)
    client._limit = 1024
    with pytest.raises(ServiceError, match="byte limit"):
        client.call("x" * 4096)


def test_a_process_without_a_channel_is_told_so():
    service_channel.set_channel(None)
    with pytest.raises(ServiceError, match="no service channel"):
        service_channel.call({})
