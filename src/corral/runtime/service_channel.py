"""Requests from a restricted worker, answered by a handler in the controller.

Newline-delimited JSON over a Unix socket pair. Requests may overlap; replies
carry the request id, and errors carry the exception's class name as ``kind``.
"""

from __future__ import annotations

import itertools
import json
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DEFAULT_MAX_MESSAGE_BYTES",
    "ServiceClient",
    "ServiceError",
    "ServiceServer",
    "call",
    "set_channel",
    "socket_pair",
]

DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class ServiceError(RuntimeError):
    """The controller refused or failed a request; ``kind`` names its exception."""

    def __init__(self, message: str, kind: str = "") -> None:
        super().__init__(message)
        self.kind = kind


def socket_pair() -> tuple[socket.socket, socket.socket]:
    """Return ``(controller_end, worker_end)``; pass the worker end's fd on."""
    controller, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    controller.set_inheritable(False)
    return controller, worker


def _line(payload: dict[str, Any], limit: int) -> bytes:
    data = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    if len(data) > limit:
        raise ServiceError(
            f"service message is {len(data):,} bytes, over the {limit:,} byte limit"
        )
    return data


class ServiceServer:
    """Answer one worker's requests on a controller thread pool."""

    def __init__(
        self,
        sock: socket.socket,
        handler: Callable[[Any], Any],
        *,
        workers: int = 16,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self._sock = sock
        self._handler = handler
        self._limit = max_message_bytes
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="corral-service"
        )
        self._write_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)

    def start(self) -> ServiceServer:
        self._reader.start()
        return self

    def close(self) -> None:
        """Stop reading, finish answering what arrived, and close the socket."""
        with suppress(OSError):
            self._sock.shutdown(socket.SHUT_RD)
        self._reader.join(timeout=5)
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._sock.close()

    def _read(self) -> None:
        with self._sock.makefile("rb") as stream:
            while line := stream.readline(self._limit + 1):
                if len(line) > self._limit:
                    # An oversize request ends the conversation: its tail would
                    # otherwise be read as the next message.
                    return
                try:
                    request = json.loads(line)
                    request_id = int(request["id"])
                except (ValueError, KeyError, TypeError):
                    return
                self._pool.submit(self._answer, request_id, request.get("body"))

    def _answer(self, request_id: int, body: Any) -> None:
        try:
            reply = {"id": request_id, "ok": True, "body": self._handler(body)}
            data = _line(reply, self._limit)
        except Exception as exc:  # the worker sees the error, the controller lives
            data = _line(
                {
                    "id": request_id,
                    "ok": False,
                    "error": str(exc)[:4000],
                    "kind": type(exc).__name__,
                },
                self._limit,
            )
        with self._write_lock, suppress(OSError):
            self._sock.sendall(data)


class ServiceClient:
    """Send requests to the controller from inside a worker; thread-safe."""

    def __init__(
        self, sock: socket.socket, *, max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    ) -> None:
        self._sock = sock
        self._limit = max_message_bytes
        self._ids = itertools.count(1)
        self._pending: dict[int, list[Any]] = {}
        self._lock = threading.Lock()
        self._closed: str | None = None
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def call(self, body: Any, timeout: float | None = None) -> Any:
        request_id = next(self._ids)
        slot: list[Any] = [threading.Event(), None]
        with self._lock:
            if self._closed is not None:
                raise ServiceError(self._closed)
            self._pending[request_id] = slot
            try:
                self._sock.sendall(_line({"id": request_id, "body": body}, self._limit))
            except OSError as exc:
                self._pending.pop(request_id, None)
                raise ServiceError(f"service channel is closed: {exc}") from exc
        if not slot[0].wait(timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise ServiceError("service request timed out")
        reply = slot[1]
        if not reply.get("ok"):
            raise ServiceError(str(reply.get("error", "")), str(reply.get("kind", "")))
        return reply.get("body")

    def _read(self) -> None:
        reason = "service channel closed by the controller"
        try:
            with self._sock.makefile("rb") as stream:
                while line := stream.readline(self._limit + 1):
                    if len(line) > self._limit:
                        reason = "service reply exceeded the message limit"
                        break
                    reply = json.loads(line)
                    with self._lock:
                        slot = self._pending.pop(int(reply.get("id", -1)), None)
                    if slot is not None:
                        slot[1] = reply
                        slot[0].set()
        except (OSError, ValueError) as exc:
            reason = f"service channel failed: {exc}"
        with self._lock:
            self._closed = reason
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            slot[1] = {"ok": False, "error": reason}
            slot[0].set()


_channel_fd: int | None = None
_client: ServiceClient | None = None
_client_lock = threading.Lock()


def set_channel(descriptor: int | None) -> None:
    """Install this process's channel to the controller (trusted bootstrap only)."""
    global _channel_fd, _client  # noqa: PLW0603 - one channel per worker process
    with _client_lock:
        _channel_fd = descriptor
        _client = None
    if descriptor is not None:
        os.set_inheritable(descriptor, False)


def call(body: Any, timeout: float | None = None) -> Any:
    """Send one request to the controller and return its answer."""
    global _client  # noqa: PLW0603 - created lazily, after privileges are dropped
    with _client_lock:
        if _client is None:
            if _channel_fd is None:
                raise ServiceError("no service channel is available in this process")
            _client = ServiceClient(socket.socket(fileno=_channel_fd))
        client = _client
    return client.call(body, timeout)
