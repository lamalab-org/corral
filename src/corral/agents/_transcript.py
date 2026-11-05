"""Ordered, continuously drained SDK transcripts and explicit event clocks."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any


def event_clock(
    *, sdk: str | None = None, item: Any = None, envelope: Any = None
) -> dict[str, Any]:
    """Never present receipt time as a provider-supplied execution timestamp."""
    observed = datetime.now(timezone.utc)
    supplied = getattr(item, "timestamp", None)
    raw_timestamp = supplied
    timestamp_field = "timestamp"
    for field in ("started_at_ms", "completed_at_ms"):
        milliseconds = getattr(envelope, field, None)
        if isinstance(milliseconds, int | float):
            raw_timestamp = milliseconds
            timestamp_field = field
            supplied = datetime.fromtimestamp(milliseconds / 1000, timezone.utc)
            break
    if isinstance(supplied, str):
        try:
            supplied = datetime.fromisoformat(supplied.replace("Z", "+00:00"))
        except ValueError:
            supplied = None
    if not isinstance(supplied, datetime) or supplied.tzinfo is None:
        supplied = None
    return {
        "observed_at": observed.isoformat(),
        "occurred_at": (supplied or observed).isoformat(),
        "timestamp_source": "sdk" if supplied else "local_observation",
        **(
            {
                "sdk_timestamp": raw_timestamp.isoformat()
                if isinstance(raw_timestamp, datetime)
                else raw_timestamp,
                "sdk_timestamp_field": timestamp_field,
            }
            if raw_timestamp is not None
            else {}
        ),
        **({"sdk": sdk} if sdk else {}),
        **({"sdk_item_id": str(item.id)} if getattr(item, "id", None) else {}),
    }


class StreamingTranscript(list):
    """One writer on the session loop, usable from SDK callback threads.

    Appends capture observation time before queuing. Flush is a barrier for all
    messages received so far, including when called from an MCP executor loop.
    Writer failures propagate at the barrier; a failed write is never skipped.
    """

    def __init__(self, sdk: str):
        super().__init__()
        self.sdk = sdk
        self._clock = ContextVar(f"transcript_clock_{id(self)}", default=None)
        self._loop = None
        self._session = None
        self._writer = None
        self._recorded = 0
        self._writing = False
        self._lock = None

    def start(self, session) -> None:
        self._loop = asyncio.get_running_loop()
        self._session = session
        self._lock = asyncio.Lock()
        self._schedule()

    @contextmanager
    def event(self, item, *, envelope=None):
        token = self._clock.set(event_clock(sdk=self.sdk, item=item, envelope=envelope))
        try:
            yield
        finally:
            self._clock.reset(token)

    def _message(self, raw):
        message = dict(raw)
        calls = message.get("tool_calls", ())
        clock = dict(self._clock.get() or event_clock(sdk=self.sdk))
        clock["phase"] = (
            "tool_started"
            if calls
            else "tool_completed"
            if message.get("role") == "tool"
            else "message"
        )
        if calls or message.get("tool_call_id"):
            clock.setdefault(
                "sdk_item_id", calls[0]["id"] if calls else message["tool_call_id"]
            )
        message.setdefault("chronology", clock)
        return message

    def append(self, raw) -> None:
        super().append(self._message(raw))
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._schedule)

    def insert(self, index, raw) -> None:
        if index < self._recorded + int(self._writing):
            # A late system prompt cannot precede already durable messages.
            self.append(raw)
            return
        super().insert(index, self._message(raw))
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._schedule)

    def extend(self, messages) -> None:
        for message in messages:
            self.append(message)

    def _schedule(self) -> None:
        # Retain a failed task so flush surfaces its exception.
        if self._writer is None or (
            self._writer.done() and self._writer.exception() is None
        ):
            self._writer = self._loop.create_task(self._drain())

    async def _drain(self) -> None:
        async with self._lock:
            while self._recorded < len(self):
                self._writing = True
                try:
                    await self._session.record_message(self[self._recorded])
                    self._recorded += 1
                finally:
                    self._writing = False

    def flush_from_thread(self) -> None:
        """Wait for durable writes from the SDK worker before it advances."""
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self.flush(), self._loop).result()

    async def flush(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(self.flush(), self._loop)
            )
            return
        if self._writer is not None:
            await asyncio.shield(self._writer)
        await self._drain()
