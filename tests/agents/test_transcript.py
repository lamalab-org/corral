"""Continuous writer failure and callback ordering regressions."""

import asyncio
from types import SimpleNamespace

import pytest

from corral.agents._transcript import StreamingTranscript


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_late_system_prompt_cannot_replace_an_inflight_message():
    writing = asyncio.Event()
    release = asyncio.Event()
    recorded = []

    async def record(message):
        writing.set()
        await release.wait()
        recorded.append(message)

    transcript = StreamingTranscript("test")
    transcript.append({"role": "user", "content": "prompt"})
    transcript.start(SimpleNamespace(record_message=record))
    await writing.wait()
    transcript.insert(0, {"role": "system", "content": "late system"})
    release.set()
    await transcript.flush()
    assert [message["content"] for message in recorded] == ["prompt", "late system"]


@pytest.mark.anyio
async def test_writer_failure_is_reported_without_skipping_messages():
    attempts = []

    async def record(message):
        attempts.append(message)
        raise OSError("store unavailable")

    transcript = StreamingTranscript("test")
    transcript.start(SimpleNamespace(record_message=record))
    transcript.extend(
        [
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
    )
    with pytest.raises(OSError, match="store unavailable"):
        await transcript.flush()
    assert [message["content"] for message in attempts] == ["first"]
    assert transcript._recorded == 0


def test_timezone_unknown_sdk_clock_is_preserved_without_guessing():
    from corral.agents._transcript import event_clock

    timestamp = "2026-11-04T10:00:00"
    clock = event_clock(
        sdk="openhands", item=SimpleNamespace(timestamp=timestamp, id="event-1")
    )
    assert clock["sdk_timestamp"] == timestamp
    assert clock["sdk_item_id"] == "event-1"
    assert clock["timestamp_source"] == "local_observation"
    assert clock["occurred_at"] == clock["observed_at"]
