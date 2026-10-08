"""Exercise NMR rate limits without contacting the prediction service."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import aiohttp
import pytest
from spectra_elucidation import spectra_utils
from spectra_elucidation.spectra_utils import SpectraAPI
from spectra_elucidation.tools import carbon_nmr_spectra, proton_nmr_spectra


@pytest.fixture(autouse=True)
def reset_rate_limit(monkeypatch):
    monkeypatch.setattr(SpectraAPI, "_last_request_time", 0)
    monkeypatch.setattr(SpectraAPI, "_nmr_retry_after", 0)
    monkeypatch.setattr(spectra_utils, "logger", Mock())


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=1000.0, sleeps=[], on_sleep=None)

    async def sleep(delay):
        state.sleeps.append(delay)
        if state.on_sleep:
            state.on_sleep()
        state.now += delay

    monkeypatch.setattr(
        spectra_utils,
        "time",
        SimpleNamespace(monotonic=lambda: state.now, time=lambda: state.now - 1000),
    )
    monkeypatch.setattr(spectra_utils, "asyncio", SimpleNamespace(sleep=sleep))
    return state


class Session:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.active = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        pass

    @asynccontextmanager
    async def post(self, url, **kwargs):
        self.requests.append((spectra_utils.time.monotonic(), url, kwargs))
        status, retry_after = next(self.replies)
        headers = {} if retry_after is None else {"Retry-After": retry_after}

        def raise_for_status():
            if status >= 400:
                raise aiohttp.ClientResponseError(
                    SimpleNamespace(real_url=url),
                    (),
                    status=status,
                    headers=headers,
                )

        async def json():
            return {
                "data": {
                    "signals": [{"delta": 10}],
                    "ranges": [
                        {"signals": [{"delta": 1.2}], "integration": 3},
                    ],
                }
            }

        async def text():
            return "Too Many Requests"

        self.active = True
        try:
            yield SimpleNamespace(
                status=status,
                headers=headers,
                raise_for_status=raise_for_status,
                json=json,
                text=text,
            )
        finally:
            self.active = False


@pytest.mark.parametrize(
    ("tool", "endpoint", "expected"),
    [
        (carbon_nmr_spectra, "carbon", "Deltas: 10.00"),
        (proton_nmr_spectra, "proton", "Deltas 1.20 (m, 3H)."),
    ],
)
def test_tools_retry_after_header(monkeypatch, clock, tool, endpoint, expected):
    session = Session([(429, "166"), (200, None)])
    monkeypatch.setattr(spectra_utils.aiohttp, "ClientSession", lambda: session)

    def check_response_released():
        assert not session.active

    clock.on_sleep = check_response_released
    assert tool.execute(h_smiles="CCO") == expected
    assert [request[0] for request in session.requests] == [1000, 1166]
    assert all(request[1].endswith(f"/{endpoint}") for request in session.requests)
    assert all(request[2]["json"] == {"smiles": "CCO"} for request in session.requests)
    spectra_utils.logger.error.assert_not_called()


@pytest.mark.parametrize(
    ("header", "delay"),
    [
        ("Thu, 01 Jan 1970 00:02:46 GMT", 166),
        ("Wed, 31 Dec 1969 23:59:59 GMT", 1),
        ("0", 1),
        (None, 2),
        ("invalid", 2),
    ],
)
def test_retry_after_formats_and_fallback(clock, header, delay):
    session = Session([(429, header), (200, None)])
    asyncio.run(SpectraAPI.get_prediction_async(session, "CCO", "nmr"))
    assert session.requests[1][0] - session.requests[0][0] == delay
    assert clock.sleeps == [delay]


def test_repeated_rate_limits_stop_after_three_attempts(clock):
    session = Session([(429, None)] * 3)
    with pytest.raises(aiohttp.ClientResponseError) as error:
        asyncio.run(SpectraAPI.get_prediction_async(session, "CCO", "nmr"))
    assert error.value.status == 429
    assert len(session.requests) == 3
    assert clock.sleeps == [2, 4]
    assert spectra_utils.logger.error.called
    # The last response still informs other callers of the shared cooldown.
    assert SpectraAPI._nmr_retry_after == clock.now + 8


@pytest.mark.parametrize(
    ("prediction_type", "status"),
    [("nmr", 400), ("nmr", 401), ("nmr", 403), ("nmr", 404), ("nmr", 500), ("ir", 429)],
)
@pytest.mark.usefixtures("clock")
def test_other_failures_are_not_retried(prediction_type, status):
    session = Session([(status, "166")])
    with pytest.raises(aiohttp.ClientResponseError) as error:
        asyncio.run(SpectraAPI.get_prediction_async(session, "CCO", prediction_type))
    assert error.value.status == status
    assert len(session.requests) == 1
    assert SpectraAPI._nmr_retry_after == 0


def test_waiting_call_rechecks_extended_cooldown(clock, monkeypatch):
    monkeypatch.setattr(SpectraAPI, "_last_request_time", clock.now)

    def extend_cooldown():
        SpectraAPI._nmr_retry_after = clock.now + 166
        clock.on_sleep = None

    clock.on_sleep = extend_cooldown
    session = Session([(200, None)])
    asyncio.run(SpectraAPI.get_prediction_async(session, "CCO", "nmr"))
    assert clock.sleeps == [1, 165]
    assert session.requests[0][0] == 1166


def test_cooldown_shared_across_worker_event_loops(monkeypatch):
    monkeypatch.setattr(SpectraAPI, "_min_request_interval", 0.02)
    rate_limited = threading.Event()
    spectra_utils.logger.warning.side_effect = lambda *_args: rate_limited.set()
    proton = Session([(429, "1"), (200, None)])
    carbon = Session([(200, None)])

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(
            asyncio.run, SpectraAPI.get_prediction_async(proton, "CCO", "nmr", "proton")
        )
        assert rate_limited.wait(timeout=5)
        deadline = SpectraAPI._nmr_retry_after
        second = workers.submit(
            asyncio.run, SpectraAPI.get_prediction_async(carbon, "CCO", "nmr", "carbon")
        )
        assert first.result(timeout=5)["data"]
        assert second.result(timeout=5)["data"]

    times = sorted([proton.requests[1][0], carbon.requests[0][0]])
    assert times[0] >= deadline
    assert times[1] - times[0] >= 0.019
    assert time.monotonic() >= deadline


def test_cancellation_during_cooldown_propagates(clock):
    def cancel():
        raise asyncio.CancelledError

    clock.on_sleep = cancel
    session = Session([(429, "166")])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(SpectraAPI.get_prediction_async(session, "CCO", "nmr"))
    assert len(session.requests) == 1
    spectra_utils.logger.error.assert_not_called()
