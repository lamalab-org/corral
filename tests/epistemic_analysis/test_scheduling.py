import asyncio

import pytest

from corral.epistemic_analysis.scheduling import rolling_map


def test_rolling_map_refills_slots_and_preserves_order():
    async def exercise():
        last_started = asyncio.Event()
        active = peak = 0

        async def process(number):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                if number == 0:
                    await asyncio.wait_for(last_started.wait(), 2)
                if number == 9:
                    last_started.set()
                await asyncio.sleep(0)
                return number
            finally:
                active -= 1

        assert await rolling_map(process, range(10), concurrency=2) == list(range(10))
        assert peak == 2
        assert active == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_rolling_map_drains_tasks_on_failure_or_cancellation(cancel):
    async def exercise():
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def process(number):
            if number == 0:
                await started.wait()
                if cancel:
                    raise asyncio.CancelledError
                raise RuntimeError("failed")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await rolling_map(process, range(2), concurrency=2)
        assert stopped.is_set()

    asyncio.run(exercise())
