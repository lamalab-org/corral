"""Bounded asynchronous work that refills each slot as soon as it is free."""

from __future__ import annotations

import asyncio

DEFAULT_TRACE_CONCURRENCY = 4


async def rolling_map(function, items, *, concurrency):
    """Run independent items continuously, returning results in input order.

    Only ``concurrency`` tasks are created, even for very large queues. Failure
    or cancellation drains every task before the caller can retry the work.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    iterator = enumerate(items)
    results = []

    async def consume():
        for index, item in iterator:
            results.append(None)
            results[index] = await function(item)

    tasks = [asyncio.create_task(consume()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return results
