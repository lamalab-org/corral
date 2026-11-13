from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from wetlab.shared_inventory import SharedInventory


def test_transactions_share_finite_inventory_and_ignore_failed_calls(tmp_path) -> None:
    store = SharedInventory(tmp_path / "inventory.json")
    seed = {"chemical_system": {"elements": "test"}, "inventory": {"sample": 20}}
    store.initialize(seed)

    def draw(amount: int) -> bool:
        with store.transaction() as transaction:
            remaining = transaction.inventory["inventory"]["sample"]
            if amount > remaining:
                return False
            next_inventory = {
                **transaction.inventory,
                "inventory": {"sample": remaining - amount},
            }
            transaction.commit(next_inventory)
            return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(draw, (12, 12)))
    assert results == [True, False] or results == [False, True]
    assert store.read()["inventory"]["sample"] == 8

    with (
        pytest.raises(RuntimeError, match="failed experiment"),
        store.transaction() as transaction,
    ):
        transaction.inventory["inventory"]["sample"] = 0
        raise RuntimeError("failed experiment")
    assert store.read()["inventory"]["sample"] == 8

    with store.transaction() as transaction:
        transaction.record("branch:action", "observed")
        transaction.commit(transaction.inventory)
    with store.transaction() as transaction:
        assert transaction.actions["branch:action"] == "observed"

    store.path.unlink()
    with pytest.raises(RuntimeError, match="refusing to replenish"):
        store.read()
