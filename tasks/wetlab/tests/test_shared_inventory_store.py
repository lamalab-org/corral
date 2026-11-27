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


def test_main_and_cloned_branches_resolve_to_one_inventory(tmp_path) -> None:
    from wetlab.shared_inventory import inventory_path

    from corral.runtime.permissions import NODE_WORKSPACE_DIR

    trial = tmp_path / "trial"
    main = inventory_path(str(trial))
    assert inventory_path(str(trial / NODE_WORKSPACE_DIR / "first")) == main
    assert inventory_path(str(trial / NODE_WORKSPACE_DIR / "clone")) == main
    assert inventory_path(str(tmp_path / "other-trial")) != main
    assert not main.is_relative_to(trial)
