"""Trial-wide physical inventory for agents that fork experiment branches."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

from corral.runtime.permissions import NODE_WORKSPACE_DIR


def inventory_path(workspace: str) -> Path:
    """Keep the physical inventory outside every agent-accessible workspace."""
    path = Path(workspace).resolve()
    trial = path.parent.parent if path.parent.name == NODE_WORKSPACE_DIR else path
    return trial.parent / ".wetlab-private" / trial.name / "inventory.json"


@dataclass
class InventoryTransaction:
    inventory: dict[str, Any]
    actions: dict[str, Any]
    changed: bool = False
    pending: dict[str, Any] = field(default_factory=dict)

    def record(self, action_key: str, result: Any) -> None:
        self.pending[action_key] = result

    def commit(self, inventory: dict[str, Any]) -> None:
        self.inventory = inventory
        self.actions.update(self.pending)
        self.pending.clear()
        self.changed = True


class SharedInventory:
    """A locked JSON checkpoint, with action results for safe invocation retries."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise RuntimeError(
                "Shared WetLab inventory is missing; refusing to replenish it"
            )
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("inventory"), dict):
            raise ValueError("Invalid shared WetLab inventory checkpoint")
        return data

    def _write(self, data: Mapping[str, Any]) -> None:
        staged = self.path.with_suffix(".tmp")
        staged.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
        staged.replace(self.path)

    def initialize(self, inventory: Mapping[str, Any]) -> None:
        with self._lock():
            if self.path.exists():
                current = self._read()
                if (
                    current["inventory"]["chemical_system"]
                    != inventory["chemical_system"]
                ):
                    raise ValueError(
                        "Shared WetLab inventory uses a different chemical system"
                    )
            else:
                self._write({"inventory": dict(inventory), "actions": {}})

    def read(self) -> dict[str, Any]:
        with self._lock():
            return self._read()["inventory"]

    @contextmanager
    def transaction(self) -> Iterator[InventoryTransaction]:
        with self._lock():
            data = self._read()
            transaction = InventoryTransaction(
                data["inventory"], data.get("actions", {})
            )
            yield transaction
            if transaction.changed:
                self._write(
                    {"inventory": transaction.inventory, "actions": transaction.actions}
                )
