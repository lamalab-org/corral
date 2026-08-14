"""StateStore interface and storage-domain errors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from corral.core.state import State


class StateStoreError(RuntimeError):
    """Base error for State persistence failures."""


class StateNotFoundError(StateStoreError):
    """The requested State or revision does not exist."""


class StateTransitionConflictError(StateStoreError):
    """A transition ID was reused for a different child State."""


class StateIntegrityError(StateStoreError):
    """Persisted data does not match its content hash or revision chain."""


@runtime_checkable
class StateStore(Protocol):
    """Content-addressed persistence contract for complete immutable States."""

    async def save(
        self,
        state: State,
        transition_id: str | None = None,
    ) -> State:
        """Persist a complete State after validating its parent relationship."""
        ...

    async def load(self, state_hash: str) -> State:
        """Load and integrity-check the complete State with this content hash."""
        ...

    async def children(self, state_hash: str) -> tuple[State, ...]:
        """Return every directly persisted fork of the selected State."""
        ...
