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
        *,
        advance_head: bool = False,
    ) -> State:
        """Persist a complete State after validating its parent relationship.

        ``advance_head`` atomically moves the execution's durable head to this
        State. It is used for the canonical task path; speculative branches may
        be saved without moving the head.
        """
        ...

    async def load(self, state_hash: str) -> State:
        """Load and integrity-check the complete State with this content hash."""
        ...

    async def load_initial(self, state_id: str) -> State | None:
        """Load the initial State for an execution identity, if it exists."""
        ...

    async def load_head(self, state_id: str) -> State | None:
        """Load the latest canonical checkpoint for an execution identity."""
        ...

    async def children(self, state_hash: str) -> tuple[State, ...]:
        """Return every directly persisted fork of the selected State."""
        ...

    async def load_transition(
        self,
        parent_hash: str,
        transition_id: str,
    ) -> State | None:
        """Return an already-committed transition, or ``None``.

        Action proposal and observation checkpoints use stable transition IDs,
        so retrying the same parent transition is an idempotent lookup.
        """
        ...
