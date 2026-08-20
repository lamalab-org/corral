"""Commit-store interface and persistence-domain errors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from corral.core.actors import ActorRef
    from corral.core.commit import Commit, CommitRequest
    from corral.core.state import ExecutionState


class CommitStoreError(RuntimeError):
    """Base error for commit-ledger persistence failures."""


class CommitNotFoundError(CommitStoreError):
    """The requested commit or execution branch does not exist."""


class CommitConflictError(CommitStoreError):
    """An append, branch, idempotency, or shared-state precondition conflicts."""


class CommitIntegrityError(CommitStoreError):
    """Persisted commit data fails schema, hash, or history validation."""


class AuthorPermissionError(CommitStoreError, PermissionError):
    """An actor is not authorized to author the requested event."""


@runtime_checkable
class CommitStore(Protocol):
    """Persistence contract for a linear authored log per explicit branch."""

    execution_id: str | None

    async def append(self, request: CommitRequest) -> Commit: ...

    async def head(self, branch_id: str) -> Commit | None: ...

    def iter_commits(
        self,
        branch_id: str | None = None,
        *,
        after_sequence: int = -1,
        through_hash: str | None = None,
    ) -> AsyncIterator[Commit]: ...

    async def materialize(
        self, branch_id: str, at_hash: str | None = None
    ) -> ExecutionState: ...

    async def get_commit(self, commit_hash: str) -> Commit: ...

    async def trace_commits(
        self, run_id: str, *, branch_id: str | None = None
    ) -> tuple[Commit, ...]: ...

    async def create_branch(
        self,
        *,
        branch_id: str,
        from_hash: str,
        execution_id: str | None = None,
    ) -> None: ...

    def bind(
        self,
        author: ActorRef,
        *,
        branch_id: str = "main",
        execution_id: str | None = None,
    ) -> BoundCommitStore: ...

    def for_execution(self, execution_id: str) -> CommitStore: ...


@runtime_checkable
class BoundCommitStore(Protocol):
    """A commit capability bound to one trusted author and branch."""

    @property
    def author(self) -> ActorRef: ...

    @property
    def branch_id(self) -> str: ...

    async def append(self, request: CommitRequest) -> Commit: ...


__all__ = [
    "AuthorPermissionError",
    "BoundCommitStore",
    "CommitConflictError",
    "CommitIntegrityError",
    "CommitNotFoundError",
    "CommitStore",
    "CommitStoreError",
]
