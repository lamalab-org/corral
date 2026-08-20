"""Canonical requests and records for the authored commit ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from corral.core._immutable import FrozenModel, validate_sha256_hex
from corral.core.actors import ActorRef
from corral.core.events import StateEvent

COMMIT_SCHEMA_VERSION = 1


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("commit timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


class CommitRequest(FrozenModel):
    """An authenticated request; ordering and hashes are store-owned fields."""

    request_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    based_on_hash: str | None = None
    author: ActorRef
    event: StateEvent
    occurred_at: datetime | None = None
    # A store is normally execution-bound. This optional field makes shared
    # database/service adapters possible without weakening the record model.
    execution_id: str | None = Field(default=None, min_length=1)

    @field_validator("based_on_hash")
    @classmethod
    def _validate_based_on_hash(cls, value: str | None) -> str | None:
        if value is not None:
            validate_sha256_hex(value, field_name="based_on_hash")
        return value

    @field_validator("occurred_at")
    @classmethod
    def _validate_occurred_at(cls, value: datetime | None) -> datetime | None:
        return _aware(value)


class Commit(FrozenModel):
    """One small immutable event accepted into an execution branch."""

    schema_version: int = COMMIT_SCHEMA_VERSION
    hash: str
    execution_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    branch_sequence: int = Field(ge=0)
    parent_hash: str | None = None
    based_on_hash: str | None = None
    author: ActorRef
    event: StateEvent
    occurred_at: datetime
    recorded_at: datetime

    @field_validator("hash", "parent_hash", "based_on_hash")
    @classmethod
    def _validate_hashes(cls, value: str | None, info: Any) -> str | None:
        if value is not None:
            validate_sha256_hex(value, field_name=info.field_name)
        return value

    @field_validator("occurred_at", "recorded_at")
    @classmethod
    def _validate_timestamps(cls, value: datetime) -> datetime:
        normalized = _aware(value)
        assert normalized is not None
        return normalized

    @model_validator(mode="after")
    def _validate_content_hash(self) -> Commit:
        expected = commit_hash(self.model_dump(mode="json", exclude={"hash"}))
        if self.hash != expected:
            raise ValueError("commit hash does not match its canonical record")
        return self

    def to_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))

    @classmethod
    def create(cls, **record: Any) -> Commit:
        """Create a record after calculating its canonical content hash."""
        payload = {"schema_version": COMMIT_SCHEMA_VERSION, **record}
        payload["hash"] = commit_hash(payload)
        return cls.model_validate(payload)


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def commit_hash(record_without_hash: Mapping[str, Any]) -> str:
    payload = dict(record_without_hash)
    payload.pop("hash", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        normalized = _aware(value)
        assert normalized is not None
        return normalized.isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "COMMIT_SCHEMA_VERSION",
    "Commit",
    "CommitRequest",
    "canonical_json",
    "commit_hash",
]
