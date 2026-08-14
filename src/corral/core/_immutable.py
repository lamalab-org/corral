"""Shared deeply immutable Pydantic model primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

SHA256_HEX_LENGTH = 64
_SHA256_HEX_CHARACTERS = frozenset("0123456789abcdef")


class FrozenDict(dict[str, Any]):
    """A JSON-serializable mapping that rejects mutation."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError(
            "State values are immutable; use State.fork() or WorkspaceState.fork()"
        )

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[type[FrozenDict], tuple[dict[str, Any]]]:
        return type(self), (dict(self),)


class FrozenList(list[Any]):
    """A JSON-serializable sequence that rejects mutation."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError(
            "State values are immutable; use State.fork() or WorkspaceState.fork()"
        )

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __copy__(self) -> FrozenList:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenList:
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[type[FrozenList], tuple[list[Any]]]:
        return type(self), (list(self),)


def _freeze(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


class FrozenModel(BaseModel):
    """Base model with deep rather than merely field-level immutability."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        ser_json_timedelta="iso8601",
    )

    @model_validator(mode="after")
    def _freeze_nested_values(self) -> FrozenModel:
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, _freeze(getattr(self, field_name)))
        return self

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> FrozenModel:
        """Copy frozen data but reject Pydantic's unvalidated update shortcut."""
        if update:
            raise TypeError(
                "State models are immutable; use State.fork() or "
                "WorkspaceState.fork()"
            )
        return super().model_copy(deep=deep)


def validate_sha256_hex(value: str, *, field_name: str) -> None:
    """Validate the canonical lowercase representation of a SHA-256 digest."""
    if len(value) != SHA256_HEX_LENGTH or any(
        character not in _SHA256_HEX_CHARACTERS for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
