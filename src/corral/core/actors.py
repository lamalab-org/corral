"""Trusted identities that author durable execution events."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from corral.core._immutable import FrozenModel


class ActorRef(FrozenModel):
    """Stable actor identity plus the identity of this concrete invocation."""

    kind: Literal["agent", "tool", "runtime"]
    actor_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    parent_run_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_parent(self) -> ActorRef:
        if self.parent_run_id == self.run_id:
            raise ValueError("an actor invocation cannot be its own parent")
        return self


__all__ = ["ActorRef"]
