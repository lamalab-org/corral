"""Load and validate submitted policy modules."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inference_opt.api import PolicyManifest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from inference_opt.api import Question, RunContext

__all__ = [
    "LoadedPolicy",
    "PolicyError",
    "discover_policy",
    "manifest_from_mapping",
]

_MODULE_NAME = "submitted_policy"
_IMPORT_LOCK = threading.Lock()

_MANIFEST_FIELDS = frozenset({"name", "version", "max_tokens_per_call"})


class PolicyError(ValueError):
    """Raised when a submitted policy violates the policy contract."""


def manifest_from_mapping(raw: Mapping[str, Any] | None) -> PolicyManifest:
    """Build a validated manifest from a mapping."""
    if not raw:
        return PolicyManifest()
    if not isinstance(raw, dict):
        raise PolicyError(f"MANIFEST must be a dict, got {type(raw).__name__}")

    unknown = set(raw) - _MANIFEST_FIELDS
    if unknown:
        raise PolicyError(
            f"MANIFEST has unknown key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(_MANIFEST_FIELDS))}."
        )

    values = dict(raw)
    try:
        return PolicyManifest(**values)
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"invalid MANIFEST: {exc}") from exc


@dataclass(frozen=True, slots=True)
class LoadedPolicy:
    """A discovered policy plus everything the host needs to run it."""

    obj: Any
    manifest: PolicyManifest
    root: Path

    @property
    def runs_whole_set(self) -> bool:
        """True when the policy defines ``run(questions, ctx)`` itself."""
        return callable(getattr(self.obj, "run", None))

    def run(self, questions: list[Question], ctx: RunContext) -> Any:
        return self.obj.run(questions, ctx)

    def solve(self, question: Question, ctx: RunContext) -> Any:
        return self.obj.solve(question, ctx)


def _import_module(root: Path) -> ModuleType:
    """Import ``<root>/policy.py`` once, with ``root`` importable for helpers."""
    entry = root / "policy.py"
    if not entry.is_file():
        raise PolicyError(
            f"a policy directory must contain policy.py (looked in {root})"
        )

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, entry)
    if spec is None or spec.loader is None:
        raise PolicyError(f"could not load {entry}")
    module = importlib.util.module_from_spec(spec)

    with _IMPORT_LOCK:
        added = str(root) not in sys.path
        if added:
            sys.path.insert(0, str(root))
        previous = sys.modules.get(_MODULE_NAME)
        sys.modules[_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise PolicyError(
                f"policy.py raised {type(exc).__name__} while being imported: {exc}"
            ) from exc
        finally:
            if previous is not None:
                sys.modules[_MODULE_NAME] = previous
            else:
                sys.modules.pop(_MODULE_NAME, None)
            if added:
                try:
                    sys.path.remove(str(root))
                except ValueError:
                    pass
    return module


def _check_signature(obj: Any) -> None:
    """Accept ``run(questions, ctx)``, or ``solve(question, ctx)`` as a shortcut."""
    name = "run" if callable(getattr(obj, "run", None)) else "solve"
    method = getattr(obj, name, None)
    if not callable(method):
        raise PolicyError(
            "the policy object must define run(questions, ctx), or "
            "solve(question, ctx) to answer one question at a time"
        )
    positional = [
        parameter
        for parameter in inspect.signature(method).parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name not in ("self", "cls")
    ]
    if len(positional) != 2:
        expected = "(questions, ctx)" if name == "run" else "(question, ctx)"
        raise PolicyError(
            f"{name}() must take exactly {expected}; found "
            f"{len(positional)} positional parameter(s): "
            f"{', '.join(parameter.name for parameter in positional) or 'none'}"
        )


def discover_policy(root: Path | str) -> LoadedPolicy:
    """Import ``<root>/policy.py`` and return its ``Policy()``, ``policy`` or module.

    Importing runs the policy's code: call this only inside the jail.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise PolicyError(f"policy directory does not exist: {root}")

    module = _import_module(root)
    candidate = getattr(module, "Policy", None)
    if isinstance(candidate, type):
        try:
            obj = candidate()
        except Exception as exc:
            raise PolicyError(
                f"Policy() raised {type(exc).__name__} during construction: {exc}. "
                "The constructor must take no arguments."
            ) from exc
    elif (obj := getattr(module, "policy", None)) is None:
        if not any(callable(getattr(module, name, None)) for name in ("run", "solve")):
            raise PolicyError(
                "policy.py must define run(questions, ctx) or solve(question, ctx), "
                "as functions or as methods of a Policy class"
            )
        obj = module
    _check_signature(obj)

    # A MANIFEST on the object wins over one on the module.
    raw_manifest = getattr(obj, "MANIFEST", None)
    if raw_manifest is None:
        raw_manifest = getattr(module, "MANIFEST", None)
    return LoadedPolicy(obj=obj, manifest=manifest_from_mapping(raw_manifest), root=root)
