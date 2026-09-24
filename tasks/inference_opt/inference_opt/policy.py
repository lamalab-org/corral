"""Load and validate submitted policy modules."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inference_opt.api import ComponentSpec, PolicyManifest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from inference_opt.api import Question, SetupContext, SolveContext

__all__ = [
    "LegacyPolicy",
    "LoadedPolicy",
    "PolicyError",
    "discover_policy",
    "load_policy",
    "manifest_from_mapping",
    "validate_policy",
]

_MODULE_NAME = "submitted_policy"
_IMPORT_LOCK = threading.Lock()

_MANIFEST_FIELDS = frozenset(
    {
        "name",
        "version",
        "max_calls_per_question",
        "setup_calls",
        "max_tokens_per_call",
        "concurrent",
        "components",
        "config",
    }
)


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
    components = values.get("components") or ()
    parsed: list[ComponentSpec] = []
    for entry in components:
        if isinstance(entry, str):
            parsed.append(ComponentSpec(name=entry))
        elif isinstance(entry, dict):
            if "name" not in entry:
                raise PolicyError("each MANIFEST component needs a 'name'")
            parsed.append(
                ComponentSpec(
                    name=str(entry["name"]),
                    kind=str(entry.get("kind", "other")),
                    purpose=str(entry.get("purpose", "")),
                )
            )
        else:
            raise PolicyError(
                f"MANIFEST components must be strings or dicts, got {type(entry).__name__}"
            )
    values["components"] = tuple(parsed)

    try:
        return PolicyManifest(**values)
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"invalid MANIFEST: {exc}") from exc


class LegacyPolicy:
    """Adapt the legacy module-level solve function."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self.manifest = PolicyManifest(name=getattr(fn, "__name__", "legacy"))

    def setup(self, ctx: SetupContext) -> None:
        """Legacy policies have no setup phase."""

    def solve(self, question: Question, ctx: SolveContext) -> Any:
        context = {
            "benchmark": question.benchmark,
            "id": question.id,
            "answer_type": question.answer_type,
            "choices": list(question.choices or ()),
            "topic": question.topic,
            "index": question.index,
            "total": question.total,
        }
        return self._fn(question.text, ctx.student, context)


@dataclass(frozen=True, slots=True)
class LoadedPolicy:
    """A discovered policy plus everything the evaluator needs to run it."""

    obj: Any
    manifest: PolicyManifest
    root: Path
    is_legacy: bool = False

    @property
    def has_setup(self) -> bool:
        setup = getattr(self.obj, "setup", None)
        return callable(setup) and not isinstance(self.obj, LegacyPolicy)

    def solve(self, question: Question, ctx: SolveContext) -> Any:
        return self.obj.solve(question, ctx)

    def setup(self, ctx: SetupContext) -> None:
        setup = getattr(self.obj, "setup", None)
        if callable(setup):
            setup(ctx)


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


def _check_solve_signature(obj: Any) -> None:
    solve = getattr(obj, "solve", None)
    if not callable(solve):
        raise PolicyError(
            "the policy object must define a callable solve(question, ctx)"
        )
    positional = [
        parameter
        for parameter in inspect.signature(solve).parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name not in ("self", "cls")
    ]
    if len(positional) != 2:
        raise PolicyError(
            f"solve() must take exactly (question, ctx); found "
            f"{len(positional)} positional parameter(s): "
            f"{', '.join(parameter.name for parameter in positional) or 'none'}"
        )


def discover_policy(root: Path | str) -> LoadedPolicy:
    """Import a policy directory and normalise it to the class interface.

    Resolution order, first match wins:

    1. a ``Policy`` class (instantiated with no arguments),
    2. a ``policy`` instance,
    3. a ``POLICY`` class or instance,
    4. a module-level ``solve(question, model_client, context)``, wrapped in
       :class:`LegacyPolicy`.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise PolicyError(f"policy directory does not exist: {root}")

    module = _import_module(root)
    obj: Any = None
    is_legacy = False

    candidate = getattr(module, "Policy", None)
    if isinstance(candidate, type):
        try:
            obj = candidate()
        except Exception as exc:
            raise PolicyError(
                f"Policy() raised {type(exc).__name__} during construction: {exc}. "
                "The constructor must take no arguments; do setup work in setup(ctx)."
            ) from exc
    elif (candidate := getattr(module, "policy", None)) is not None:
        obj = candidate
    elif (candidate := getattr(module, "POLICY", None)) is not None:
        obj = candidate() if isinstance(candidate, type) else candidate
    elif callable(getattr(module, "solve", None)):
        obj = LegacyPolicy(module.solve)
        is_legacy = True
    else:
        raise PolicyError(
            "policy.py must define a Policy class with a callable solve(question, "
            "ctx), or a module-level callable solve(question, model_client, context)."
        )

    if not is_legacy:
        _check_solve_signature(obj)

    # A MANIFEST on the object wins over one on the module, so a policy file may
    # define several classes and have each carry its own.
    raw_manifest = getattr(obj, "MANIFEST", None)
    if raw_manifest is None:
        raw_manifest = getattr(module, "MANIFEST", None)
    if raw_manifest is not None:
        manifest = manifest_from_mapping(raw_manifest)
    else:
        existing = getattr(obj, "manifest", None)
        manifest = (
            existing if isinstance(existing, PolicyManifest) else PolicyManifest()
        )

    return LoadedPolicy(
        obj=obj,
        manifest=manifest,
        root=root,
        is_legacy=is_legacy,
    )


def validate_policy(path: str | Path) -> Path:
    """Backwards-compatible check that a directory holds a loadable policy."""
    return discover_policy(path).root


def load_policy(path: str | Path) -> Any:
    """Backwards-compatible accessor returning a callable ``solve``."""
    loaded = discover_policy(path)
    return loaded.solve
