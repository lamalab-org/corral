"""Validate direct inputs and text lists before contacting any annotator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .langfuse import LangfuseLocator, parse_langfuse_url


@dataclass(frozen=True)
class InputEntry:
    locator: str
    kind: str
    path: Path | None = None
    remote: LangfuseLocator | None = None
    list_path: str | None = None
    line_number: int | None = None

    @property
    def diagnostic(self) -> str:
        return (
            f"{self.list_path}:{self.line_number}" if self.list_path else self.locator
        )


def resolve_inputs(
    inputs: list[str] | None = None,
    input_list: str | None = None,
    source: str | None = None,
) -> list[InputEntry]:
    if bool(inputs) == bool(input_list):
        raise ValueError("Supply either --input (repeatable) or --input-list")
    if source not in {None, "path", "langfuse"}:
        raise ValueError("source must be path or langfuse")
    list_path = Path(input_list).expanduser().resolve() if input_list else None
    base = list_path.parent if list_path else Path.cwd()
    values = (
        list(enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1))
        if list_path
        else [(None, value) for value in inputs]
    )
    entries, errors = [], []
    for number, raw in values:
        locator = raw.strip() if list_path else raw
        if list_path and (not locator or locator.startswith("#")):
            continue
        where = f"{list_path}:{number}" if list_path else locator
        try:
            if not locator:
                raise ValueError("Empty input")
            is_url = bool(urlsplit(locator).scheme)
            kind = "langfuse" if is_url else "path"
            if source and kind != source:
                raise ValueError(f"Expected a {source} input")
            remote = parse_langfuse_url(locator) if is_url else None
            path = None if is_url else (base / Path(locator).expanduser()).resolve()
            if path and not path.exists():
                raise ValueError(f"Path does not exist: {path}")
            if path and path.is_file() and path.suffix.lower() != ".json":
                raise ValueError(
                    "Expected a JSON trace; use --input-list for text lists"
                )
            entries.append(
                InputEntry(
                    locator,
                    kind,
                    path,
                    remote,
                    str(list_path) if list_path else None,
                    number,
                )
            )
        except (ValueError, OSError) as exc:
            errors.append(f"{where}: {exc}")
    if errors:
        raise ValueError("Invalid analysis inputs:\n" + "\n".join(errors))
    if not entries:
        raise ValueError("No analysis inputs selected")
    return entries
