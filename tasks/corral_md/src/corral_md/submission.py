"""Parse direct MD JSON answers and resolve their workspace artifact links."""

import errno
import json
import posixpath
import re
from pathlib import Path

from corral.workspace import confine_workspace_path, workspace_relative_path


def parse_submission(submission: str) -> dict:
    """Read one leading JSON object, optionally fenced, ignoring trailing prose."""

    trailing = ""
    fenced = None
    if isinstance(submission, str):
        submission = submission.strip()
        fenced = re.match(
            r"```[ \t]*(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```(?=\s|$)",
            submission,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced is not None:
            trailing = submission[fenced.end() :].strip()
            submission = fenced.group(1)

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    decoder = json.JSONDecoder(parse_constant=invalid, object_pairs_hook=unique)
    try:
        if fenced is not None:
            # Everything inside the submission fence must still be valid JSON.
            value = decoder.decode(submission)
        else:
            value, end = decoder.raw_decode(submission)
            trailing = submission[end:].strip()
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Return the submission JSON object directly, not a file path"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("Return the submission JSON object directly, not a file path")
    # Also reject overflow such as 1e999, which parse_constant does not see.
    json.dumps(value, allow_nan=False)
    if trailing.startswith(("{", "}", "[", "]", ",", "```")):
        raise ValueError("Return one submission JSON object followed only by prose")
    # Do not silently choose between competing objects in the explanation.
    for match in re.finditer(r"\{", trailing):
        try:
            decoder.raw_decode(trailing, match.start())
        except json.JSONDecodeError:
            continue
        raise ValueError("Return only one submission JSON object")
    return value


class _ResolvedManifest(str):
    """Carry trusted path context without inserting it into agent JSON."""

    __slots__ = ("manifest_dir", "workspace")

    def __new__(cls, value: str, workspace: Path, manifest_dir: Path):
        result = super().__new__(cls, value)
        result.workspace = workspace
        result.manifest_dir = manifest_dir
        return result


def resolve_submission(submission: str, workspace: str | Path) -> str:
    """Resolve artifact links in a directly returned JSON answer."""
    value = parse_submission(submission)
    root = Path(workspace).resolve()

    def paths(item):
        if isinstance(item, dict):
            return {key: paths(child) for key, child in item.items()}
        if isinstance(item, list):
            return [paths(child) for child in item]
        if not isinstance(item, str) or not item:
            return item
        supplied = Path(item)
        if supplied.is_relative_to("/workspace"):
            candidate = root / workspace_relative_path(item)
        elif not supplied.is_absolute():
            candidate = root / supplied
        elif supplied.is_relative_to(root):
            candidate = supplied
        else:
            # Preserve external links for Evidence to reject per check; never
            # select another run's file by basename or read a host file.
            return str(supplied)
        # Preserve safe missing paths for per-check partial credit.
        return str(
            confine_workspace_path(root, Path(posixpath.normpath(str(candidate))))
        )

    resolved = dict(value)
    for key in ("artifacts", "settings", "scripts", "report"):
        if key in resolved:
            if key == "report" and isinstance(resolved[key], dict):
                continue
            resolved[key] = paths(resolved[key])
    return _ResolvedManifest(json.dumps(resolved), root, root)


def resolve_local_submission(submission: str, workspace: str | Path) -> str:
    """Score an absent agent submission as zero in local benchmark reports."""
    try:
        return resolve_submission(submission, workspace)
    except (ValueError, UnicodeError, OSError) as exc:
        if (
            isinstance(exc, OSError)
            and not isinstance(exc, FileNotFoundError)
            and exc.errno
            not in {errno.ENOENT, errno.ENAMETOOLONG, errno.EACCES, errno.ELOOP}
        ):
            raise
        return json.dumps(
            {
                "results": {},
                "artifacts": {},
                "invalid_submission": str(exc)[:500],
            }
        )
