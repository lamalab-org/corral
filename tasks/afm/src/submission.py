"""Map workstation scan paths into the immutable evaluation workspace."""

import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate submission field: {key}")
        result[key] = value
    return result


def resolve_submission(submission: str, workspace: str | Path) -> str:
    """Resolve bare absolute paths and numbered JSON paths against saved scans.

    Absolute Windows (including UNC) and POSIX paths may refer to a workstation
    that is no longer available. Match their full workspace-relative suffix,
    preferring the longest match and rejecting ambiguity. Never read a live file
    outside the evaluation workspace. Measurements and other fields stay intact.
    """
    root = Path(workspace).resolve()

    def confined_file(path):
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Submitted scan must not be a symbolic link")
        if not path.is_file():
            raise ValueError("Submitted scan is missing from the saved workspace")
        return path

    def resolve_path(value):
        if not isinstance(value, str):
            raise ValueError("Expected an absolute NID path")
        value = value.strip()
        windows = PureWindowsPath(value)
        is_windows = bool(windows.drive) or "\\" in value
        supplied = windows if is_windows else PurePosixPath(value)
        if (
            not supplied.is_absolute()
            or supplied.suffix.lower() != ".nid"
            or ".." in supplied.parts
        ):
            raise ValueError("Expected an absolute NID path without parent traversal")

        # Canonical tool paths and paths already inside this materialization
        # have an exact mapping. A missing file must not select another scan.
        if not is_windows and supplied.is_relative_to("/workspace"):
            return str(confined_file(root / supplied.relative_to("/workspace")))
        native = Path(value)
        if native.is_absolute() and native.is_relative_to(root):
            return str(confined_file(native))

        expected = supplied.parts
        if is_windows:
            expected = tuple(part.casefold() for part in expected)
        matches = []
        for candidate in root.rglob("*"):
            if candidate.suffix.lower() != ".nid" or candidate.is_symlink():
                continue
            relative = candidate.relative_to(root).parts
            actual = tuple(p.casefold() for p in relative) if is_windows else relative
            if len(actual) <= len(expected) and expected[-len(actual) :] == actual:
                matches.append((len(actual), confined_file(candidate)))
        if not matches:
            raise ValueError("Submitted scan is missing from the saved workspace")
        longest = max(length for length, _ in matches)
        best = [path for length, path in matches if length == longest]
        if len(best) != 1:
            raise ValueError("Submitted scan path is ambiguous in the saved workspace")
        return str(best[0])

    try:
        if submission.lstrip().startswith("{"):
            report = json.loads(submission, object_pairs_hook=_unique_object)
            return json.dumps(
                {
                    key: resolve_path(value)
                    if re.fullmatch(r"path_[1-9]\d*", key)
                    else value
                    for key, value in report.items()
                }
            )
        return resolve_path(submission)
    except (OSError, ValueError, TypeError):
        # The scorer treats an invalid answer as zero. Do not return the original
        # path here: it could still point at a mutable file on the live machine.
        return ""
