"""Resolve MD result files inside the workspace restored for evaluation."""

import errno
import json
import posixpath
import re
from pathlib import Path

from corral.workspace import confine_workspace_path, workspace_relative_path


def _manifest_reference(value: str) -> str:
    """Accept one explicit manifest path in an otherwise textual answer.

    Accept inline code references or standalone/labeled path lines. Do not choose
    among different manifests, infer a missing path, or use file existence to
    disambiguate. Resolution and workspace confinement still apply to the path.
    """
    candidates = {
        line.strip()
        for line in value.splitlines()
        if re.fullmatch(r"[^\s`\"'<>]*manifest\.json", line.strip())
        and Path(line.strip()).name == "manifest.json"
    }
    for line in value.splitlines():
        labeled = re.match(
            r"(?:submission\s+)?manifest(?:\s+path)?\s*:\s*"
            r"([^\s`\"'<>()\[\],;]*manifest\.json)(?=$|\s|[,;)]|\.(?:\s|$))",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if labeled and Path(labeled[1]).name == "manifest.json":
            candidates.add(labeled[1])
    remaining = value
    for match in re.finditer(r"(?<!`)`([^`\r\n]+)`(?!`)", value):
        candidate = match.group(1).strip()
        if Path(candidate).name == "manifest.json":
            candidates.add(candidate)
            remaining = remaining.replace(match.group(0), "")
    if not candidates:
        return value
    references = set(re.findall(r"[^\s`\"'<>()\[\],;]*manifest\.json\b", remaining))
    # A filename mentioned in a deliverables list is not a second explicit
    # path. Standalone or code-formatted bare paths remain in candidates.
    references.discard("manifest.json")
    references.update(candidates)
    if len(references) != 1:
        raise ValueError("Submitted manifest path is ambiguous in the answer")
    return references.pop()


class _ResolvedManifest(str):
    """Carry trusted path context without inserting it into agent JSON."""

    __slots__ = ("manifest_dir", "workspace")

    def __new__(cls, value: str, workspace: Path, manifest_dir: Path):
        result = super().__new__(cls, value)
        result.workspace = workspace
        result.manifest_dir = manifest_dir
        return result


def resolve_submission(submission: str, workspace: str | Path) -> str:
    """Handle numeric answers, result paths, JSON objects, and JSON manifests."""
    root = Path(workspace).resolve()

    def confine_link(path: Path) -> Path:
        # Linked paths are relative to the manifest. A sibling such as
        # ../scripts/run.py is valid when it stays within the workspace.
        # Normalize parent segments first, then apply the usual confinement
        # and symlink checks to the resulting path.
        return confine_workspace_path(root, Path(posixpath.normpath(path.as_posix())))

    def find_file(value: str, base: Path) -> Path:
        supplied = Path(value)
        if supplied.is_relative_to("/workspace"):
            candidate = confine_workspace_path(root, workspace_relative_path(value))
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"Submitted file is missing from the workspace: {value}"
                )
            return candidate
        if not supplied.is_absolute() or supplied.is_relative_to(root):
            for directory in (base, root):
                candidate = confine_link(directory / supplied)
                if candidate.is_file():
                    return candidate
            if supplied.is_absolute() or len(supplied.parts) > 1:
                raise FileNotFoundError(
                    f"Submitted file is missing from the workspace: {value}"
                )

        # Absolute paths may name a former local or Modal workspace. Match the
        # longest trailing path inside the saved workspace, never the host file.
        matches = []
        for candidate in root.rglob("*"):
            if (
                candidate.name != supplied.name
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                continue
            confined_candidate = confine_workspace_path(root, candidate)
            common = 0
            for actual, expected in zip(
                reversed(confined_candidate.relative_to(root).parts),
                reversed(supplied.parts),
                strict=False,
            ):
                if actual != expected:
                    break
                common += 1
            if supplied.is_absolute() and common != len(
                confined_candidate.relative_to(root).parts
            ):
                continue
            matches.append((common, confined_candidate))
        if not matches:
            raise FileNotFoundError(
                f"Submitted file is missing from the workspace: {value}"
            )
        best = max(length for length, _ in matches)
        paths = [path for length, path in matches if length == best]
        if len(paths) != 1:
            raise ValueError(
                f"Submitted file path is ambiguous in the workspace: {value}"
            )
        return paths[0]

    def resolve_values(value, base: Path):
        if isinstance(value, dict):
            return {key: resolve_values(item, base) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve_values(item, base) for item in value]
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                return str(find_file(value, base))
        return value

    def resolve_manifest(value, base: Path, original_root: Path | None = None):
        # Shared workflow manifests contain units, model names and descriptions.
        # Only declared path fields are paths; results and metadata are opaque.
        if not isinstance(value, dict) or not (
            {"results", "artifacts", "settings", "scripts", "report"} & value.keys()
        ):
            return json.dumps(resolve_values(value, base))

        def paths(item):
            if isinstance(item, dict):
                return {key: paths(child) for key, child in item.items()}
            if isinstance(item, list):
                return [paths(child) for child in item]
            if not isinstance(item, str):
                return item
            supplied = Path(item)
            if supplied.is_relative_to("/workspace"):
                return str(confine_workspace_path(root, workspace_relative_path(item)))
            if not supplied.is_absolute():
                candidate = base / supplied
            elif supplied.is_relative_to(root):
                candidate = supplied
            elif original_root is not None and supplied.is_relative_to(original_root):
                candidate = root / supplied.relative_to(original_root)
            else:
                # There is no trusted mapping for this external path. Retain
                # it for Evidence to reject, never search for a same-name file.
                # Confinement is enforced even if it names an existing host file.
                return str(supplied)
            # Preserve safe missing paths for per-check partial credit. An
            # explicit runA/file must never resolve to runB/file by basename.
            return str(confine_link(candidate))

        resolved = dict(value)
        for key in ("artifacts", "settings", "scripts", "report"):
            if key in resolved:
                if key == "report" and isinstance(resolved[key], dict):
                    continue
                resolved[key] = paths(resolved[key])
        return _ResolvedManifest(json.dumps(resolved), root, base)

    try:
        value = json.loads(submission)
    except json.JSONDecodeError:
        value = submission.strip()
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            pass
        else:
            return submission
        value = _manifest_reference(value)
        path = find_file(value, root)
        if path.suffix.lower() != ".json":
            return str(path)
        # Paths in a submitted manifest can be relative to its own directory.
        supplied = Path(value)
        original_root = None
        if supplied.is_absolute() and not supplied.is_relative_to(root):
            relative = path.relative_to(root)
            if supplied.parts[-len(relative.parts) :] == relative.parts:
                original_root = supplied.parents[len(relative.parts) - 1]
        return resolve_manifest(
            json.loads(path.read_text()), path.parent, original_root
        )
    return resolve_manifest(value, root)


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
