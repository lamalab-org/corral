"""Resolve MD result files inside the workspace restored for evaluation."""

import json
from pathlib import Path

from corral.workspace import confine_workspace_path, workspace_relative_path


class _ResolvedManifest(str):
    """Carry trusted path context without inserting it into agent JSON."""

    __slots__ = ("workspace", "manifest_dir")

    def __new__(cls, value: str, workspace: Path, manifest_dir: Path):
        result = super().__new__(cls, value)
        result.workspace = workspace
        result.manifest_dir = manifest_dir
        return result


def resolve_submission(submission: str, workspace: str | Path) -> str:
    """Handle numeric answers, result paths, JSON objects, and JSON manifests."""
    root = Path(workspace).resolve()

    def find_file(value: str, base: Path) -> Path:
        supplied = Path(value)
        if supplied.is_relative_to("/workspace"):
            candidate = confine_workspace_path(root, workspace_relative_path(value))
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"Submitted file is missing from the workspace: {value}"
                )
            return candidate
        if ".." in supplied.parts:
            raise ValueError(f"Submitted path cannot traverse its workspace: {value}")
        if not supplied.is_absolute() or supplied.is_relative_to(root):
            for directory in (base, root):
                candidate = confine_workspace_path(root, directory / supplied)
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
            if ".." in supplied.parts:
                raise ValueError(
                    f"Submitted path cannot traverse its workspace: {item}"
                )
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
            return str(confine_workspace_path(root, candidate))

        resolved = dict(value)
        for key in ("artifacts", "settings", "scripts", "report"):
            if key in resolved:
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
