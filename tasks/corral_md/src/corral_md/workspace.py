"""Agent paths and read-only asset catalogs for the MD workspace."""

import shutil
from pathlib import Path

import modal
from modal.volume import FileEntryType

from corral.workspace import (
    AbsoluteWorkspaceFilesystem,
    confine_workspace_path,
    workspace_relative_path,
)

ASSET_DIRECTORIES = frozenset({"models", "potentials", "structures"})


def local_path(root: str | Path, path: str, *, allow_root: bool = False) -> Path:
    """Translate a validated public path to mutable controller storage."""
    relative = workspace_relative_path(path, allow_root=allow_root)
    if relative.split("/")[0] in ASSET_DIRECTORIES:
        raise PermissionError(f"Shared assets are read-only: {path}")
    return confine_workspace_path(root, relative, allow_root=allow_root)


class MDWorkspaceFilesystem(AbsoluteWorkspaceFilesystem):
    """Present remote assets alongside local files without snapshotting assets."""

    def __init__(self, root: str | Path, *, assets_dir: str | Path | None = None):
        super().__init__(root)
        self.assets_dir = (
            Path(assets_dir).expanduser().resolve() if assets_dir else None
        )

    def _asset(self, path: str):
        from corral_md.modal_workspace import (  # — avoid a client/catalog import cycle
            configured_asset_volume_name,
        )

        relative = workspace_relative_path(path, allow_root=True)
        kind, _, tail = relative.partition("/")
        if kind not in ASSET_DIRECTORIES:
            return None
        if self.assets_dir is not None:
            catalog = self.assets_dir / kind
            if not catalog.is_dir():
                raise FileNotFoundError(f"Local MD asset catalog not found: {catalog}")
            target = (catalog / tail).resolve()
            if not target.is_relative_to(catalog):
                raise ValueError(f"Asset path escapes its catalog: {path}")
            return kind, target, None
        volume = modal.Volume.from_name(configured_asset_volume_name(kind))
        return kind, volume, "/" + tail

    def _resolve(self, path: str, *, allow_root: bool = False) -> Path:
        return local_path(self.root, path, allow_root=allow_root)

    def read_file(self, path: str, *, encoding: str = "utf-8") -> str:
        asset = self._asset(path)
        if asset:
            _, volume, remote = asset
            if remote is None:
                return volume.read_text(encoding=encoding)
            return b"".join(volume.read_file(remote)).decode(encoding)
        return super().read_file(path, encoding=encoding)

    def list_files(
        self, path: str = "/workspace", *, recursive: bool = False
    ) -> list[str]:
        asset = self._asset(path)
        if asset:
            kind, volume, remote = asset
            if remote is None:
                if not volume.is_dir():
                    raise NotADirectoryError(path)
                entries = volume.rglob("*") if recursive else volume.iterdir()
                catalog = self.assets_dir / kind
                return sorted(
                    f"/workspace/{kind}/{entry.relative_to(catalog).as_posix()}"
                    for entry in entries
                    if entry.is_file() and entry.resolve().is_relative_to(catalog)
                )
            return sorted(
                f"/workspace/{kind}/{entry.path.lstrip('/')}"
                for entry in volume.listdir(remote, recursive=recursive)
                if entry.type == FileEntryType.FILE
            )
        files = super().list_files(path, recursive=recursive)
        if path == "/workspace" and recursive:
            for kind in sorted(ASSET_DIRECTORIES):
                files.extend(self.list_files(f"/workspace/{kind}", recursive=True))
        return sorted(files)

    def file_info(self, path: str) -> dict:
        asset = self._asset(path)
        if asset:
            _, volume, remote = asset
            if remote is None:
                if volume.is_file():
                    return {
                        "path": path,
                        "type": "file",
                        "size": volume.stat().st_size,
                        "read_only": True,
                    }
                if volume.is_dir():
                    return {"path": path, "type": "directory", "read_only": True}
                raise FileNotFoundError(f"Asset not found: {path}")
            if remote == "/":
                return {"path": path, "type": "directory", "read_only": True}
            entries = volume.listdir(remote)
            for entry in entries:
                if entry.path.lstrip("/") == remote.lstrip("/"):
                    if entry.type == FileEntryType.FILE:
                        return {
                            "path": path,
                            "type": "file",
                            "size": entry.size,
                            "read_only": True,
                        }
                    if entry.type != FileEntryType.DIRECTORY:
                        raise ValueError(f"Unsupported asset entry: {path}")
            if entries:
                return {"path": path, "type": "directory", "read_only": True}
            raise FileNotFoundError(f"Asset not found: {path}")
        return super().file_info(path)

    def copy_file(self, source: str, destination: str) -> None:
        target = self._resolve(destination)
        asset = self._asset(source)
        if asset:
            _, volume, remote = asset
            target.parent.mkdir(parents=True, exist_ok=True)
            if remote is None:
                if not volume.is_file():
                    raise FileNotFoundError(f"Asset not found: {source}")
                with (
                    volume.open("rb") as source_stream,
                    target.open("wb") as target_stream,
                ):
                    shutil.copyfileobj(source_stream, target_stream)
            else:
                with target.open("wb") as stream:
                    for chunk in volume.read_file(remote):
                        stream.write(chunk)
            return
        super().copy_file(source, destination)
