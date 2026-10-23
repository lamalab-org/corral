"""External reference responses for synthetic workflow tests."""

import pytest


@pytest.fixture
def md_assets(tmp_path, monkeypatch):
    """Small declared assets for environment integration without remote I/O."""
    from corral_md import env
    from corral_md.resources import RUNTIME_VERSION, asset_relative_paths

    from corral.core.resources import declare_file_resource

    root = tmp_path / "assets"
    descriptors = {}
    for name, relative in asset_relative_paths().items():
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"fixture: {relative}\n")
        descriptors[name] = declare_file_resource(
            name, source, runtime_version=RUNTIME_VERSION
        )
    monkeypatch.setattr(env, "asset_descriptors", lambda: descriptors)
    monkeypatch.setenv("CORRAL_MD_ASSETS_DIR", str(root))
    return root
