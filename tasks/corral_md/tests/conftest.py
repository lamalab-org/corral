"""External reference responses for synthetic workflow tests."""

import pytest


@pytest.fixture
def rewrite_md_recording():
    """Change serialization only, retaining every original scientific value."""
    import json

    def rewrite(path, representation, *, first_stage=False):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if first_stage:
            stop = next(i for i, row in enumerate(rows) if row.get("event") == "end")
            rows = rows[: stop + 1]
        if representation == "aliased":
            names = {
                "event": "type",
                "random_seed": "seed",
                "numbers": "atomic_numbers",
                "positions_A": "positions",
                "cell_A": "cell",
                "potential_energy_eV": "epot_eV",
            }
            rows = [
                {names.get(key, key): value for key, value in row.items()}
                for row in rows
            ]
            rows[0]["type"] = "metadata"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            return
        header = {key: value for key, value in rows[0].items() if key != "event"}
        stages = []
        for row in rows[1:]:
            payload = {key: value for key, value in row.items() if key != "event"}
            if row["event"] == "stage":
                stages.append({**payload, "samples": []})
            elif row["event"] == "sample":
                stages[-1]["samples"].append(payload)
            else:
                stages[-1]["end"] = payload
        path.write_text(json.dumps({"metadata": header, "stages": stages}))

    return rewrite


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
