import json
from pathlib import Path
from types import SimpleNamespace

from corral_md.resources import asset_descriptors, asset_relative_paths
from corral_md.scoring.local_verifier import LocalCalculationTransport


def test_local_verifier_mounts_declared_assets_at_canonical_read_only_paths(
    tmp_path, monkeypatch
):
    assets = tmp_path / "assets"
    for directory in ("models", "potentials", "structures"):
        (assets / directory).mkdir(parents=True)
    captured = []
    monkeypatch.delenv("CORRAL_MD_VERIFIER_GPUS", raising=False)
    monkeypatch.delenv("CORRAL_MD_VERIFIER_RUNTIME", raising=False)
    monkeypatch.setattr(
        "corral_md.scoring.local_verifier.subprocess.check_output",
        lambda *_args, **_kwargs: "sha256:" + "a" * 64,
    )

    def run(command, **_kwargs):
        captured.extend(command)
        for item in command:
            if item.startswith("type=bind,") and item.endswith(",dst=/output"):
                output = Path(item.split(",")[1].removeprefix("src="))
                (output / "result.json").write_text(json.dumps({"jobs": []}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("corral_md.scoring.local_verifier.subprocess.run", run)
    plan = SimpleNamespace(
        task_number=1, jobs=[{"operation": "sw"}], files={}, challenge="test"
    )
    LocalCalculationTransport(assets)(plan, "fingerprint")

    descriptors = asset_descriptors()
    expected = {
        f"type=bind,src={assets / relative},dst={descriptors[name].public_path},readonly"
        for name, relative in asset_relative_paths().items()
        if Path(relative).parts[0] in {"models", "potentials"}
    }
    assert {item for item in captured if ",dst=/workspace/" in item} == expected
    assert "--read-only" in captured
    assert captured[captured.index("--workdir") + 1] == "/output"
