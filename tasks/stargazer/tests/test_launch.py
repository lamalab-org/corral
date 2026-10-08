import json
from dataclasses import asdict

import pytest
from stargazer.bank import generate_bank
from stargazer.env import DATA_ROOT
from stargazer.launch import verify_launch
from stargazer.protocol import ASSISTED_PROTOCOL
from stargazer.score import EvaluationCriteria


@pytest.fixture
def frozen_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "stargazer.bank.audit_identifiability", lambda _task, _rules: {"accepted": True}
    )
    calibration = tmp_path / "calibration"
    generate_bank(calibration, per_level=1, levels=(1,))
    evaluation = tmp_path / "evaluation"
    manifest = generate_bank(
        evaluation, per_level=10, levels=(1,), calibration=calibration
    )
    config = {
        "bank_hash": manifest["bank_hash"],
        "calibration_hash": manifest["calibration_hash"],
        "criteria": asdict(EvaluationCriteria()),
        "protocol": ASSISTED_PROTOCOL,
        "level": 1,
        "task_count": 10,
    }
    return evaluation, config


def test_launch_resolves_exact_bank_and_execution_identity(frozen_launch):
    root, config = frozen_launch
    result = verify_launch(config, root)
    assert result["task_count"] == 10
    assert result["bank_hash"] == config["bank_hash"]
    assert result["execution_version"].startswith(ASSISTED_PROTOCOL + ":")
    selector = json.loads((root / "selectors/level_1.json").read_text())
    assert set(result["task_ids"]) == set(selector[0]["task_ids"])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("bank_hash", "wrong"),
        ("calibration_hash", "wrong"),
        ("protocol", "wrong"),
        ("criteria", {}),
        ("task_count", 9),
        ("level", 2),
    ],
)
def test_launch_rejects_mismatch_before_runner(frozen_launch, key, value):
    root, config = frozen_launch
    with pytest.raises(ValueError):
        verify_launch({**config, key: value}, root)


def test_launch_rejects_bundled_bank(frozen_launch):
    _, config = frozen_launch
    with pytest.raises(FileNotFoundError):
        verify_launch(config, DATA_ROOT)
