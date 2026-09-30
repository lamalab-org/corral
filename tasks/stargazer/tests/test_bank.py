import json
from dataclasses import asdict

import pytest
from stargazer.bank import generate_bank, generate_record, verify_bank
from stargazer.env import create_environments
from stargazer.identifiability import BankRules, audit_identifiability
from stargazer.models import load_task


def test_fresh_generator_is_reproducible_and_uses_rv_only(tmp_path):
    one = generate_record(123456, "opaque", 5)
    assert one == generate_record(123456, "opaque", 5)
    assert one["observations"] != generate_record(987654, "opaque", 5)["observations"]
    path = tmp_path / "record.json"
    path.write_text(json.dumps(one))
    task = load_task(path)
    assert task.public_fit_context().observations.rvs_ms == tuple(
        one["observations"]["rvs_ms"]
    )
    assert "withheld_observations" not in json.dumps(task.public_observation())
    assert "seed" not in json.dumps(task.public_observation())


def test_freeze_requires_independent_calibration_and_detects_tampering(
    tmp_path, monkeypatch
):
    # Exercise packaging independently from the expensive scientific audit.
    monkeypatch.setattr(
        "stargazer.bank.audit_identifiability",
        lambda _task, rules: {"accepted": True, "rules": asdict(rules)},
    )
    calibration = tmp_path / "calibration"
    original = generate_bank(calibration, per_level=1)
    evaluation = tmp_path / "evaluation"
    frozen = generate_bank(evaluation, per_level=1, calibration=calibration)
    assert frozen["calibration_hash"] == original["bank_hash"]
    assert not set(frozen["membership"]["1"]) & set(original["membership"]["1"])
    for level in (1, 2):
        envs = create_environments(
            level=level,
            data_root=evaluation,
            work_dir=tmp_path / "work",
            development_mode=True,
        )
        assert len(envs) == 1
        task = next(iter(envs.values())).current_task
        assert task.scoring_fn.bank_hash == frozen["bank_hash"]
        assert not task.allow_previous_attempt_context
    with pytest.raises(FileExistsError):
        generate_bank(evaluation, per_level=1, calibration=calibration)
    with pytest.raises(ValueError, match="Expected a evaluation"):
        create_environments(data_root=calibration)
    file = next((evaluation / "synthetic").glob("*.json"))
    file.write_text(file.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bank(evaluation)


def test_unfrozen_generation_cannot_be_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "stargazer.bank.audit_identifiability",
        lambda _task, _rules: {"accepted": False},
    )
    with pytest.raises(RuntimeError, match="unfrozen"):
        generate_bank(tmp_path / "partial", per_level=1, max_attempts=2)
    assert not (tmp_path / "partial" / "private-manifest.json").exists()


def test_identifiability_audit_flags_inadequate_observation_span(simple_task):
    rules = BankRules(starts=2, bootstraps=2, max_planets=2, minimum_cycles=100)
    result = audit_identifiability(simple_task, rules)
    assert result["checks"]["parameter_recovery"]
    assert result["checks"]["count_recovery"]
    assert not result["checks"]["cycles"]
    assert not result["accepted"]
    assert len(result["bootstrap_counts"]) == 2
    assert len(result["competing_fits"]) == 3
    assert len(result["competing_fits"][1]["start_bics"]) == 2
