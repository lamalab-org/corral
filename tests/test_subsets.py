"""Tests for released corral-mini task subsets."""

import os

import pytest

from corral.subsets import load_mini, mini_task_ids, mini_versions


def test_every_release_has_k_distinct_tasks_per_level():
    for version in mini_versions():
        manifest = load_mini(version)
        assert manifest["version"] == version
        levels = [
            ids
            for by_level in manifest["tasks_by_environment"].values()
            for ids in by_level.values()
        ]
        assert all(len(set(ids)) == manifest["tasks_per_level"] for ids in levels)
        assert sum(map(len, levels)) == manifest["total_tasks"]


def test_latest_release_is_the_default():
    assert load_mini() == load_mini(mini_versions()[-1])


def test_v2_uses_pass_hat_5_and_unsaturated_tasks():
    manifest = load_mini(2)
    assert manifest["method"] == "rank_cost"
    assert manifest["score_metric"] == "pass_hat_5"
    assert manifest["exclude_saturated"] is True
    assert manifest["random_environments"] == []


def test_uncovered_levels_return_none():
    assert mini_task_ids("samplemath", 1) is None
    assert mini_task_ids("wetlab", 1) is None


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError, match="unknown corral-mini version"):
        load_mini(max(mini_versions()) + 1)


@pytest.mark.skipif(
    not os.environ.get("CORRAL_CHECK_MINI_TASKS"),
    reason="loads every environment; set CORRAL_CHECK_MINI_TASKS=1",
)
@pytest.mark.parametrize(
    ("environment", "level"),
    [
        (environment, int(level.removeprefix("level_")))
        for environment, levels in load_mini()["tasks_by_environment"].items()
        for level in levels
    ],
)
def test_release_task_ids_exist_in_environment(environment, level):
    from corral.runtime.environment_loader import load_environment_group

    try:
        loaded = load_environment_group(environment, env_kwargs={"level": level})
    except ImportError as exc:
        pytest.skip(f"{environment} dependencies are not installed: {exc}")
    assert set(mini_task_ids(environment, level)) <= set(loaded)
