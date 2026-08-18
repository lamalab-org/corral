from datetime import timedelta

import pytest

from corral.orchestration.models import ActivityPolicy
from corral.orchestration.workflows import _activity_options


def test_default_activity_policy_disables_public_timeouts():
    policy = ActivityPolicy()
    options = _activity_options(policy)

    assert policy.start_to_close_seconds is None
    assert policy.heartbeat_timeout_seconds is None
    assert "start_to_close_timeout" not in options
    assert "heartbeat_timeout" not in options
    # Temporal requires one closing timeout. Corral uses a practical-infinity
    # Schedule-to-Close value only to satisfy that protocol invariant.
    assert options["schedule_to_close_timeout"] == timedelta(days=365 * 100)


def test_explicit_activity_timeouts_are_forwarded():
    options = _activity_options(
        ActivityPolicy(
            start_to_close_seconds=120.0,
            heartbeat_timeout_seconds=15.0,
        )
    )

    assert options["start_to_close_timeout"] == timedelta(seconds=120)
    assert options["heartbeat_timeout"] == timedelta(seconds=15)
    assert "schedule_to_close_timeout" not in options


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_to_close_seconds": 0},
        {"start_to_close_seconds": -1},
        {"heartbeat_timeout_seconds": 0},
        {"heartbeat_timeout_seconds": -1},
    ],
)
def test_configured_activity_timeouts_must_be_positive(kwargs):
    with pytest.raises(ValueError, match="greater than 0"):
        ActivityPolicy(**kwargs)
