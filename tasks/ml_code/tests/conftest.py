import os
from pathlib import Path

import pytest

os.environ["CORRAL_WORK_DIR"] = str(Path(__file__).parent / "test_data")
TEMP_DIR = Path(os.environ["CORRAL_WORK_DIR"])


@pytest.fixture()
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return str(tmp_path)


@pytest.fixture()
def config_dir():
    """Return the path to the config directory."""
    return Path(__file__).parent.parent / "config"
