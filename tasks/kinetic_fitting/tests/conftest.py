"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def data_path():
    """Provide path to experimental data file."""
    return Path(__file__).parent.parent / "data" / "experimental_data.h5"


@pytest.fixture
def task_json_path():
    """Provide path to task JSON file."""
    return Path(__file__).parent.parent / "tasks" / "kinetic_fitting_tasks.json"


@pytest.fixture
def sample_network():
    """Provide a sample reaction network for testing."""
    return {
        'reactions': [
            {'equation': 'RuII + hv -> RuII*', 'type': 'light', 'quantum_yield': [0.8, 1.0]},
            {'equation': 'RuII* + S2O8 -> RuIII + SO4_rad + SO4', 'type': 'dark', 'k_range': [1e7, 1e9]},
            {'equation': '2 RuIII + H2O2 -> 2 RuII + O2 + 2 H', 'type': 'dark', 'k_range': [1e3, 1e5]},
        ]
    }