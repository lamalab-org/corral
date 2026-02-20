"""Tests for ml_code environment setup and tool registration."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_code.env import (
    KEEP_TOOLS,
    SandboxAwareEnvironment,
    _create_sandbox_config,
    _get_retrieval_tools,
    create_environments,
)


# Tools that should NOT be present (ML-specific tools from the original ml task)
EXCLUDED_TOOLS = {
    "sort_and_get_first_from_json",
    "select_polymorphs_with_strategy",
    "select_polymorphs_with_strategy_to_file",
    "consolidate_polymorph_datasets",
    "filter_json_with_strategy",
    "prepare_tabular_dataset",
    "train_xgboost_model",
    "evaluate_xgboost_model",
    "perform_cross_validation",
}


class TestRetrievalTools:
    """Test that only retrieval tools are selected from ml.tools."""

    def test_get_retrieval_tools_returns_only_keep_set(self):
        """Verify only MP API retrieval tools are returned."""
        tools = _get_retrieval_tools()
        assert set(tools.keys()) == KEEP_TOOLS

    def test_get_retrieval_tools_excludes_ml_tools(self):
        """Verify ML-specific tools are excluded."""
        tools = _get_retrieval_tools()
        for excluded in EXCLUDED_TOOLS:
            assert excluded not in tools

    def test_retrieval_tools_are_tool_instances(self):
        """Verify returned objects are Tool instances."""
        from corral.backend.tool import Tool

        tools = _get_retrieval_tools()
        for name, tool in tools.items():
            assert isinstance(tool, Tool), f"{name} is not a Tool instance"


class TestSandboxConfig:
    """Test sandbox configuration."""

    def test_sandbox_config_uses_subprocess_backend(self):
        config = _create_sandbox_config()
        assert config.backend == "subprocess"

    def test_sandbox_config_has_scientific_packages(self):
        config = _create_sandbox_config()
        package_names = [p.split(">=")[0].split("==")[0] for p in config.python_packages]
        assert "numpy" in package_names
        assert "pandas" in package_names
        assert "scikit-learn" in package_names
        assert "xgboost" in package_names
        assert "joblib" in package_names


class TestEnvironmentCreation:
    """Test create_environments with mocked sandbox."""

    @patch("ml_code.env.create_sandbox_tools")
    def test_single_config_creates_environments(self, mock_create_sandbox, config_dir, tmp_path):
        """Test environment creation from single config."""
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "single" / "single.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        assert len(envs) == 3
        assert "ml_code_oxides" in envs
        assert "ml_code_nitrides" in envs
        assert "ml_code_sulphides" in envs
        mock_sandbox.start.assert_called_once()

    @patch("ml_code.env.create_sandbox_tools")
    def test_chained_config_creates_environments(self, mock_create_sandbox, config_dir, tmp_path):
        """Test environment creation from chained config."""
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "chained" / "chained_oxide.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        assert len(envs) == 4
        assert "batch_retrieve_oxide_polymorphs" in envs
        assert "prepare_ml_ready_dataset" in envs
        assert "train_xgboost_formation_energy_model" in envs
        assert "evaluate_model_performance" in envs
        mock_sandbox.start.assert_called_once()

    @patch("ml_code.env.create_sandbox_tools")
    def test_sandbox_tools_registered_as_common(self, mock_create_sandbox, config_dir, tmp_path):
        """Verify sandbox tools are available in all environments."""
        mock_sandbox = MagicMock()
        mock_exec_tool = MagicMock()
        mock_exec_tool.name = "execute_python_code"
        mock_term_tool = MagicMock()
        mock_term_tool.name = "run_in_terminal"
        mock_sandbox_tools = {
            "execute_python_code": mock_exec_tool,
            "run_in_terminal": mock_term_tool,
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "single" / "single.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        for env_id, env in envs.items():
            assert "execute_python_code" in env.tools, (
                f"execute_python_code missing from {env_id}"
            )
            assert "run_in_terminal" in env.tools, (
                f"run_in_terminal missing from {env_id}"
            )

    @patch("ml_code.env.create_sandbox_tools")
    def test_excluded_tools_not_registered(self, mock_create_sandbox, config_dir, tmp_path):
        """Verify ML-specific tools are not in any environment."""
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "single" / "single.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        for env_id, env in envs.items():
            for excluded in EXCLUDED_TOOLS:
                assert excluded not in env.tools, (
                    f"{excluded} should not be in {env_id}"
                )

    @patch("ml_code.env.create_sandbox_tools")
    def test_retrieval_tools_registered_per_config(self, mock_create_sandbox, config_dir, tmp_path):
        """Verify retrieval tools are registered according to JSON config."""
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "single" / "single.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        # Single config tasks list batch_retrieve_polymorphs, get_bulk_polymorphs_data,
        # get_bulk_polymorphs_data_to_file
        for env_id, env in envs.items():
            assert "batch_retrieve_polymorphs" in env.tools, (
                f"batch_retrieve_polymorphs missing from {env_id}"
            )

    @patch("ml_code.env.create_sandbox_tools")
    def test_file_tools_registered(self, mock_create_sandbox, config_dir, tmp_path):
        """Verify file tools are registered in environments."""
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "single" / "single.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        expected_file_tools = {"list_files", "read_file", "write_file", "file_info", "cat_files", "copy_file"}
        for env_id, env in envs.items():
            for ft in expected_file_tools:
                assert ft in env.tools, f"{ft} missing from {env_id}"


class TestScoringFunctions:
    """Test that scoring functions are correctly wired."""

    @patch("ml_code.env.create_sandbox_tools")
    def test_single_config_uses_evaluation_scoring(self, mock_create_sandbox, config_dir, tmp_path):
        """All single-config tasks use model_evaluation_completeness_binary."""
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "single" / "single.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        from ml.score import model_evaluation_completeness_binary

        for env_id, env in envs.items():
            assert env.current_task.scoring_fn is model_evaluation_completeness_binary

    @patch("ml_code.env.create_sandbox_tools")
    def test_chained_config_uses_correct_scoring(self, mock_create_sandbox, config_dir, tmp_path):
        """Chained config tasks map to the expected scoring functions."""
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "chained" / "chained_oxide.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        from ml.score import (
            ml_dataset_preparation_quality_binary,
            model_evaluation_completeness_binary,
            model_training_success_binary,
            score_polymorph_dataset,
        )

        assert envs["batch_retrieve_oxide_polymorphs"].current_task.scoring_fn is score_polymorph_dataset
        assert envs["prepare_ml_ready_dataset"].current_task.scoring_fn is ml_dataset_preparation_quality_binary
        assert envs["train_xgboost_formation_energy_model"].current_task.scoring_fn is model_training_success_binary
        assert envs["evaluate_model_performance"].current_task.scoring_fn is model_evaluation_completeness_binary


class TestAllChainedConfigs:
    """Verify all chained config variants load without error."""

    @pytest.mark.parametrize(
        "config_name",
        ["chained_oxide.json", "chained_nitride.json", "chained_sulphide.json"],
    )
    @patch("ml_code.env.create_sandbox_tools")
    def test_chained_config_loads(self, mock_create_sandbox, config_name, config_dir, tmp_path):
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "chained" / config_name
        envs = create_environments(config_path, work_dir=str(tmp_path))
        assert len(envs) == 4


class TestSandboxAwareEnvironment:
    """Tests for SandboxAwareEnvironment sandbox-trial coordination."""

    @patch("ml_code.env.create_sandbox_tools")
    def test_sandbox_work_dir_matches_trial_dir(self, mock_create_sandbox, config_dir, tmp_path):
        """Sandbox set_work_dir is called with each environment's current_work_dir."""
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "single" / "single.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        # Collect all work_dirs that set_work_dir was called with
        called_work_dirs = {
            str(call[0][0]) for call in mock_sandbox.set_work_dir.call_args_list
        }

        for env_id, env in envs.items():
            assert isinstance(env, SandboxAwareEnvironment)
            assert env.current_work_dir in called_work_dirs, (
                f"set_work_dir was never called with {env.current_work_dir}"
            )

    @patch("ml_code.env.create_sandbox_tools")
    def test_reset_updates_sandbox_work_dir(self, mock_create_sandbox, config_dir, tmp_path):
        """reset_state() syncs the sandbox to the new trial directory."""
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "single" / "single.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        env = next(iter(envs.values()))
        old_work_dir = env.current_work_dir
        mock_sandbox.set_work_dir.reset_mock()

        env.reset_state()

        # Work dir should have changed (new trial)
        assert env.current_work_dir != old_work_dir
        # Sandbox should have been synced
        mock_sandbox.set_work_dir.assert_called_once()
        call_arg = str(mock_sandbox.set_work_dir.call_args[0][0])
        assert call_arg == env.current_work_dir

    @patch("ml_code.env.create_sandbox_tools")
    def test_prompt_no_raw_path(self, mock_create_sandbox, config_dir, tmp_path):
        """Task prompt should not contain raw filesystem paths."""
        mock_sandbox = MagicMock()
        mock_sandbox_tools = {
            "execute_python_code": MagicMock(),
            "run_in_terminal": MagicMock(),
        }
        mock_create_sandbox.return_value = (mock_sandbox, mock_sandbox_tools)

        config_path = config_dir / "single" / "single.json"
        envs = create_environments(config_path, work_dir=str(tmp_path))

        for env_id, env in envs.items():
            prompt = env.get_task_prompt()
            # The prompt should not contain the raw tmp_path
            assert str(tmp_path) not in prompt, (
                f"Raw filesystem path leaked into prompt for {env_id}"
            )
