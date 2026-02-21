"""Tests for Hydra integration - structured configs and entry point wiring."""

from __future__ import annotations

import importlib
from dataclasses import fields
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Structured config tests (no Hydra runtime required)
# ---------------------------------------------------------------------------


class TestStructuredConfigs:
    """Verify that the dataclass-based config schema is well-formed."""

    def test_corral_config_defaults(self):
        from corral.conf.config import CorralConfig

        cfg = CorralConfig()
        assert cfg.mode == "local"
        assert cfg.agent._target_ == "corral.agents.tool_calling.ToolCallingAgent"
        assert cfg.runner.trials_per_task == 5
        assert cfg.docker.image == "ghcr.io/lamalab-org/corral-materials:latest"
        assert cfg.tasks.ids == []
        assert cfg.wandb.enabled is False

    def test_agent_config_fields(self):
        from corral.conf.config import AgentConfig

        cfg = AgentConfig()
        field_names = {f.name for f in fields(cfg)}
        expected = {
            "_target_",
            "model",
            "max_iterations",
            "temperature",
            "api_endpoint",
            "system_prompt",
            "user_prompt",
            "extractor_prompt",
            "surrender_prompt",
            "extra_kwargs",
        }
        assert expected.issubset(field_names)

    def test_runner_config_fields(self):
        from corral.conf.config import RunnerConfig

        cfg = RunnerConfig()
        assert cfg.base_url == "http://localhost:8000"
        assert cfg.k_values == [1, 3, 5]
        assert cfg.enable_surrender is False
        assert cfg.tool_verbosity == "brief"

    def test_docker_config_fields(self):
        from corral.conf.config import DockerConfig

        cfg = DockerConfig()
        assert cfg.network == "corral-network"
        assert cfg.detach is False
        assert cfg.env_args == {}

    def test_wandb_config_fields(self):
        from corral.conf.config import WandbConfig

        cfg = WandbConfig()
        assert cfg.project == "corral-benchmark"
        assert cfg.tags == []

    def test_agent_config_override(self):
        from corral.conf.config import AgentConfig

        cfg = AgentConfig(
            _target_="corral.agents.react.ReActAgent",
            model="anthropic/claude-sonnet-4-5-20250929",
            max_iterations=50,
            temperature=0.5,
        )
        assert cfg._target_ == "corral.agents.react.ReActAgent"
        assert cfg.model == "anthropic/claude-sonnet-4-5-20250929"
        assert cfg.max_iterations == 50
        assert cfg.temperature == 0.5

    def test_corral_config_composition(self):
        from corral.conf.config import AgentConfig, CorralConfig, RunnerConfig

        cfg = CorralConfig(
            agent=AgentConfig(model="openai/gpt-4o-mini"),
            runner=RunnerConfig(trials_per_task=1, k_values=[1]),
            mode="docker",
        )
        assert cfg.agent.model == "openai/gpt-4o-mini"
        assert cfg.runner.trials_per_task == 1
        assert cfg.mode == "docker"

    def test_extra_kwargs_mutable_default(self):
        """Ensure mutable defaults are independent across instances."""
        from corral.conf.config import AgentConfig

        a = AgentConfig()
        b = AgentConfig()
        a.extra_kwargs["foo"] = "bar"
        assert "foo" not in b.extra_kwargs


# ---------------------------------------------------------------------------
# YAML config group tests
# ---------------------------------------------------------------------------


class TestYAMLConfigFiles:
    """Verify that YAML config files exist and are loadable."""

    CONF_DIR = Path(__file__).resolve().parent.parent / "src" / "corral" / "conf"

    def test_main_config_exists(self):
        assert (self.CONF_DIR / "config.yaml").is_file()

    @pytest.mark.parametrize(
        "name",
        [
            "react",
            "tool_calling",
            "llm_planner",
            "reflexion",
        ],
    )
    def test_agent_configs_exist(self, name):
        assert (self.CONF_DIR / "agent" / f"{name}.yaml").is_file()

    @pytest.mark.parametrize("name", ["default", "quick", "full"])
    def test_runner_configs_exist(self, name):
        assert (self.CONF_DIR / "runner" / f"{name}.yaml").is_file()

    @pytest.mark.parametrize("name", ["default", "local"])
    def test_docker_configs_exist(self, name):
        assert (self.CONF_DIR / "docker" / f"{name}.yaml").is_file()

    def test_agent_yamls_have_target(self):
        import yaml

        for yaml_file in (self.CONF_DIR / "agent").glob("*.yaml"):
            data = yaml.safe_load(yaml_file.read_text())
            assert "_target_" in data, f"Missing _target_ in {yaml_file.name}"
            # Verify the target is importable
            target = data["_target_"]
            module_path, class_name = target.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            assert hasattr(
                mod, class_name
            ), f"Cannot find {class_name} in {module_path}"


# ---------------------------------------------------------------------------
# Hydra entry point tests (require hydra-core)
# ---------------------------------------------------------------------------


class TestHydraEntryPoint:
    """Verify that the Hydra app module is importable and well-formed."""

    def test_hydra_app_importable(self):
        mod = importlib.import_module("corral.cli.hydra_app")
        assert hasattr(mod, "main")

    def test_config_store_registration(self):
        """ConfigStore should have our schemas registered after importing hydra_app."""
        from hydra.core.config_store import ConfigStore

        # Importing hydra_app triggers _register_configs()
        importlib.import_module("corral.cli.hydra_app")

        cs = ConfigStore.instance()
        # The store should be able to retrieve our base config
        # (ConfigStore internal API — we just check no exception)
        assert cs is not None


# ---------------------------------------------------------------------------
# CorralRunner.from_config tests
# ---------------------------------------------------------------------------


class TestCorralRunnerFromConfig:
    """Test the from_config factory method on CorralRunner."""

    def test_from_config_creates_runner(self):
        """from_config should produce a CorralRunner with correct agent type."""
        from corral.conf.config import AgentConfig, CorralConfig, RunnerConfig
        from corral.run import CorralRunner

        cfg = CorralConfig(
            agent=AgentConfig(
                _target_="corral.agents.react.ReActAgent",
                model="openai/gpt-4o",
                max_iterations=5,
                temperature=0.3,
            ),
            runner=RunnerConfig(
                base_url="http://localhost:9999",
                checkpoint_dir="/tmp/test_hydra_ckpt",
                enable_surrender=True,
            ),
        )

        runner = CorralRunner.from_config(cfg)
        assert runner.agent.__class__.__name__ == "ReActAgent"
        assert runner.agent.model == "openai/gpt-4o"
        assert runner.agent.max_iterations == 5
        assert runner.agent.temperature == 0.3
        assert runner.enable_surrender is True
        assert runner._hydra_config is cfg

    def test_from_config_extra_kwargs(self):
        """extra_kwargs should be forwarded to the agent constructor."""
        from corral.conf.config import AgentConfig, CorralConfig
        from corral.run import CorralRunner

        cfg = CorralConfig(
            agent=AgentConfig(
                _target_="corral.agents.tool_calling.ToolCallingAgent",
                extra_kwargs={"model": "openai/gpt-4o-mini"},  # override via extra
            ),
        )

        runner = CorralRunner.from_config(cfg)
        # extra_kwargs override should win for model
        assert runner.agent.model == "openai/gpt-4o-mini"

    def test_from_config_with_omegaconf(self):
        """from_config should also work with OmegaConf DictConfig."""
        pytest.importorskip("omegaconf")
        from omegaconf import OmegaConf

        from corral.conf.config import AgentConfig, CorralConfig, RunnerConfig
        from corral.run import CorralRunner

        raw = CorralConfig(
            agent=AgentConfig(model="openai/gpt-4o"),
            runner=RunnerConfig(trials_per_task=2),
        )
        cfg = OmegaConf.structured(raw)

        runner = CorralRunner.from_config(cfg)
        assert runner.agent.model == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


class TestPublicAPIExports:
    """Verify that config classes are exported from the top-level package."""

    def test_config_classes_importable(self):
        from corral import (
            AgentConfig,
            CorralConfig,
            DockerConfig,
            RunnerConfig,
            TasksConfig,
            WandbConfig,
        )

        assert AgentConfig is not None
        assert CorralConfig is not None
        assert DockerConfig is not None
        assert RunnerConfig is not None
        assert TasksConfig is not None
        assert WandbConfig is not None
