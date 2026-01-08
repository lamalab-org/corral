"""Tests for the Code2Latex utility."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from corral.backend.env import Environment
from corral.backend.tool import Tool
from corral.utils.code2latex import CacheMetadata, Code2Latex


def dummy_score(answer: dict | str) -> float:
    """Dummy scoring function for tests."""
    return 1.0


@pytest.fixture()
def temp_dir():
    """Create a temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture()
def sample_metadata():
    """Create sample CacheMetadata for testing."""
    return CacheMetadata(
        env_name="test_env",
        level=1,
    )


class TestCacheMetadata:
    """Tests for CacheMetadata dataclass."""

    def test_create_metadata(self, sample_metadata):
        """Test creating CacheMetadata."""
        assert sample_metadata.env_name == "test_env"
        assert sample_metadata.level == 1

    def test_create_metadata_with_string_level(self):
        """Test creating metadata with string level."""
        metadata = CacheMetadata(
            env_name="test_env",
            level="advanced",
        )
        assert metadata.level == "advanced"


class TestCode2LatexCachePath:
    """Tests for cache path generation."""

    def test_get_cache_path(self, sample_metadata, temp_dir):
        """Test cache path generation."""
        cache_path = Code2Latex._get_cache_path(sample_metadata, Path(temp_dir))
        assert cache_path == Path(temp_dir) / "test_env_level_1.json"

    def test_get_cache_path_string_level(self, temp_dir):
        """Test cache path with string level."""
        metadata = CacheMetadata(env_name="catalyst", level="advanced")
        cache_path = Code2Latex._get_cache_path(metadata, Path(temp_dir))
        assert cache_path == Path(temp_dir) / "catalyst_level_advanced.json"


class TestCode2LatexCacheOperations:
    """Tests for cache save/load operations."""

    def test_save_and_load_cache(self, temp_dir):
        """Test saving and loading cache."""
        cache_path = Path(temp_dir) / "test_cache.json"
        test_data = {
            "metadata": {"env_name": "test", "level": 1},
            "main_task": {"task_id": "t1", "name": "Task 1"},
            "subtasks": [],
        }

        Code2Latex._save_cache(cache_path, test_data)
        loaded = Code2Latex._load_cache(cache_path)

        assert loaded == test_data

    def test_load_nonexistent_cache(self, temp_dir):
        """Test loading non-existent cache returns None."""
        cache_path = Path(temp_dir) / "nonexistent.json"
        result = Code2Latex._load_cache(cache_path)
        assert result is None

    def test_load_malformed_cache(self, temp_dir):
        """Test loading malformed JSON returns None."""
        cache_path = Path(temp_dir) / "malformed.json"
        cache_path.write_text("{ invalid json }")
        result = Code2Latex._load_cache(cache_path)
        assert result is None


class TestCode2LatexMerge:
    """Tests for task data merging."""

    def test_merge_main_task(self):
        """Test merging a main task into empty cache."""
        existing = {
            "metadata": {"env_name": "test", "level": 1},
            "main_task": None,
            "subtasks": [],
        }
        task_dict = {
            "task_id": "test_task_1",
            "name": "Test Task",
            "tools": ["tool_a"],
            "is_subtask": False,
        }
        result = Code2Latex._merge_task_data(existing, task_dict)

        assert result["main_task"] is not None
        assert result["main_task"]["task_id"] == "test_task_1"

    def test_merge_subtask(self):
        """Test merging a subtask into cache."""
        existing = {
            "metadata": {"env_name": "test", "level": 1},
            "main_task": {"task_id": "main", "name": "Main Task"},
            "subtasks": [],
        }
        subtask_dict = {
            "task_id": "test_task_1_subtask_1",
            "name": "Test Subtask 1",
            "tools": ["subtool_a"],
            "is_subtask": True,
        }
        result = Code2Latex._merge_task_data(existing, subtask_dict)

        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["task_id"] == "test_task_1_subtask_1"

    def test_merge_update_existing_subtask(self):
        """Test updating an existing subtask."""
        existing = {
            "metadata": {"env_name": "test", "level": 1},
            "main_task": None,
            "subtasks": [
                {"task_id": "test_task_1_subtask_1", "name": "Old Name", "tools": []},
            ],
        }
        subtask_dict = {
            "task_id": "test_task_1_subtask_1",
            "name": "Test Subtask 1",
            "tools": ["subtool_a"],
            "is_subtask": True,
        }
        result = Code2Latex._merge_task_data(existing, subtask_dict)

        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["name"] == "Test Subtask 1"


class TestCode2LatexGeneration:
    """Tests for LaTeX content generation."""

    def test_generate_main_task_colorbox(self):
        """Test generating colorbox for main task."""
        task_dict = {
            "task_id": "test_task_1",
            "name": "Test Task",
            "description": "Test description",
            "tools": ["tool_a", "tool_b"],
            "scoring_function": "score",
        }
        latex = Code2Latex._generate_task_colorbox(task_dict, is_main=True)

        assert r"\begin{tcolorbox}" in latex
        assert "Test Task" in latex
        assert "blue!5" in latex
        assert r"\texttt{tool\_a}" in latex

    def test_generate_subtask_colorbox(self):
        """Test generating colorbox for subtask."""
        task_dict = {
            "task_id": "test_subtask_1",
            "name": "Test Subtask 1",
            "description": "Subtask description",
            "tools": ["subtool_a"],
            "scoring_function": "score_subtask",
            "is_subtask": True,
        }
        latex = Code2Latex._generate_task_colorbox(
            task_dict, is_main=False, subtask_num=1
        )

        assert r"\begin{tcolorbox}" in latex
        assert "Subtask 1:" in latex
        assert "gray!5" in latex
        assert r"\end{tcolorbox}" in latex

    def test_generate_subtask_with_dependencies(self):
        """Test generating colorbox for subtask with dependencies."""
        task_dict = {
            "task_id": "test_subtask_2",
            "name": "Test Subtask 2",
            "description": "Subtask with deps",
            "tools": ["subtool_a"],
            "scoring_function": "score_subtask",
            "is_subtask": True,
            "input_from_tasks": ["subtask_1", "main_task"],
        }
        latex = Code2Latex._generate_task_colorbox(
            task_dict, is_main=False, subtask_num=2
        )

        assert r"\textbf{Depends on:}" in latex
        assert r"\texttt{subtask\_1}" in latex
        assert r"\texttt{main\_task}" in latex

    def test_escape_latex_special_chars(self):
        """Test escaping special LaTeX characters."""
        text = "Test & % $ # _ { } ~ ^"
        escaped = Code2Latex._escape_latex(text)

        assert r"\&" in escaped
        assert r"\%" in escaped
        assert r"\$" in escaped
        assert r"\#" in escaped
        assert r"\_" in escaped


class TestCode2LatexColorbox:
    """Tests for the main colorbox method."""

    def test_colorbox_creates_files(self, sample_metadata, temp_dir):
        """Test that colorbox creates both cache and output files."""
        output_dir = Path(temp_dir) / "output"
        cache_dir = Path(temp_dir) / "cache"

        result_path = Code2Latex.colorbox(
            name="Test Task",
            description="This is a test task description.",
            tools=["tool_a", "tool_b", "tool_c"],
            scoring_fn=dummy_score,
            metadata=sample_metadata,
            output_dir=str(output_dir),
            task_id="test_task_1",
            cache_dir=str(cache_dir),
        )

        # Check output file created
        assert Path(result_path).exists()
        assert result_path.endswith(".tex")

        # Check cache file created
        cache_file = cache_dir / "test_env_level_1.json"
        assert cache_file.exists()

    def test_colorbox_accumulates_subtasks(self, sample_metadata, temp_dir):
        """Test that subsequent calls accumulate subtasks."""
        output_dir = Path(temp_dir) / "output"
        cache_dir = Path(temp_dir) / "cache"

        # First call with main task
        Code2Latex.colorbox(
            name="Main Task",
            description="Main task description.",
            tools=["tool_a", "tool_b", "tool_c"],
            scoring_fn=dummy_score,
            metadata=sample_metadata,
            output_dir=str(output_dir),
            task_id="test_task_main",
            cache_dir=str(cache_dir),
        )

        # Second call with subtask
        Code2Latex.colorbox(
            name="Test Subtask 1",
            description="This is a test subtask description.",
            tools=["subtool_a"],
            scoring_fn=dummy_score,
            metadata=sample_metadata,
            output_dir=str(output_dir),
            task_id="test_task_subtask_1",
            is_subtask=True,
            subtask_index=1,
            input_from_tasks=["test_task_main"],
            cache_dir=str(cache_dir),
        )

        # Check cache has both
        cache_file = cache_dir / "test_env_level_1.json"
        with cache_file.open() as f:
            cache_data = json.load(f)

        assert cache_data["main_task"] is not None
        assert len(cache_data["subtasks"]) == 1
        assert cache_data["subtasks"][0]["input_from_tasks"] == ["test_task_main"]


class TestCode2LatexClearCache:
    """Tests for cache clearing."""

    def test_clear_specific_cache(self, temp_dir):
        """Test clearing a specific cache file."""
        cache_dir = Path(temp_dir)
        (cache_dir / "test_level_1.json").write_text("{}")
        (cache_dir / "test_level_2.json").write_text("{}")

        deleted = Code2Latex.clear_cache(
            env_name="test", level=1, cache_dir=str(cache_dir)
        )

        assert deleted == 1
        assert not (cache_dir / "test_level_1.json").exists()
        assert (cache_dir / "test_level_2.json").exists()

    def test_clear_env_cache(self, temp_dir):
        """Test clearing all cache for an environment."""
        cache_dir = Path(temp_dir)
        (cache_dir / "test_level_1.json").write_text("{}")
        (cache_dir / "test_level_2.json").write_text("{}")
        (cache_dir / "other_level_1.json").write_text("{}")

        deleted = Code2Latex.clear_cache(env_name="test", cache_dir=str(cache_dir))

        assert deleted == 2
        assert (cache_dir / "other_level_1.json").exists()

    def test_clear_all_cache(self, temp_dir):
        """Test clearing all cache files."""
        cache_dir = Path(temp_dir)
        (cache_dir / "test_level_1.json").write_text("{}")
        (cache_dir / "other_level_1.json").write_text("{}")

        deleted = Code2Latex.clear_cache(cache_dir=str(cache_dir))

        assert deleted == 2


class TestEnvironmentToLatex:
    """Tests for Environment.to_latex() method."""

    def test_to_latex_method_exists(self):
        """Test that to_latex method exists on Environment."""
        assert hasattr(Environment, "to_latex")

    def test_to_latex_generates_file(self, temp_dir):
        """Test that to_latex generates a LaTeX file."""

        # Create a concrete implementation of Environment for testing
        class TestEnvironment(Environment):
            def get_task_prompt(self):
                return "Test task prompt"

            def score(self):
                return 1.0

        env = TestEnvironment(task_id="test_env_task", base_work_dir=temp_dir)

        # Add a test tool
        test_tool = Tool(
            name="test_tool",
            description="A test tool",
            arguments=[],
            func=lambda: "result",
        )
        env.add_tool(test_tool)

        output_dir = Path(temp_dir) / "latex_output"
        cache_dir = Path(temp_dir) / "latex_cache"

        task_result, tools_result = env.to_latex(
            output_dir=str(output_dir),
            env_name="test",
            level=1,
            cache_dir=str(cache_dir),
        )

        # Check task file
        assert Path(task_result).exists()
        content = Path(task_result).read_text()
        assert "Test task prompt" in content
        assert "test_tool" in content

        # Check tools file
        assert Path(tools_result).exists()
        tools_content = Path(tools_result).read_text()
        assert "test_tool" in tools_content
        assert "longtable" in tools_content


class TestLatexOutput:
    """Tests for LaTeX output format."""

    def test_latex_has_required_packages_comment(self, sample_metadata, temp_dir):
        """Test that LaTeX output has package requirements comment."""
        output_dir = Path(temp_dir) / "output"

        result_path = Code2Latex.colorbox(
            name="Test Task",
            description="Test description.",
            tools=["tool_a", "tool_b", "tool_c"],
            scoring_fn=dummy_score,
            metadata=sample_metadata,
            output_dir=str(output_dir),
            task_id="test_task",
            cache_dir=str(temp_dir),
        )

        content = Path(result_path).read_text()
        assert "tcolorbox" in content
        assert "listings" in content

    def test_latex_has_correct_structure(self, sample_metadata, temp_dir):
        """Test that LaTeX output has correct structure."""
        output_dir = Path(temp_dir) / "output"

        result_path = Code2Latex.colorbox(
            name="Test Task",
            description="Test description.",
            tools=["tool_a", "tool_b", "tool_c"],
            scoring_fn=dummy_score,
            metadata=sample_metadata,
            output_dir=str(output_dir),
            task_id="test_task",
            cache_dir=str(temp_dir),
        )

        content = Path(result_path).read_text()

        # Check for main structural elements
        assert r"\begin{tcolorbox}" in content
        assert r"\end{tcolorbox}" in content
        assert r"\textbf{Tools:}" in content
        assert r"\textbf{Scoring Function:}" in content
