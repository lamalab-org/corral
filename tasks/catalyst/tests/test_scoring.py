import os
from pathlib import Path

import pytest

# Import the scoring functions to test
from catalyst.score import (
    check_valid_json_file,
)
from hypothesis import given
from hypothesis import strategies as st

os.environ["CORRAL_WORK_DIR"] = str(Path(__file__).parent / "test_score_files")
TEST_SCORE_FILES = Path(os.environ["CORRAL_WORK_DIR"])


class TestBasicFileValidation:
    """Test basic file validation functions."""

    @pytest.fixture()
    def sample_json_files(self):
        """Create sample JSON files for testing."""
        tmp_path = Path(TEST_SCORE_FILES, "json_files")
        files = {}

        # Valid JSON files
        valid_simple = tmp_path / "valid_simple.json"
        files["valid_simple"] = str(valid_simple)

        valid_complex = tmp_path / "valid_complex.json"
        files["valid_complex"] = str(valid_complex)

        # Invalid JSON files
        invalid_syntax = tmp_path / "invalid_syntax.json"
        files["invalid_syntax"] = str(invalid_syntax)

        # Empty file
        empty_file = tmp_path / "empty.json"
        files["empty"] = str(empty_file)

        # Null content
        null_content = tmp_path / "null_content.json"
        files["null_content"] = str(null_content)

        # Non-JSON file
        text_file = tmp_path / "not_json.txt"
        files["not_json"] = str(text_file)

        return files

    def test_check_valid_json_file_with_valid_files(self, sample_json_files):
        """Test check_valid_json_file with valid JSON files."""
        assert check_valid_json_file(sample_json_files["valid_simple"]) == 1.0
        assert check_valid_json_file(sample_json_files["valid_complex"]) == 1.0

    def test_check_valid_json_file_with_invalid_files(self, sample_json_files):
        """Test check_valid_json_file with invalid files."""
        assert check_valid_json_file(sample_json_files["invalid_syntax"]) == 0.0
        assert check_valid_json_file(sample_json_files["empty"]) == 0.0
        assert check_valid_json_file(sample_json_files["not_json"]) == 0.0

    def test_check_valid_json_file_with_null_content(self, sample_json_files):
        """Test check_valid_json_file with null JSON content."""
        # Null is valid JSON but should return 0.0 according to the function
        assert check_valid_json_file(sample_json_files["null_content"]) == 0.0

    def test_check_valid_json_file_with_nonexistent_file(self):
        """Test check_valid_json_file with nonexistent file."""
        assert check_valid_json_file("nonexistent_file.json") == 0.0

    def test_check_valid_json_file_with_empty_input(self):
        """Test check_valid_json_file with empty/invalid input."""
        assert check_valid_json_file("") == 0.0
        assert check_valid_json_file("   ") == 0.0
        assert check_valid_json_file(None) == 0.0

    def test_check_valid_json_file_with_directory(self, tmp_path):
        """Test check_valid_json_file with directory instead of file."""
        test_dir = tmp_path / "test_directory"
        test_dir.mkdir()
        assert check_valid_json_file(str(test_dir)) == 0.0

    @given(st.text())
    def test_check_valid_json_file_property_always_returns_float(self, input_text):
        """Property test: check_valid_json_file should always return a float."""
        result = check_valid_json_file(input_text)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
