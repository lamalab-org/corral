import os
import tempfile
import time
from pathlib import Path

import pytest
from catalyst.utils import (
    extract_path_from_answer,
    find_file_by_name,
    smart_resolve_path,
)


@pytest.fixture()
def temp_dir_structure():
    """
    Creates a temporary directory with a predefined file structure for testing.
    Yields the path to the temporary base directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Create files
        (base_path / "file1.txt").write_text("content1")
        (base_path / "data.json").write_text("{}")
        (base_path / "report.pdf").write_text("%PDF-1.4...")

        # Create a subdirectory and files within it
        subdir1 = base_path / "subdir1"
        subdir1.mkdir()
        (subdir1 / "nested_file.log").write_text("log content")
        (subdir1 / "config.ini").write_text("[settings]")

        # Create another subdirectory with a file of the same name
        subdir2 = base_path / "subdir2"
        subdir2.mkdir()
        # Create an older file
        time.sleep(0.01)  # Ensure different modification times
        (subdir2 / "data.json").write_text("{'old': true}")
        time.sleep(0.01)  # Ensure different modification times
        # Create a newer file
        (base_path / "data.json").write_text("{'new': true}")

        # Set CORRAL_WORK_DIR for tests that rely on it
        os.environ["CORRAL_WORK_DIR"] = str(base_path)

        yield base_path

        # Cleanup is handled by TemporaryDirectory context manager
        del os.environ["CORRAL_WORK_DIR"]


# --- Tests for extract_path_from_answer ---
def test_extract_path_from_answer_basic_filename():
    assert extract_path_from_answer("filename.txt") == "filename.txt"

    # def test_extract_path_from_answer_with_prefix():
    #     assert (
    #         extract_path_from_answer("final answer is path/to/file.json")
    #         == "path/to/file.json"
    #     )
    assert extract_path_from_answer("answer: `another_file.py`") == "another_file.py"


def test_extract_path_from_answer_markdown_backticks():
    assert (
        extract_path_from_answer("The file is `path/to/document.pdf`.")
        == "path/to/document.pdf"
    )
    assert extract_path_from_answer("Please check `config.yaml`.") == "config.yaml"


def test_extract_path_from_answer_double_quotes():
    assert (
        extract_path_from_answer('The data is in "results/output.csv".')
        == "results/output.csv"
    )
    assert extract_path_from_answer('"image.png"') == "image.png"


def test_extract_path_from_answer_single_quotes():
    assert extract_path_from_answer("Open 'report.docx'.") == "report.docx"
    assert extract_path_from_answer("'my_script.sh'") == "my_script.sh"


def test_extract_path_from_answer_absolute_path():
    assert extract_path_from_answer("/var/log/syslog.log") == "/var/log/syslog.log"
    assert (
        extract_path_from_answer("The file is at /home/user/documents/report.txt.")
        == "/home/user/documents/report.txt"
    )


def test_extract_path_from_answer_no_file_extension():
    # The current regex specifically looks for extensions, so these should return the original string
    assert extract_path_from_answer("just_a_word") == "just_a_word"
    assert extract_path_from_answer("path/to/directory/") == "path/to/directory/"
    assert (
        extract_path_from_answer("`no_extension`") == "`no_extension`"
    )  # No match for file pattern within backticks
    assert (
        extract_path_from_answer('"no_extension"') == '"no_extension"'
    )  # No match for file pattern within quotes


def test_extract_path_from_answer_non_string_input():
    assert extract_path_from_answer(123) == "123"
    assert extract_path_from_answer(None) == "None"
    assert (
        extract_path_from_answer(["list", "of", "items"]) == "['list', 'of', 'items']"
    )


# --- Tests for find_file_by_name ---
def test_find_file_by_name_found_in_base_dir(temp_dir_structure):
    base_path = temp_dir_structure
    found_path = find_file_by_name("file1.txt", str(base_path))
    assert Path(found_path) == base_path / "file1.txt"
    assert Path(found_path).exists()


def test_find_file_by_name_found_in_subdir(temp_dir_structure):
    base_path = temp_dir_structure
    found_path = find_file_by_name("nested_file.log", str(base_path))
    assert Path(found_path) == base_path / "subdir1" / "nested_file.log"
    assert Path(found_path).exists()


def test_find_file_by_name_not_found(temp_dir_structure):
    base_path = temp_dir_structure
    result = find_file_by_name("non_existent_file.xyz", str(base_path))
    assert result == "non_existent_file.xyz"
    assert not Path(result).exists()


def test_find_file_by_name_multiple_matches_returns_most_recent(temp_dir_structure):
    base_path = temp_dir_structure
    # The fixture ensures base_path/data.json is newer than subdir2/data.json
    found_path = find_file_by_name("data.json", str(base_path))
    assert Path(found_path) == base_path / "data.json"
    assert Path(found_path).exists()


def test_find_file_by_name_no_base_dir_provided(temp_dir_structure):
    # This test relies on CORRAL_WORK_DIR being set by the fixture
    base_path = temp_dir_structure
    found_path = find_file_by_name("report.pdf")  # No base_dir argument
    assert Path(found_path) == base_path / "report.pdf"
    assert Path(found_path).exists()


def test_find_file_by_name_invalid_base_dir():
    # Ensure CORRAL_WORK_DIR is not set for this specific test
    original_corral_work_dir = os.environ.pop("CORRAL_WORK_DIR", None)
    try:
        result = find_file_by_name("some_file.txt", "/non/existent/path")
        assert result == "some_file.txt"
    finally:
        if original_corral_work_dir is not None:
            os.environ["CORRAL_WORK_DIR"] = original_corral_work_dir


# --- Tests for smart_resolve_path ---
def test_smart_resolve_path_direct_existing_path(temp_dir_structure):
    base_path = temp_dir_structure
    existing_path = base_path / "file1.txt"
    resolved_path = smart_resolve_path(str(existing_path))
    assert Path(resolved_path) == existing_path
    assert Path(resolved_path).exists()


def test_smart_resolve_path_path_does_not_exist_but_file_found(temp_dir_structure):
    base_path = temp_dir_structure
    # Simulate an extracted path that doesn't exist, but the file name does
    non_existent_path_str = str(base_path / "non_existent_dir" / "nested_file.log")
    resolved_path = smart_resolve_path(non_existent_path_str)
    # It should find the actual nested_file.log
    assert Path(resolved_path) == base_path / "subdir1" / "nested_file.log"
    assert Path(resolved_path).exists()


def test_smart_resolve_path_path_does_not_exist_and_file_not_found(temp_dir_structure):
    _base_path = temp_dir_structure
    non_existent_file_path = "answer: `unknown_file.doc`"
    resolved_path = smart_resolve_path(non_existent_file_path)
    assert (
        resolved_path == "unknown_file.doc"
    )  # Should return the extracted name if not found
    assert not Path(resolved_path).exists()


def test_smart_resolve_path_with_markdown_extraction(temp_dir_structure):
    base_path = temp_dir_structure
    # This file exists at base_path/data.json
    input_answer = "The latest data is in `data.json`."
    resolved_path = smart_resolve_path(input_answer)
    assert Path(resolved_path) == base_path / "data.json"
    assert Path(resolved_path).exists()


def test_smart_resolve_path_with_quotes_extraction(temp_dir_structure):
    base_path = temp_dir_structure
    # This file exists at base_path/subdir1/config.ini
    input_answer = "Check the 'config.ini' file."
    resolved_path = smart_resolve_path(input_answer)
    assert Path(resolved_path) == base_path / "subdir1" / "config.ini"
    assert Path(resolved_path).exists()


def test_smart_resolve_path_absolute_path_does_not_exist_but_file_found(
    temp_dir_structure,
):
    base_path = temp_dir_structure
    # Simulate an absolute path that doesn't exist, but the file name does
    non_existent_abs_path = "/tmp/some_other_place/file1.txt"
    resolved_path = smart_resolve_path(non_existent_abs_path)
    assert Path(resolved_path) == base_path / "file1.txt"
    assert Path(resolved_path).exists()
