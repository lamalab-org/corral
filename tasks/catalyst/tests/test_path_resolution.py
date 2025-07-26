import os
from pathlib import Path

os.environ["CORRAL_WORK_DIR"] = str(Path(__file__).parent / "test_files")
import pytest
from catalyst.utils import (
    extract_path_from_answer,
    find_file_by_name,
    smart_resolve_path,
)

TEMP_DIR = Path(os.environ["CORRAL_WORK_DIR"])


@pytest.fixture(scope="module", autouse=True)
def setup_corral_work_dir():
    test_files_dir = Path(__file__).parent / "test_files"
    original = os.environ.get("CORRAL_WORK_DIR")
    os.environ["CORRAL_WORK_DIR"] = str(test_files_dir)
    yield test_files_dir
    if original:
        os.environ["CORRAL_WORK_DIR"] = original
    else:
        os.environ.pop("CORRAL_WORK_DIR", None)


# --- Tests for extract_path_from_answer ---
def test_extract_path_from_answer_basic_filename():
    assert extract_path_from_answer("filename.txt") == "filename.txt"

    def test_extract_path_from_answer_with_prefix():
        assert (
            extract_path_from_answer("final answer is path/to/file.json")
            == "path/to/file.json"
        )

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
def test_find_file_by_name_found_in_base_dir():
    base_path = TEMP_DIR
    found_path = find_file_by_name("slabs.json", str(base_path))
    assert Path(found_path) == base_path / "slabs.json"
    assert Path(found_path).exists()


def test_find_file_by_name_not_found():
    base_path = TEMP_DIR
    result = find_file_by_name("non_existent_file.xyz", str(base_path))
    assert result == "non_existent_file.xyz"
    assert not Path(result).exists()


def test_find_file_by_name_multiple_matches_returns_most_recent():
    base_path = TEMP_DIR

    newest_file = Path(base_path / "subdir3" / "data.json")
    newest_file.write_text('{"key": "newest"}')
    found_path = find_file_by_name("data.json", str(base_path))
    assert Path(found_path) == base_path / "subdir3" / "data.json"
    assert Path(found_path).exists()


@pytest.mark.usefixtures("setup_corral_work_dir")
def test_find_file_by_name_no_base_dir_provided():
    # This test relies on CORRAL_WORK_DIR being set by the fixture
    base_path = TEMP_DIR
    found_path = find_file_by_name("bulk_structure.cif")  # No base_dir argument
    assert Path(found_path) == base_path / "bulk_structure.cif"
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
def test_smart_resolve_path_direct_existing_path():
    base_path = TEMP_DIR
    existing_path = base_path / "file1.txt"
    resolved_path = smart_resolve_path(str(existing_path))
    assert Path(resolved_path) == existing_path
    assert Path(resolved_path).exists()


def test_smart_resolve_path_path_does_not_exist_and_file_not_found():
    _base_path = TEMP_DIR
    non_existent_file_path = "answer: `unknown_file.doc`"
    resolved_path = smart_resolve_path(non_existent_file_path)
    assert (
        resolved_path == "unknown_file.doc"
    )  # Should return the extracted name if not found
    assert not Path(resolved_path).exists()


def test_smart_resolve_path_with_markdown_extraction():
    base_path = TEMP_DIR
    # This file exists at base_path/data.json
    input_answer = "The latest data is in `data.json`."
    resolved_path = smart_resolve_path(input_answer)
    assert Path(resolved_path) == base_path / "subdir3" / "data.json"
    assert Path(resolved_path).exists()


def test_smart_resolve_path_with_quotes_extraction():
    base_path = TEMP_DIR
    # This file exists at base_path/subdir1/config.ini
    input_answer = "Check the 'config.ini' file."
    resolved_path = smart_resolve_path(input_answer)
    assert Path(resolved_path) == base_path / "subdir1" / "config.ini"
    assert Path(resolved_path).exists()


def test_smart_resolve_path_absolute_path_does_not_exist_but_file_found():
    base_path = TEMP_DIR
    # Simulate an absolute path that doesn't exist, but the file name does
    non_existent_abs_path = "/tmp/some_other_place/file1.txt"
    resolved_path = smart_resolve_path(non_existent_abs_path)
    assert Path(resolved_path) == base_path / "file1.txt"
    assert Path(resolved_path).exists()
