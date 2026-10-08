"""Full workstation paths must grade files from the saved workspace only."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "afm_submission", Path(__file__).resolve().parents[1] / "src/submission.py"
)
submission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submission)
resolve = submission.resolve_submission


@pytest.mark.parametrize(
    "original",
    [
        r"C:\Users\Admin\run\scans\scan.nid",
        "C:/Users/Admin/run/scans/scan.nid",
        r"\\lab-server\AFM data\run\scans\SCAN.NID",
        "/old/workstation/run/scans/scan.nid",
        "/workspace/scans/scan.nid",
    ],
)
def test_absolute_paths_resolve_to_snapshot(original, tmp_path):
    saved = tmp_path / "scans/scan.nid"
    saved.parent.mkdir()
    saved.write_bytes(b"saved scan")
    # Prefer the complete relative suffix over a same-name root file.
    (tmp_path / "scan.nid").write_bytes(b"other scan")
    assert resolve(original, tmp_path) == str(saved)
    assert resolve(str(saved), tmp_path) == str(saved)


def test_numbered_paths_preserve_measurements_and_unknown_fields(tmp_path):
    report = {
        "rms_roughness_1": "1.25",
        "rms_roughness_percent_change_2": None,
        "extra_field": "leave this for the scorer to reject",
    }
    expected = report.copy()
    for i in range(1, 4):
        saved = tmp_path / f"scan{i}.nid"
        saved.write_bytes(f"scan {i}".encode())
        report[f"path_{i}"] = rf"D:\lab\run\scan{i}.nid"
        expected[f"path_{i}"] = str(saved)
    assert json.loads(resolve(json.dumps(report), tmp_path)) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "scan.nid",
        "C:scan.nid",
        "/old/../scan.nid",
        "/old/scan.txt",
        "/workspace/missing/scan.nid",
        "/old/missing.nid",
        "{invalid}",
        '{"path_1": 42}',
        '{"path_1": "/old/scan.nid", "path_1": "/old/scan.nid"}',
        '{"path_1": "/old/scan.nid", "rms_roughness_1": 1, "rms_roughness_1": 2}',
    ],
)
def test_invalid_paths_and_duplicate_keys_fail_closed(answer, tmp_path):
    (tmp_path / "scan.nid").write_bytes(b"saved scan")
    assert resolve(answer, tmp_path) == ""


def test_missing_snapshot_never_reads_existing_live_scan(tmp_path):
    original = tmp_path / "original/scan.nid"
    original.parent.mkdir()
    original.write_bytes(b"live scan")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    assert resolve(str(original), snapshot) == ""


def test_explicit_nested_path_cannot_select_another_directory(tmp_path):
    saved = tmp_path / "scanB/scan.nid"
    saved.parent.mkdir()
    saved.write_bytes(b"other scan")
    assert resolve("/old/scanA/scan.nid", tmp_path) == ""


def test_windows_case_ambiguity_is_rejected(tmp_path):
    first, second = tmp_path / "Scan.nid", tmp_path / "scan.nid"
    first.write_bytes(b"first scan")
    if second.exists():
        pytest.skip("A case-sensitive filesystem is required")
    second.write_bytes(b"second scan")
    assert resolve(r"C:\old\scan.nid", tmp_path) == ""


@pytest.mark.parametrize("directory_link", [False, True])
def test_symbolic_links_cannot_escape_snapshot(tmp_path, directory_link):
    original = tmp_path / "original/scan.nid"
    original.parent.mkdir()
    original.write_bytes(b"live scan")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    link = snapshot / ("linked" if directory_link else "scan.nid")
    try:
        link.symlink_to(
            original.parent if directory_link else original,
            target_is_directory=directory_link,
        )
    except OSError:
        pytest.skip("Symbolic links are unavailable")
    path = "/workspace/linked/scan.nid" if directory_link else "/workspace/scan.nid"
    assert resolve(path, snapshot) == ""
