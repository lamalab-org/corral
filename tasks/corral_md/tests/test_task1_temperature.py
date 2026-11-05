"""Level 1 Task 1 grades only the final recorded temperature."""

from pathlib import Path

import pytest
from corral_md.scoring import WorkflowScorer, check_level1_workflow


def submission(tmp_path, text):
    path = tmp_path / "log.lammps"
    path.write_text(text)
    return {"artifacts": {"lammps_log": str(path)}}


@pytest.mark.parametrize(
    "text",
    [
        "Temp\n2500\n",
        "Step Temp\n1000000 2500\n",
        "Press Temp Step\nnan 2500 1000000\n",
        "Step Time Temp Press Density\n1000000 1000 2500 nan nan\n",
    ],
)
def test_one_temperature_sample_needs_no_trajectory_or_result(tmp_path, text):
    answer = submission(tmp_path, text)
    scorer = check_level1_workflow(1, verification_backend="offline")
    assert isinstance(scorer, WorkflowScorer)
    result = scorer.evaluate_submission(answer)
    report = result.metadata["workflow_evaluation"]
    assert result.score == 1, report
    assert report["status"] == "complete"
    assert report["mode"] == "final_temperature"
    assert report["scoring_policy"] == "level1-task1-temperature-v1"
    assert report["calculated"]["final_temperature_K"] == 2500
    assert "phase" not in report["calculated"]
    assert "diffusion_m2_s" not in report["calculated"]


def test_last_row_across_native_tables_defines_final_temperature(tmp_path):
    answer = submission(
        tmp_path,
        "LAMMPS\nStep Temp\n1000000 8000\n"
        "Loop time of 1 on 1 procs for 100 steps\n"
        "Step Time Temp\n0 0 2200\n10 1 2400\n"
        "Loop time of 1 on 1 procs for 10 steps\n",
    )
    report = check_level1_workflow(1).evaluate(answer)
    assert report["score"] == 1, report
    assert report["calculated"]["final_temperature_K"] == 2400


def test_setup_and_earlier_thermal_history_do_not_gate_final_temperature(tmp_path):
    answer = submission(
        tmp_path,
        "units real\natom_style atomic\npair_style wrong\n"
        "velocity all create 9000 0\nrun 1\n"
        "Step Time Temp Press Density\n"
        "1000000 nan nan nan nan\n"
        "0 -1 -999 -inf -1\n"
        "2 0.01 2500 nan nan\n",
    )
    report = check_level1_workflow(1).evaluate(answer)
    assert report["score"] == 1, report
    assert report["calculated"]["final_temperature_K"] == 2500


@pytest.mark.parametrize("last_row", ["2 nan", "2 inf", "2 -inf", "2 invalid", "2"])
def test_invalid_final_temperature_does_not_fall_back_to_earlier_row(
    tmp_path, last_row
):
    answer = submission(tmp_path, f"Step Temp\n1 2500\n{last_row}\n")
    report = check_level1_workflow(1).evaluate(answer)
    assert report["status"] == "complete", report
    assert report["score"] == 0, report


@pytest.mark.parametrize(
    "last_table",
    [
        "Step Temp\n",
        "Step Press\n2 0\n",
        "Press Density\n1 2\nLoop time of 1\n",
        "c_pressure[1] v_density\n1 2\nLoop time of 1\n",
    ],
)
def test_missing_final_table_temperature_does_not_reuse_earlier_table(
    tmp_path, last_table
):
    answer = submission(tmp_path, "Step Temp\n1 2500\nLoop time of 1\n" + last_table)
    report = check_level1_workflow(1).evaluate(answer)
    assert report["status"] == "complete", report
    assert report["score"] == 0, report


@pytest.mark.parametrize(
    "text", ["", "not thermo data\n", "Step Temp\n", "Step Press\n0 0\n"]
)
def test_absent_temperature_scores_zero(tmp_path, text):
    report = check_level1_workflow(1).evaluate(submission(tmp_path, text))
    assert report["status"] == "complete", report
    assert report["score"] == 0, report


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [(1999, 0), (2000, 0), (2000.01, 1), (2500, 1), (2999.99, 1), (3000, 0), (3001, 0)],
)
def test_existing_strict_endpoint_temperature_allowance(
    tmp_path, temperature, expected
):
    answer = submission(tmp_path, f"Temp\n{temperature}\n")
    report = check_level1_workflow(1).evaluate(answer)
    assert report["score"] == expected, report


@pytest.mark.parametrize("retained", [False, True])
def test_legacy_extra_trajectory_does_not_affect_temperature_score(tmp_path, retained):
    answer = submission(tmp_path, "Step Temp\n1 2500\n")
    trajectory = Path(tmp_path / "old-trajectory.dump")
    if retained:
        trajectory.write_text("not a valid trajectory")
    answer["artifacts"]["trajectory"] = str(trajectory)
    report = check_level1_workflow(1).evaluate(answer)
    assert report["score"] == 1, report
    assert report["calculated"]["final_temperature_K"] == 2500
