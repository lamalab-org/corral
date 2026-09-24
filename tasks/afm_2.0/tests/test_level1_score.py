"""Level 1 scoring tests using synthetic images and mocked instrument access."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("afm_level1_score", ROOT / "src/score.py")
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)
TASKS = [
    json.loads(p.read_text())[0]
    for p in sorted((ROOT / "environments/level_1").glob("*.json"))
]


@pytest.fixture
def acquisition(tmp_path, monkeypatch):
    path = tmp_path / "image.nid"
    path.write_bytes(b"mock NID; decoding supplied by the reader fixture")
    state = SimpleNamespace(path=path)
    state.params = {
        "lines_per_frame": 2,
        "points_per_line": 2,
        "mode": 2,
        "setpoint_v": 0.1,
    }
    state.image = {
        "Forward": {
            "Z-Axis": np.array([[9.0, 9.0], [11.0, 11.0]]),
            "Friction force": np.array([[2.0, 4.0], [2.0, 4.0]]),
        },
        "Backward": {"Friction force": np.zeros((2, 2))},
    }
    state.reader = Mock(
        side_effect=lambda _: SimpleNamespace(data={"Image": state.image})
    )

    def current_params():
        return state.params.copy()

    state.live = Mock(side_effect=current_params)
    monkeypatch.setattr(score, "read", state.reader)
    monkeypatch.setattr(score, "get_params", state.live)
    return state


def submission(path, **metrics):
    return json.dumps(
        {"path_1": str(path), **{f"{k}_1": str(v) for k, v in metrics.items()}}
    )


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task["id"])
def test_all_level1_task_factories(task, acquisition):
    params = task["scoring_params"]["final_params"]
    acquisition.params = params.copy()
    shape = (params["lines_per_frame"], params["points_per_line"])
    height = np.tile([9.0, 11.0], (shape[0], shape[1] // 2))
    acquisition.image = {
        "Forward": {"Z-Axis": height, "Friction force": np.full(shape, 4.0)},
        "Backward": {"Friction force": np.zeros(shape)},
    }
    fn = getattr(score, task["scoring_function"])(**task["scoring_params"])
    if task["scoring_function"] == "score_single_topography":
        result = str(acquisition.path)
    else:
        template, _ = json.JSONDecoder().raw_decode(task["submission_format"])
        report = {
            key: (
                str(acquisition.path)
                if key == "path_1"
                else "2"
                if "friction" in key
                else "1"
            )
            for key in template
        }
        result = json.dumps(report)
    assert fn(result) == 1.0
    acquisition.params["igain"] += 100
    assert fn(result) == 0.0


@pytest.mark.parametrize(
    "bad", [None, True, "NaN", "Infinity", "-Infinity", "not a number", "1 nm", 2.0]
)
def test_invalid_and_incorrect_measurements_fail(acquisition, bad):
    fn = score.score_single_rms_roughness(0.05, acquisition.params)
    assert (
        fn(json.dumps({"path_1": str(acquisition.path), "rms_roughness_1": bad})) == 0.0
    )


def test_required_fields_and_one_image_only(acquisition):
    fn = score.score_single_rms_roughness(0.05, acquisition.params)
    for report in (
        {"rms_roughness_1": "1"},
        {"path_1": str(acquisition.path)},
        [],
        {
            "path_1": str(acquisition.path),
            "rms_roughness_1": "1",
            "path_2": str(acquisition.path),
        },
    ):
        assert fn(json.dumps(report)) == 0.0
    assert fn("not JSON") == 0.0


def test_raw_units_and_relative_tolerance(acquisition):
    acquisition.image["Forward"]["Z-Axis"] = np.array([[-1e-9, 1e-9], [-1e-9, 1e-9]])
    fn = score.score_single_rms_roughness(0.05, acquisition.params)
    assert fn(submission(acquisition.path, rms_roughness=1e-9)) == 1.0
    assert fn(submission(acquisition.path, rms_roughness=1.04e-9)) == 1.0
    assert fn(submission(acquisition.path, rms_roughness=1.06e-9)) == 0.0
    assert fn(submission(acquisition.path, rms_roughness=1)) == 0.0
    acquisition.image["Forward"]["Z-Axis"] = np.ones((2, 2))
    assert fn(submission(acquisition.path, rms_roughness=0)) == 1.0
    assert fn(submission(acquisition.path, rms_roughness=1e-20)) == 0.0


def test_friction_matches_analyzer_signed_mean(acquisition):
    # Half-difference is [-1, 2]; signed mean=.5, mean magnitude=1.5, RMS=sqrt(2.5).
    acquisition.image["Forward"]["Friction force"] = np.array(
        [[-2.0, 4.0], [-2.0, 4.0]]
    )
    assert (
        score.score_single_average_friction(0.05, acquisition.params)(
            submission(acquisition.path, average_friction=0.5)
        )
        == 1.0
    )
    rms = score.score_single_rms_friction(0.05, acquisition.params)
    assert rms(submission(acquisition.path, rms_friction=np.sqrt(2.5))) == 1.0
    assert rms(submission(acquisition.path, rms_friction=1.5)) == 0.0
    assert (
        score.score_single_roughness_and_friction(0.05, acquisition.params)(
            submission(
                acquisition.path,
                rms_roughness=1,
                mean_roughness=1,
                average_friction=0.5,
            )
        )
        == 1.0
    )


@pytest.mark.parametrize(
    "bad_image",
    [
        np.ones((1, 2)),
        np.array([[1, np.nan], [1, 1]]),
        np.full((2, 2), np.inf),
        np.array([]),
    ],
)
def test_invalid_image_arrays_fail(acquisition, bad_image):
    acquisition.image["Forward"]["Z-Axis"] = bad_image
    assert (
        score.score_single_topography(0.05, acquisition.params)(str(acquisition.path))
        == 0.0
    )


def test_missing_channels_and_read_failure(acquisition):
    fn = score.score_single_lateral_roughness(0.05, acquisition.params)
    result = submission(acquisition.path, rms_roughness=1, mean_roughness=1)
    del acquisition.image["Backward"]["Friction force"]
    assert fn(result) == 0.0
    acquisition.reader.side_effect = ValueError("Corrupt NID")
    assert (
        score.score_single_topography(0.05, acquisition.params)(str(acquisition.path))
        == 0.0
    )


def test_bad_paths(acquisition, tmp_path):
    fn = score.score_single_topography(0.05, acquisition.params)
    other = tmp_path / "image.txt"
    other.touch()
    for path in (
        str(tmp_path),
        str(other),
        "image.nid",
        str(tmp_path / "missing.nid"),
        "",
        None,
    ):
        assert fn(path) == 0.0


def test_parameter_validation(acquisition):
    assert score.check_params({"setpoint_v": 0.1}) == 1.0
    assert score.check_params({"setpoint_p": 70}) == 0.0
    assert score.check_params({"lines_per_frame": 2.01}) == 0.0
    acquisition.params["setpoint_v"] = float("nan")
    assert score.check_params({"setpoint_v": 0.1}) == 0.0


@pytest.mark.parametrize(
    ("mode", "unit", "key"),
    [(4, 2, "setpoint_p"), (3, 2, "setpoint_p"), (2, 0, "setpoint_v"), (2, 2, None)],
)
def test_live_setpoint_units_and_centers(monkeypatch, mode, unit, key):
    zcontrol = SimpleNamespace(
        PGain=100, IGain=50, DGain=0, SetPoint=0.1, SetPointForceUnitMode=unit
    )
    scan = SimpleNamespace(
        ImageHeight=5e-6,
        ImageWidth=5e-6,
        Scantime=0.075,
        Points=256,
        Lines=256,
        rotation=0,
        CenterPosX=3e-6,
        CenterPosY=0,
    )
    app = SimpleNamespace(
        Scan=scan,
        ZController=zcontrol,
        ScanHead=SimpleNamespace(CantileverByGUID="tip"),
        OperatingMode=SimpleNamespace(OperatingMode=mode),
    )
    monkeypatch.setattr(
        score, "nanosurf", SimpleNamespace(SPM=lambda: SimpleNamespace(application=app))
    )
    monkeypatch.setattr(score, "pythoncom", None)
    params = score.get_params()
    assert params["centre_x"] == pytest.approx(3000)
    assert ({"setpoint_p", "setpoint_v"} & params.keys()) == ({key} if key else set())
    if key:
        assert params[key] == 0.1


def test_environment_registry_covers_level1():
    tree = ast.parse((ROOT / "src/env.py").read_text())
    imports = [
        n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "score"
    ]
    for node in imports:
        for alias in node.names:
            assert hasattr(score, alias.name)
    registry = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "SCORING_FUNCTIONS" for t in n.targets
        )
    )
    names = {key.value for key in registry.keys}
    assert {task["scoring_function"] for task in TASKS} <= names


def test_mean_and_rms_roughness_are_distinct(acquisition):
    acquisition.image["Forward"]["Z-Axis"] = np.array([[0.0, 0.0], [0.0, 4.0]])
    fn = score.score_single_topography_roughness(0.05, acquisition.params)
    assert (
        fn(submission(acquisition.path, rms_roughness=np.sqrt(3), mean_roughness=1.5))
        == 1.0
    )
    assert (
        fn(submission(acquisition.path, rms_roughness=1.5, mean_roughness=np.sqrt(3)))
        == 0.0
    )


@pytest.mark.parametrize(
    ("key", "value", "mode"), [("setpoint_v", 0.1, 2), ("setpoint_p", 70, 4)]
)
def test_reset_applies_new_setpoint_keys(key, value, mode):
    tree = ast.parse((ROOT / "src/env.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AFMEnvironment"
    )
    reset = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "reset_params"
    )
    zcontrol = SimpleNamespace()

    class OperatingMode:
        @property
        def OperatingMode(self):
            return self.mode

        @OperatingMode.setter
        def OperatingMode(self, mode):
            self.mode = mode
            zcontrol.SetPoint = -1  # Switching mode restores a previous setpoint.

    app = SimpleNamespace(
        Scan=SimpleNamespace(),
        ZController=zcontrol,
        ScanHead=SimpleNamespace(),
        OperatingMode=OperatingMode(),
        SetGalleryHistoryDirectoryPath=lambda _: None,
        GetGalleryHistoryDirectoryPath="test",
    )
    ns = {
        "pythoncom": None,
        "nanosurf": SimpleNamespace(SPM=lambda: SimpleNamespace(application=app)),
        "gc": SimpleNamespace(collect=lambda: None),
        "event": lambda *args, **kwargs: None,
    }
    exec(compile(ast.Module(body=[reset], type_ignores=[]), "env.py", "exec"), ns)
    instance = SimpleNamespace(
        initial_params={"mode": mode, key: value},
        workspace_path="test",
        task_id="test",
        afm_dir="test",
    )
    ns["reset_params"](instance)
    assert zcontrol.SetPoint == value
    if key == "setpoint_v":
        assert zcontrol.SetPointForceUnitMode == 0


def test_combined_scorer_rejects_absolute_friction_mean(acquisition):
    acquisition.image["Forward"]["Friction force"] = np.full((2, 2), -4.0)
    fn = score.score_single_roughness_and_friction(0.05, acquisition.params)
    assert (
        fn(
            submission(
                acquisition.path, rms_roughness=1, mean_roughness=1, average_friction=-2
            )
        )
        == 1.0
    )
    assert (
        fn(
            submission(
                acquisition.path, rms_roughness=1, mean_roughness=1, average_friction=2
            )
        )
        == 0.0
    )


def test_friction_uses_named_channel_even_when_other_channels_exist(acquisition):
    for direction in ("Forward", "Backward"):
        acquisition.image[direction]["Lateral"] = np.full((2, 2), 100.0)
    fn = score.score_single_average_friction(0.05, acquisition.params)
    assert fn(submission(acquisition.path, average_friction=1.5)) == 1.0
    del acquisition.image["Forward"]["Friction force"]
    assert fn(submission(acquisition.path, average_friction=1.5)) == 0.0
