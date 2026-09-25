"""Artifact-based scoring regressions for all twenty AFM tasks."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import NSFopen
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("afm_score", ROOT / "src/score.py")
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)
TASKS = [
    json.loads(p.read_text())[0]
    for p in sorted((ROOT / "environments").glob("*/tasks_json/*.json"))
]


def fake_nid(params, amplitude=1):
    """NSFopen-shaped data with independent, unit-bearing header strings."""
    shape = (params["lines_per_frame"], params["points_per_line"])
    height = np.tile([-amplitude, amplitude], (shape[0], shape[1] // 2)) * 1e-9
    info = {
        "P-Gain": str(params.get("pgain", 100)),
        "I-Gain": str(params.get("igain", 50)),
        "D-Gain": str(params.get("dgain", 10)),
        "Time/Line": f"{params.get('times_per_line', 0.075) * 1000} ms",
        "Points": str(shape[1]),
        "Lines": str(shape[0]),
        "Image size": f"{params.get('image_width', 5000) / 1000} Âµm",
        "X-Pos": f"{params.get('centre_x', 0)} nm",
        "Y-Pos": f"{params.get('centre_y', 0)} nm",
        "Rotation": f"{params.get('rotation', 0)} Â°",
        "Op. mode": "Phase Contrast" if params.get("mode", 4) == 4 else "Contact",
        "Cantilever type": "Multi75Al-G",
        "Setpoint": f"{params['setpoint']['value']} {params['setpoint']['unit']}",
    }
    headers = {"DataSet-Info": info}
    for direction in ("Forward", "Backward"):
        for name, unit in [("Z-Axis", "m"), ("Friction force", "V")]:
            headers[f"{direction}-{name}"] = {
                "Frame": f"Scan {direction.lower()}",
                "Dim2Name": name,
                "Dim2Unit": unit,
                "Points": str(shape[1]),
                "Lines": str(shape[0]),
                "Dim0Range": str(params.get("image_width", 5000) * 1e-9),
                "Dim0Unit": "m",
                "Dim1Range": str(params.get("image_height", 5000) * 1e-9),
                "Dim1Unit": "m",
            }
    return SimpleNamespace(
        param=pd.Series({("HeaderDump", k): v for k, v in headers.items()}),
        data={
            "Image": {
                "Forward": {
                    "Z-Axis": height,
                    "Friction force": np.full(shape, 2 * amplitude),
                },
                "Backward": {"Z-Axis": height, "Friction force": np.zeros(shape)},
            }
        },
    )


@pytest.fixture
def acquisition(tmp_path, monkeypatch):
    params = {
        "lines_per_frame": 2,
        "points_per_line": 2,
        "pgain": 100,
        "igain": 50,
        "dgain": 10,
        "times_per_line": 0.075,
        "image_width": 5000,
        "image_height": 5000,
        "mode": 4,
        "setpoint": {"value": 70, "unit": "%"},
    }
    path = tmp_path / "scan.nid"
    path.write_bytes(b"fixture")
    afm = fake_nid(params)
    monkeypatch.setattr(score, "read", lambda _: afm)
    return SimpleNamespace(params=params, afm=afm, path=path)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_all_tasks_accept_correct_artifacts_and_reject_bad_settings(
    task, tmp_path, monkeypatch
):
    config = task["scoring_params"]
    sequence = config["final_params"]
    if isinstance(sequence, dict):
        sequence = [sequence]
    files, report = {}, {}
    for i, params in enumerate(sequence, 1):
        path = tmp_path / f"image{i}.nid"
        path.write_bytes(f"fixture-{i}".encode())
        files[str(path)] = fake_nid(params, amplitude=i)
        report[f"path_{i}"] = str(path)
        for metric in config.get("metrics", []):
            report[f"{metric}_{i}"] = str(i)
            if "percent_change_reference" in config:
                report[f"{metric}_percent_change_{i}"] = 100 * (i - 2) / 2
    monkeypatch.setattr(score, "read", lambda p: files[p])
    fn = getattr(score, task["scoring_function"])(**config)
    result = (
        report["path_1"]
        if len(sequence) == 1 and not config.get("metrics")
        else json.dumps(report)
    )
    assert fn(result) == 1
    # Change every image individually: no earlier acquisition may escape checks.
    for afm in files.values():
        info = afm.param[("HeaderDump", "DataSet-Info")]
        original = info["I-Gain"]
        info["I-Gain"] = "9999"
        assert fn(result) == 0
        info["I-Gain"] = original
    if config.get("metrics"):
        key = f"{config['metrics'][0]}_1"
        bad = report.copy()
        bad.pop(key)
        assert fn(bad) == 0
        bad = report.copy()
        bad[key] = "9999"
        assert fn(bad) == 0
    if len(sequence) == 3:
        bad = report.copy()
        bad["path_2"] = bad["path_1"]
        assert fn(bad) == 0
        bad = report.copy()
        bad.pop("path_3")
        assert fn(bad) == 0
        if sequence[0] != sequence[2]:
            bad = report.copy()
            bad["path_1"], bad["path_3"] = bad["path_3"], bad["path_1"]
            assert fn(bad) == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("500 mV", {"value": 0.5, "unit": "V"}),
        ("70 %", {"value": 70, "unit": "%"}),
        ("5e-1 V", {"value": 0.5, "unit": "V"}),
    ],
)
def test_setpoint_units(text, expected):
    assert score.parse_setpoint(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("5 Âµm", 5e-6), ("-3 μm", -3e-6), ("5000 nm", 5e-6), ("2 mm", 0.002)],
)
def test_length_units(text, expected):
    assert score.to_meters(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text", ["NaN V", "Infinity V", "70 N", "70", "1.2.3 V", "5 volts"]
)
def test_invalid_units(text):
    with pytest.raises(ValueError):
        score.parse_setpoint(text)


@pytest.mark.parametrize("bad", [None, True, "NaN", "Infinity", "1 nm", "wrong", 2])
def test_invalid_measurements(acquisition, bad):
    fn = score.score_roughness(0.05, acquisition.params, metrics=["rms_roughness"])
    assert fn({"path_1": str(acquisition.path), "rms_roughness_1": bad}) == 0


def test_roughness_units_tolerance_and_required_fields(acquisition):
    fn = score.score_roughness(0.05, acquisition.params, metrics=["rms_roughness"])
    report = {"path_1": str(acquisition.path), "rms_roughness_1": 1}
    assert fn(report) == 1
    for v, expected in [(1.04, 1), (1.06, 0), (1e-9, 0)]:
        report["rms_roughness_1"] = v
        assert fn(report) == expected
    report["rms_roughness_1"] = 1
    report["extra"] = 42
    assert fn(report) == 0
    assert fn('{"path_1":"a","path_1":"b","rms_roughness_1":1}') == 0
    afm = acquisition.afm
    afm.data["Image"]["Forward"]["Z-Axis"] *= 1e9
    afm.param[("HeaderDump", "Forward-Z-Axis")]["Dim2Unit"] = "nm"
    report.pop("extra")
    assert fn(report) == 1


def test_signed_and_magnitude_friction(acquisition):
    acquisition.afm.data["Image"]["Forward"]["Friction force"] = np.array(
        [[-2, 4], [-2, 4]]
    )
    report = {"path_1": str(acquisition.path), "average_friction_1": 0.5}
    fn = score.score_friction(0.05, acquisition.params)
    assert fn(report) == 1
    report["average_friction_1"] = 1.5
    assert fn(report) == 0
    assert (
        score.score_friction(0.05, acquisition.params, friction_absolute=True)(report)
        == 1
    )
    report = {"path_1": str(acquisition.path), "rms_friction_1": np.sqrt(2.5)}
    assert (
        score.score_friction(0.05, acquisition.params, metrics=["rms_friction"])(report)
        == 1
    )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("Setpoint", "70 V"),
        ("Points", "2.1"),
        ("Op. mode", "Contact"),
        ("P-Gain", "NaN"),
        ("Time/Line", "75 s"),
    ],
)
def test_header_mismatches_fail(acquisition, field, bad):
    acquisition.afm.param[("HeaderDump", "DataSet-Info")][field] = bad
    assert score.score_topography(0.05, acquisition.params)(str(acquisition.path)) == 0


def test_missing_header_and_channel_units_fail(acquisition):
    header = acquisition.afm.param[("HeaderDump", "DataSet-Info")]
    del header["P-Gain"]
    fn = score.score_topography(0.05, acquisition.params)
    assert fn(str(acquisition.path)) == 0
    header["P-Gain"] = "100"
    acquisition.afm.param[("HeaderDump", "Forward-Z-Axis")]["Dim2Unit"] = "V"
    assert fn(str(acquisition.path)) == 0


@pytest.mark.parametrize(
    "bad",
    [np.zeros((1, 2)), np.full((2, 2), np.nan), np.full((2, 2), np.inf), np.array([])],
)
def test_bad_images(acquisition, bad):
    acquisition.afm.data["Image"]["Forward"]["Z-Axis"] = bad
    assert score.score_topography(0.05, acquisition.params)(str(acquisition.path)) == 0


def test_paths_and_read_errors(acquisition, monkeypatch, tmp_path):
    fn = score.score_topography(0.05, acquisition.params)
    for path in [
        "relative.nid",
        str(tmp_path),
        str(tmp_path / "missing.nid"),
        None,
        "",
    ]:
        assert fn(path) == 0

    def broken(_):
        raise ValueError("corrupt file")

    monkeypatch.setattr(score, "read", broken)
    assert fn(str(acquisition.path)) == 0


def test_percent_changes_use_measured_reference_and_null(
    tmp_path, monkeypatch, acquisition
):
    params = acquisition.params
    sequence = [copy.deepcopy(params) for _ in range(3)]
    afms, report = {}, {}
    for i in range(1, 4):
        p = tmp_path / f"{i}.nid"
        p.touch()
        afms[str(p)] = fake_nid(params, amplitude=0)
        report.update(
            {
                f"path_{i}": str(p),
                f"rms_roughness_{i}": 0,
                f"rms_roughness_percent_change_{i}": None,
            }
        )
    monkeypatch.setattr(score, "read", lambda p: afms[p])
    fn = score.score_roughness(
        0.05,
        sequence,
        metrics=["rms_roughness"],
        percent_change_reference=2,
    )
    assert fn(report) == 1
    report["rms_roughness_percent_change_1"] = 0
    assert fn(report) == 0


def test_registry_and_loader_cover_all_tasks(tmp_path):
    tree = ast.parse((ROOT / "src/env.py").read_text())
    wanted = {"SCORING_FUNCTIONS", "get_scoring_function", "load_tasks_from_json"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)
        )
    ]
    ns = {n: getattr(score, n) for n in {t["scoring_function"] for t in TASKS}}
    ns.update(
        Path=Path,
        json=json,
        Callable=object,
        TaskDefinition=SimpleNamespace,
        InputRef=lambda x: x,
        event=lambda *a, **kw: None,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "env.py", "exec"), ns)
    for level in (1, 2):
        tasks = ns["load_tasks_from_json"](
            ROOT / f"environments/level_{level}", str(tmp_path)
        )
        assert len(tasks) == 10
        assert all(callable(t.scoring_fn) for t in tasks.values())
    assert len(ns["SCORING_FUNCTIONS"]) == 4


def test_real_nid_reader():

    path = Path(NSFopen.__file__).parent / "example/Plotting_Data/MoS2.nid"
    if not path.is_file():
        pytest.skip("NSFopen distribution does not include its sample NID")
    params = {
        "pgain": 0,
        "igain": 2500,
        "dgain": 0,
        "times_per_line": 0.680,
        "points_per_line": 512,
        "lines_per_frame": 256,
        "centre_x": -1840,
        "centre_y": -427,
        "rotation": 0,
        "mode": 4,
        "setpoint": {"value": 50, "unit": "%"},
        "tip": "ACL-A",
    }
    assert score.score_topography(0.05, params)(str(path)) == 1
    params["setpoint"]["value"] = 70
    assert score.score_topography(0.05, params)(str(path)) == 0


@pytest.mark.parametrize(
    ("mode", "setpoint"),
    [(2, {"value": 0.1, "unit": "V"}), (4, {"value": 70, "unit": "%"})],
)
def test_initial_structured_setpoint_applied_after_mode(mode, setpoint):
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
        def OperatingMode(self, value):
            self.mode = value
            zcontrol.SetPoint = -1
            zcontrol.PGain = zcontrol.IGain = zcontrol.DGain = -1

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
        "event": lambda *a, **kw: None,
    }
    exec(compile(ast.Module(body=[reset], type_ignores=[]), "env.py", "exec"), ns)
    ns["reset_params"](
        SimpleNamespace(
            initial_params={
                "mode": mode,
                "setpoint": setpoint,
                "pgain": 80,
                "igain": 40,
                "dgain": 0 if mode == 2 else 5,
            },
            workspace_path="test",
            task_id="test",
            afm_dir="test",
        )
    )
    assert zcontrol.SetPoint == setpoint["value"]
    assert (zcontrol.PGain, zcontrol.IGain, zcontrol.DGain) == (
        80,
        40,
        0 if mode == 2 else 5,
    )
    if setpoint["unit"] == "V":
        assert zcontrol.SetPointForceUnitMode == 0


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_task_unit_contract(task):
    level, number = map(int, (task["id"].split("_")[3], task["id"].split("_")[-1]))
    c = task["scoring_params"]
    sequence = [c["final_params"]] if level == 1 else c["final_params"]
    assert len(sequence) == (1 if level == 1 else 3)
    tapping = number in (1, 3, 4, 5)
    for i, p in enumerate(sequence):
        width = (
            2000
            if number == 10
            else [10000, 5000, 2000][i]
            if (level, number) == (2, 5)
            else 5000
        )
        line_time = [0.1, 0.075, 0.05][i] if level == 2 and number in (6, 8) else 0.075
        setpoint = (
            {2: [0.1, 0.2, 0.5], 3: [80, 70, 60], 7: [0.1, 0.2, 0.3]}[number][i]
            if level == 2 and number in (2, 3, 7)
            else 70
            if tapping
            else 0.1
        )
        assert p["image_width"] == p["image_height"] == width
        assert p["times_per_line"] == line_time
        assert p["lines_per_frame"] == p["points_per_line"] == (512 if tapping else 256)
        assert p["setpoint"] == {"value": setpoint, "unit": "%" if tapping else "V"}
        assert p["mode"] == (4 if tapping else 2)
    initial = task["initial_input"]["params"]
    assert initial["image_width"] == initial["image_height"] == 1000
    assert initial["points_per_line"] == initial["lines_per_frame"] == 128
    assert initial["mode"] == (2 if tapping else 4)
    assert initial["dgain"] == (0 if tapping else 5)
    assert initial["setpoint"] == {
        "value": 0.05 if tapping else 90,
        "unit": "V" if tapping else "%",
    }
    assert initial["times_per_line"] == 0.125
    metrics = c.get("metrics", [])
    if any("roughness" in m for m in metrics):
        assert "nanometres" in task["description"]
        assert "Roughness values must be in nm" in task["submission_format"]
    if any("friction" in m for m in metrics):
        assert "volts" in task["description"]
        assert "Friction-signal values must be in V" in task["submission_format"]


@pytest.mark.parametrize(
    ("unit", "raw_height"), [("m", 2.4e-9), ("nm", 2.4), ("µm", 0.0024)]
)
def test_height_units_convert_once(acquisition, unit, raw_height):
    afm = acquisition.afm
    afm.data["Image"]["Forward"]["Z-Axis"] = np.tile([-raw_height, raw_height], (2, 1))
    afm.param[("HeaderDump", "Forward-Z-Axis")]["Dim2Unit"] = unit
    measured = score.measure_image(afm, ["mean_roughness", "rms_roughness"])
    assert measured == pytest.approx({"mean_roughness": 2.4, "rms_roughness": 2.4})


def test_friction_mixed_units(acquisition):
    afm = acquisition.afm
    afm.data["Image"]["Forward"]["Friction force"] = np.full((2, 2), 300)
    afm.param[("HeaderDump", "Forward-Friction force")]["Dim2Unit"] = "mV"
    afm.data["Image"]["Backward"]["Friction force"] = np.full((2, 2), 0.1)
    assert score.measure_image(
        afm, ["average_friction", "rms_friction"]
    ) == pytest.approx({"average_friction": 0.1, "rms_friction": 0.1})
    afm.param[("HeaderDump", "Forward-Friction force")]["Dim2Unit"] = "N"
    with pytest.raises(ValueError):
        score.measure_image(afm, ["average_friction"])


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("75ms", 0.075),
        ("75000µs", 0.075),
        ("75000us", 0.075),
        ("75000000ns", 0.075),
        (".075s", 0.075),
    ],
)
def test_time_units(value, seconds):
    assert score.to_seconds(value) == pytest.approx(seconds)


def test_image_analyzer_returns_task_units(acquisition):
    tree = ast.parse((ROOT / "src/tools.py").read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "Image_Analyzer"
    )
    fn.decorator_list = []
    ns = {
        "Any": object,
        "read": lambda _: acquisition.afm,
        "measure_image": score.measure_image,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "tools.py", "exec"), ns)
    output = ns["Image_Analyzer"](
        str(acquisition.path),
        calculate_friction=True,
        calculate_mean_roughness=True,
        calculate_rms_roughness=True,
    )
    assert output["mean_roughness"] == pytest.approx(1)
    assert output["rms_roughness"] == pytest.approx(1)
    assert output["average_friction"] == pytest.approx(1)
    assert output["metric_units"] == {
        "mean_roughness": "nm",
        "rms_roughness": "nm",
        "average_friction": "V",
    }
    assert output["image_data"] is acquisition.afm.data["Image"]["Forward"]["Z-Axis"]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_all_tasks_use_one_percent(task):
    assert task["scoring_params"]["tolerance"] == 0.01


@pytest.mark.parametrize(
    ("target", "answer", "passes"),
    [
        (100, 99, True),
        (100, 101, True),
        (100, 101.001, False),
        (100, 98.999, False),
        (-100, -99, True),
        (-100, -101, True),
        (-100, -101.001, False),
        (0, 0, True),
        (0, 1e-15, False),
        (0.05, 0.0505, True),
        (0.05, 0.0506, False),
        (1, 1.01, True),
        (1, 1.01001, False),
    ],
)
def test_one_percent_target_based_bounds(target, answer, passes):
    assert score._close(answer, target, 0.01) is passes
    assert score.check_params({"pgain": target}, {"pgain": answer}) is passes


def test_one_percent_settings_and_measurements_in_scorer(acquisition):
    info = acquisition.afm.param[("HeaderDump", "DataSet-Info")]
    fn = score.score_roughness(0.01, acquisition.params, metrics=["rms_roughness"])
    report = {"path_1": str(acquisition.path), "rms_roughness_1": 1.01}
    info["P-Gain"] = "101"
    assert fn(report) == 1
    info["P-Gain"] = "101.001"
    assert fn(report) == 0
    info["P-Gain"] = "100"
    report["rms_roughness_1"] = 1.01001
    assert fn(report) == 0
    report["rms_roughness_1"] = 1
    info["Time/Line"] = "75.8 ms"
    assert fn(report) == 0  # Old 0.001 s absolute allowance would accept this.
    info["Time/Line"] = "75 ms"
    info["Setpoint"] = "70.701 %"
    assert fn(report) == 0


def test_counts_and_units_remain_exact():
    assert not score.check_params({"points_per_line": 512}, {"points_per_line": 513})
    assert not score.check_params(
        {"setpoint": {"value": 70, "unit": "%"}},
        {"setpoint": {"value": 70, "unit": "V"}},
    )


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_reset_state_cannot_pass_scoring(task, tmp_path, monkeypatch):
    config = task["scoring_params"]
    sequence = config["final_params"]
    if isinstance(sequence, dict):
        sequence = [sequence]
    initial = task["initial_input"]["params"]
    for expected in sequence:
        assert initial["mode"] != expected["mode"]
        assert initial["setpoint"]["unit"] != expected["setpoint"]["unit"]
        assert not score.check_params(expected, initial, config["tolerance"])
        for key in (
            "pgain",
            "igain",
            "times_per_line",
            "image_width",
            "image_height",
            "setpoint",
        ):
            assert not score.check_params(
                {key: expected[key]}, {key: initial[key]}, config["tolerance"]
            )
    # Even with correct requested metrics, raw artifacts captured at reset
    # settings must fail rather than count as completion of the task.
    report = {}
    for i in range(1, len(sequence) + 1):
        path = tmp_path / f"reset_{i}.nid"
        path.touch()
        report[f"path_{i}"] = str(path)
        for metric in config.get("metrics", []):
            report[f"{metric}_{i}"] = 1
            if "percent_change_reference" in config:
                report[f"{metric}_percent_change_{i}"] = 0
    monkeypatch.setattr(score, "read", lambda _: fake_nid(initial))
    result = (
        report["path_1"] if len(sequence) == 1 and not config.get("metrics") else report
    )
    fn = getattr(score, task["scoring_function"])(**config)
    assert fn(result) == 0


@pytest.mark.parametrize(
    ("friction_values", "magnitude", "signed_mean"),
    [([-2.0, -2.0], 2.0, -2.0), ([-2.0, 1.0], 1.5, -0.5)],
)
def test_level1_task10_accepts_mean_friction_magnitude(
    tmp_path, monkeypatch, friction_values, magnitude, signed_mean
):
    task = next(t for t in TASKS if t["id"] == "afm_experiment_level_1_task_10")
    config = task["scoring_params"]
    assert config["friction_absolute"] is True
    afm = fake_nid(config["final_params"])
    shape = afm.data["Image"]["Forward"]["Z-Axis"].shape
    # The scorer takes half the trace/retrace difference; backward is zero.
    afm.data["Image"]["Forward"]["Friction force"] = 2 * np.tile(
        friction_values, (shape[0], shape[1] // 2)
    )
    path = tmp_path / "negative_friction.nid"
    path.touch()
    monkeypatch.setattr(score, "read", lambda _: afm)
    fn = getattr(score, task["scoring_function"])(**config)
    report = {
        "path_1": str(path),
        "rms_roughness_1": 1,
        "mean_roughness_1": 1,
        "average_friction_1": magnitude,
    }
    assert fn(report) == 1
    report["average_friction_1"] = signed_mean
    assert fn(report) == 0
    if abs(signed_mean) != magnitude:
        report["average_friction_1"] = abs(signed_mean)
        assert fn(report) == 0  # abs(mean(signal)) is not mean(abs(signal)).
