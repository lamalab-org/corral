"""Score Task 2 final quantities from saved LAMMPS temperature/density tables.

Level 1 measures the final hold means. Level 2 fits Tg from cooling samples and
checks answer/reference agreement. Neither level audits the simulation setup,
run completion, sampling cadence, pressure, or settling of the trajectory.
"""

from __future__ import annotations

import numpy as np

from corral.evaluation import SubmissionScore

from .common import Evidence, EvidenceError, Rubric

SOURCE_PATH = "/workspace/resources/md-structures-melt-liq4000/liq4000.dat"
SOURCE_SHA256 = "1674c66496b0bab9eda0ce79b439ad90f9ce2097c5d5450a86965442f5cdfa63"
COOLING_STEPS = 370_000
TOTAL_STEPS = 470_000
# Historical Level 2 Task 2 target, from environments/level_2/
# tasks_json/task_2.json at commit 75d7b3560b7ab922a63fe4fbff9c01d73f83dff3.
# The 2600 K cutoff is only a data-selection limit, not an acceptable target Tg.
REFERENCE_TG_K = 1576.88
REFERENCE_TG_ATOL_K = 0.1 * REFERENCE_TG_K
VERSION = "task2-final-quantities-v1"


def _require(condition, detail):
    if not condition:
        raise EvidenceError(detail)


def read_logs(e: Evidence, *, level: int) -> np.ndarray:
    """Read elapsed-fs/temperature/density samples for the scored interval.

    Native thermo tables suffice. Time is elapsed fs in the task's real units;
    Step supplies the clock when Time is absent, using the prescribed 1 fs step.
    Commands, completion messages, pressure and other columns are not scored.
    """
    aliases = {
        "time": ("Time", "time_fs", "elapsed_time_fs"),
        "step": ("Step", "step"),
        "temperature": ("Temp", "Temperature", "temperature_K"),
        "density": ("Density", "density", "density_g_cm3"),
    }
    names = {name for group in aliases.values() for name in group}

    def column(header, fields, group):
        present = [header.index(name) for name in aliases[group] if name in header]
        _require(len(present) == 1, f"Thermo output needs one {group} column")
        try:
            value = float(fields[present[0]])
        except ValueError as exc:
            raise EvidenceError(f"Invalid thermo {group}") from exc
        _require(np.isfinite(value), f"Nonfinite thermo {group}")
        return value

    samples = []
    for path in dict.fromkeys(e.artifacts("raw_logs")):
        header = None
        for raw in path.read_text().splitlines():
            fields = raw.split("#", 1)[0].split()
            if not fields or fields[0] == "WARNING:":
                continue
            if fields[0] in names and any(
                name in fields for name in (*aliases["time"], *aliases["step"])
            ):
                header = fields
                continue
            if header is None:
                continue
            try:
                float(fields[0])
            except ValueError:
                header = None
                continue
            _require(len(fields) == len(header), "Malformed numerical thermo row")

            clock = (
                "time" if any(name in header for name in aliases["time"]) else "step"
            )
            time = column(header, fields, clock)
            _require(time >= 0, "Thermo times must be nonnegative")
            if level == 1:
                selected = 390_000 <= time <= TOTAL_STEPS
            else:
                selected = time <= COOLING_STEPS
            if not selected:
                continue
            temperature = column(header, fields, "temperature")
            _require(temperature > 0, "Thermo temperature must be positive")
            if level == 2 and temperature > 2600:
                continue
            density = column(header, fields, "density")
            _require(density > 0, "Thermo density must be positive")
            samples.append((time, temperature, density))
    _require(bool(samples), "No numerical thermo samples in the scored interval")
    # Repeated files and shared run endpoints must not weight observations twice.
    # There is no continuity or complete-run requirement between thermo tables.
    return np.unique(np.asarray(samples), axis=0)


def hold_statistics(trace: np.ndarray) -> dict:
    """Calculate Level 1 final hold means, without settling or protocol gates."""
    temperature, density = trace[:, 1], trace[:, 2]
    mean = float(temperature.mean())
    _require(abs(mean - 300) <= 15, "Mean hold temperature misses the 300 K target")
    return {
        "hold_temperature_K": mean,
        "hold_density_g_cm3": float(density.mean()),
        "hold_sample_count": len(trace),
    }


def _fit_transition(x, y):
    """Choose two independent linear regimes and calculate their intersection."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    _require(
        len(x) >= 6 and np.ptp(x) > 0,
        "Insufficient cooling data to fit two linear regimes",
    )
    origin = x.mean()
    fits = []
    # Three samples per regime leave residual degrees of freedom. Select the
    # partition from the data, without fixed temperature windows or averaging.
    for split in range(3, len(x) - 2):
        if x[split - 1] == x[split]:
            continue
        lines = []
        for temperatures, densities in (
            (x[:split], y[:split]),
            (x[split:], y[split:]),
        ):
            centered = temperatures - temperatures.mean()
            sxx = float(centered @ centered)
            if sxx == 0:
                break
            slope = float(centered @ (densities - densities.mean()) / sxx)
            intercept = float(densities.mean() + slope * (origin - temperatures.mean()))
            residual = densities - (intercept + slope * (temperatures - origin))
            lines.append((intercept, slope, float(residual @ residual), sxx))
        if len(lines) == 2:
            fits.append((sum(line[2] for line in lines), split, lines))
    _require(bool(fits), "Insufficient distinct temperatures in the two regimes")
    sse, split, (low, high) = min(fits, key=lambda fit: fit[0])
    slope_change = high[1] - low[1]
    # The task still requires an independently resolved transition. Parallel
    # lines, or a slope difference explained by residual noise, cannot supply
    # one. Pool the residual variance so a tiny segment cannot claim spurious
    # precision merely because its few points happen to be nearly collinear.
    slope_se = float(np.sqrt(sse / (len(x) - 4) * (1 / low[3] + 1 / high[3])))
    numerical_floor = 32 * np.finfo(float).eps * np.max(np.abs(y)) / np.ptp(x)
    separated = abs(slope_change) > max(3 * slope_se, numerical_floor)
    estimate = float(origin + (low[0] - high[0]) / slope_change) if separated else None
    resolved = bool(
        estimate is not None and np.isfinite(estimate) and x[0] < estimate < x[-1]
    )
    return {
        "resolved": resolved,
        "estimate_K": estimate
        if estimate is not None and np.isfinite(estimate)
        else None,
        "split_temperature_K": float((x[split - 1] + x[split]) / 2),
        "low_slope_g_cm3_K": low[1],
        "high_slope_g_cm3_K": high[1],
        "slope_change_g_cm3_K": slope_change,
        "slope_change_se_g_cm3_K": slope_se,
        "residual_rms_g_cm3": float(np.sqrt(sse / len(x))),
    }


def cooling_transition(trace: np.ndarray) -> dict:
    """Estimate Tg from two fitted lines using only cooling data at T <= 2600 K."""
    cooling = trace[(trace[:, 0] <= COOLING_STEPS) & (trace[:, 1] <= 2600)]
    fit = _fit_transition(cooling[:, 1], cooling[:, 2])
    return {
        "transition": "resolved" if fit["resolved"] else "unresolved",
        "cooling_tg_K": fit["estimate_K"] if fit["resolved"] else None,
        "fit": fit,
    }


class Task2Scorer:
    """Binary final-quantity scorer for the hold means and cooling Tg."""

    task_number = 2
    verifier = None

    def __init__(self, *, level=2):
        if type(level) is not int or level not in (1, 2):
            raise ValueError("level must be 1 or 2")
        self.level = level

    def __call__(self, submission):
        return self.score_submission(submission)

    def score_submission(self, submission):
        return self.evaluate_submission(submission).score

    def evaluate_submission(self, submission):
        report = self.evaluate(submission)
        failures = [
            f"{c['name']}: {c['detail']}"
            for c in report["checks"]
            if c["status"] == "failed"
        ]
        return SubmissionScore(
            score=report["score"],
            feedback="; ".join(failures) or None,
            metadata={"workflow_evaluation": report},
        )

    def evaluate(self, submission):
        rubric = Rubric(2)
        computed = {}

        def check():
            evidence = Evidence(submission, task_number=2, level=self.level)
            trace = read_logs(evidence, level=self.level)
            results = evidence.results
            if self.level == 1:
                computed.update(hold_statistics(trace))
                for key, atol in {
                    "hold_temperature_K": 1.0,
                    "hold_density_g_cm3": 0.001,
                }.items():
                    if key not in results:
                        continue
                    actual = results[key]
                    _require(
                        type(actual) in (float, int) and np.isfinite(actual),
                        f"Report a finite {key}",
                    )
                    _require(
                        abs(actual - computed[key]) <= atol + 0.01 * abs(computed[key]),
                        f"{key}: reported {actual:g}, recomputed {computed[key]:g}",
                    )
            if self.level == 2:
                computed.update(cooling_transition(trace))
                tolerance = REFERENCE_TG_ATOL_K
                lower, upper = REFERENCE_TG_K - tolerance, REFERENCE_TG_K + tolerance
                computed["reference_tg"] = {
                    "cooling_tg_K": REFERENCE_TG_K,
                    "absolute_tolerance_K": REFERENCE_TG_ATOL_K,
                    "accepted_range_K": [lower, upper],
                }
                _require(
                    computed["transition"] == "resolved",
                    "Level 2 requires a resolved Tg; the cooling transition is "
                    "unresolved in the retained data",
                )
                actual = results.get("tg")
                _require(
                    type(actual) in (int, float) and np.isfinite(actual),
                    "Report a finite numerical tg in kelvin",
                )
                _require(
                    abs(actual - computed["cooling_tg_K"]) <= 100,
                    "Cooling Tg differs from the independently fitted "
                    f"{computed['cooling_tg_K']:g} K",
                )
                _require(
                    # Include the tolerance boundaries despite line-fit roundoff.
                    # Check both values so the 100 K agreement allowance cannot
                    # extend the reference range for either estimate.
                    all(
                        np.isclose(
                            value,
                            REFERENCE_TG_K,
                            rtol=0,
                            atol=REFERENCE_TG_ATOL_K + 1e-6,
                        )
                        for value in (actual, computed["cooling_tg_K"])
                    ),
                    f"Cooling Tg must match the historical reference {REFERENCE_TG_K:g} K "
                    f"within {REFERENCE_TG_ATOL_K:g} K "
                    f"([{lower:.2f}, {upper:.2f}] K); "
                    f"reported: {actual:g} K; "
                    f"independently fitted: {computed['cooling_tg_K']:g} K. "
                    "Both values must lie within the reference range.",
                )
            return True

        rubric.check("hold_quantities" if self.level == 1 else "cooling_tg", 100, check)
        return {
            **rubric.as_dict(),
            "level": self.level,
            "mode": "saved_logs",
            "scoring_policy": VERSION,
            "computed": computed,
        }
