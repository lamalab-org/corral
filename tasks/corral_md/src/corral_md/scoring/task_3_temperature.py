"""Level 2 Task 3: the historical final-window trajectory temperature score."""

from __future__ import annotations

import numpy as np

from corral.evaluation import SubmissionScore

from .common import Evidence, EvidenceError, Rubric, finite_array

VERSION = "level2-task3-temperature-v1"
TARGET_K = 300.0
WINDOW_FRAMES = 10
RELATIVE_TOLERANCE = 0.1


class Task3TemperatureScorer:
    """Check ASE kinetic temperatures without loading a model or training data."""

    task_number = 3
    level = 2

    def evaluate(self, submission):
        rubric, computed, evidence = Rubric(3), {}, None

        def calculate():
            nonlocal evidence
            evidence = Evidence(submission, task_number=3, level=2)
            try:
                frames = evidence.trajectory("md_trajectory")
            except (OSError, ValueError, KeyError, IndexError, EOFError) as exc:
                raise EvidenceError(f"Cannot read MD trajectory: {exc}") from exc
            temperatures = []
            for atoms in frames[-WINDOW_FRAMES:]:
                if not len(atoms) or "momenta" not in atoms.arrays:
                    raise EvidenceError(
                        "Temperature frames must contain atoms and saved momenta"
                    )
                masses = finite_array(atoms.get_masses(), shape=(len(atoms),))
                if np.any(masses <= 0):
                    raise EvidenceError("Atomic masses must be positive")
                finite_array(atoms.get_momenta(), shape=(len(atoms), 3))
                temperatures.append(atoms.get_temperature())
            temperatures = finite_array(temperatures, ndim=1)
            mean = float(temperatures.mean())
            if not np.isfinite(mean):
                raise EvidenceError("Mean trajectory temperature must be finite")
            allowance = RELATIVE_TOLERANCE * abs(TARGET_K)
            lower, upper = TARGET_K - allowance, TARGET_K + allowance
            computed.update(
                mean_temperature_K=mean,
                target_temperature_K=TARGET_K,
                allowed_range_K=[lower, upper],
                total_frames=len(frames),
                averaged_frames=len(temperatures),
                reference_basis="ase_kinetic_temperature",
            )
            # Preserve inclusive endpoints despite kinetic-energy roundoff.
            passed = lower <= mean <= upper or np.any(
                np.isclose(mean, [lower, upper], rtol=0, atol=1e-10)
            )
            return bool(passed), (
                f"Mean temperature over the last {len(temperatures)} frames: "
                f"{mean:.8g} K; allowed range: [{lower:g}, {upper:g}] K"
            )

        rubric.check("trajectory_temperature", 100, calculate)
        return {
            **rubric.as_dict(),
            "level": self.level,
            "mode": "trajectory_temperature",
            "scoring_policy": VERSION,
            "computed": computed,
            "evidence_sha256": evidence.fingerprint() if evidence else None,
        }

    def evaluate_submission(self, submission):
        report = self.evaluate(submission)
        feedback = (
            "; ".join(c["detail"] for c in report["checks"] if c["status"] == "failed")
            or None
        )
        return SubmissionScore(
            score=report["score"],
            feedback=feedback,
            metadata={"workflow_evaluation": report},
        )

    def score_submission(self, submission):
        return self.evaluate_submission(submission).score

    __call__ = score_submission
