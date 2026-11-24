"""Grade a policy's answers with Inspect's scorers, in the controller."""

from __future__ import annotations

from inference_opt.grading.grader import Grader
from inference_opt.grading.spec import GradeSpec, GradeSummary

__all__ = ["GradeSpec", "GradeSummary", "Grader"]
