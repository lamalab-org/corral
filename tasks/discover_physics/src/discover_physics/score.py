"""Scoring for discover_physics: trajectory-accuracy (MSE) only.

The upstream DiscoverPhysics benchmark scores two things: trajectory MSE
(fit accuracy of the submitted law) and an LLM-judged score of the agent's
prose explanation against a per-world rubric. This environment implements
MSE-only scoring for now — the explanation-judge half needs a live LLM call
made *during scoring* (not just during the agent's run), which no other
Corral task does yet, so it's left out of scope here rather than bolted on.

The submission (`answer`) is expected to be a self-contained Python source
string defining `discovered_law(...)` (and matching the per-world signature
documented in each task's `law_stub`), with any constants the agent inferred
from its own `run_experiment` calls hardcoded in. Unlike the upstream
harness's interactive `<run_mse_fit>` step, no free-parameter fitting against
the agent's own collected data happens at scoring time here.
"""

from __future__ import annotations

from discover_physics.worlds import build_evaluator, build_world


def check_physics_law(
    answer: str,
    *,
    world: str,
    engine: str | None = None,
    noise_std: float = 0.0,
    noise_seed: int | None = None,
) -> float:
    """Score a submitted `discovered_law` source against held-out test cases.

    Reconstructs the same hidden simulator the agent experimented against
    (with noise disabled for scoring, matching upstream `Evaluator.evaluate`),
    runs the world's Evaluator, and maps the resulting mean particle MSE
    (`mean_pos_error` — mean squared L2 position error over particle/time/
    case) onto a 0-1 score. Submissions that clear the upstream pass
    threshold (`result["passed"]`) are floored at 0.9 so near-miss
    submissions don't score far below solved ones just because 1/(1+mse) is
    a steep curve near zero.

    Raises whatever `evaluate()` raises (bad syntax, missing
    `discovered_law`, a law call that times out, ...); the caller
    (`Environment.score`) already catches and records that as a 0.0.
    """
    world_cfg = build_world(
        world, engine=engine, noise_std=noise_std, noise_seed=noise_seed
    )
    evaluator = build_evaluator(world, world_cfg["executor"])

    result = evaluator.evaluate(answer, verbose=False)
    mean_pos_error = result.get("mean_pos_error", float("inf"))
    if mean_pos_error is None or mean_pos_error != mean_pos_error:  # NaN
        return 0.0

    score = 1.0 / (1.0 + mean_pos_error)
    if result.get("passed"):
        score = max(score, 0.9)
    return float(score)
