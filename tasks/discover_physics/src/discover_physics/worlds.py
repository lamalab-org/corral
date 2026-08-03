"""Registry mapping DiscoverPhysics world names to their engine/evaluator.

Wraps ``scienceagent.worlds.get_world`` (mission text, law stub, executor
construction) and ``scienceagent.evaluator`` (per-world scoring). Both are
reused as-is from the upstream ``scienceagent`` package (see pyproject.toml)
rather than reimplemented.

Upstream defines 13 world names in ``configs/bench.yml`` (adding
``force_geography`` and ``running_coupling``), but only these 11 are actually
wired into ``scienceagent.worlds.WORLDS`` / the run_discovery.py CLI as of
this writing — the other two have no registered executor or evaluator, so
they are excluded here.

Each world requires a specific simulation engine: ``field`` (FFT/CIC
FieldSampler) for worlds whose physics is a static PDE with a portable
operator, and ``nbody`` (direct pairwise integrator) for worlds whose physics
has no FieldSampler equivalent (body forces, time-modulated coupling, image
charges). Mixing these up raises inside ``get_world``.
"""

from __future__ import annotations

# NOTE: scienceagent/physchool (and their jax dependency) are imported lazily
# inside build_world()/build_evaluator(), not at module level, so this module
# (and WORLD_ENGINE/ALL_WORLDS) stays importable by the task-JSON generator
# script without those heavy dependencies installed.

# world name -> required simulation engine
WORLD_ENGINE: dict[str, str] = {
    "gravity": "field",
    "yukawa": "field",
    "fractional": "field",
    "dark_matter": "field",
    "three_species": "field",
    "circle": "field",
    "ether": "nbody",
    "hubble": "nbody",
    "oscillator": "nbody",
    "coulomb_easy": "nbody",
    "extra_dimensions": "nbody",
}

# world name -> Evaluator subclass name (mirrors the elif chain in
# ScienceAgent/run_discovery.py; every world not listed explicitly there
# uses the default two-particle Evaluator). Resolved lazily in
# build_evaluator() to avoid importing scienceagent at module level.
WORLD_EVALUATOR: dict[str, str] = {
    "circle": "CircleEvaluator",
    "three_species": "ThreeSpeciesEvaluator",
    "dark_matter": "DarkMatterEvaluator",
    "ether": "EtherEvaluator",
    "hubble": "HubbleEvaluator",
}

ALL_WORLDS: tuple[str, ...] = tuple(sorted(WORLD_ENGINE))


def default_engine(world: str) -> str:
    if world not in WORLD_ENGINE:
        raise ValueError(f"Unknown world {world!r}. Available: {ALL_WORLDS}")
    return WORLD_ENGINE[world]


def build_world(
    world: str,
    *,
    engine: str | None = None,
    noise_std: float = 0.0,
    noise_seed: int | None = None,
) -> dict:
    """Return the ``get_world`` config dict for ``world`` (executor, mission,
    law_stub, experiment_format, ...) with noise/engine applied."""
    from scienceagent.worlds import get_world

    resolved_engine = engine or default_engine(world)
    return get_world(
        world,
        engine=resolved_engine,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )


def build_evaluator(world: str, executor):
    """Instantiate the correct Evaluator subclass for ``world`` against
    ``executor`` (the same executor config used for experimentation)."""
    import scienceagent.evaluator as evaluator_module

    evaluator_cls_name = WORLD_EVALUATOR.get(world, "Evaluator")
    evaluator_cls = getattr(evaluator_module, evaluator_cls_name)
    return evaluator_cls(executor)
