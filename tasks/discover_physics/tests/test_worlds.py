from discover_physics.worlds import (
    ALL_WORLDS,
    WORLD_ENGINE,
    WORLD_EVALUATOR,
    default_engine,
)


class TestWorldRegistry:
    def test_eleven_worlds(self):
        assert len(ALL_WORLDS) == 11
        assert ALL_WORLDS == tuple(sorted(ALL_WORLDS))

    def test_every_world_has_an_engine(self):
        for world in ALL_WORLDS:
            assert WORLD_ENGINE[world] in ("field", "nbody")

    def test_engine_matches_known_upstream_constraints(self):
        # These worlds have no FieldSampler twin upstream (executor_class not
        # in scienceagent.worlds._FIELD_EXECUTOR_CLASSES) and must use nbody.
        nbody_only = {"ether", "hubble", "oscillator", "coulomb_easy", "extra_dimensions"}
        for world in nbody_only:
            assert WORLD_ENGINE[world] == "nbody"
        for world in set(ALL_WORLDS) - nbody_only:
            assert WORLD_ENGINE[world] == "field"

    def test_evaluator_mapping_is_a_subset_of_worlds(self):
        assert set(WORLD_EVALUATOR) <= set(ALL_WORLDS)

    def test_default_engine_unknown_world_raises(self):
        import pytest

        with pytest.raises(ValueError):
            default_engine("not_a_real_world")

    def test_module_importable_without_scienceagent(self):
        # worlds.py must not import scienceagent/physchool at module level so
        # the task-JSON generator script stays usable without jax installed.
        # Run in a subprocess so a prior test's `import scienceagent` in this
        # same interpreter can't mask a module-level import creeping back in.
        import subprocess
        import sys

        code = (
            "import sys\n"
            "import discover_physics.worlds\n"
            "assert 'scienceagent' not in sys.modules, "
            "'discover_physics.worlds must not import scienceagent at module level'\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
