from discover_physics.content import WORLD_CONTENT, WorldContent, build_description
from discover_physics.worlds import ALL_WORLDS

# Real excerpts from upstream's WORLDS[world]["description"] / "optimal_explanation"
# fields (ground truth, never shown to the agent by run_discovery.py either) --
# a regression guard against accidentally including them in our task prompts.
_FORBIDDEN_SNIPPETS = {
    "gravity": ["mediated by a 2D Laplacian field", "obeying the 2D Poisson", "logarithmic in 2D"],
    "yukawa": ["Screened (Yukawa) potential", "Helmholtz equation", "screening length"],
    "fractional": ["fractional Laplacian operator", "non-local operator"],
    "coulomb_easy": ["hidden opposite signs", "q_0 = +|p1|"],
    "extra_dimensions": ["Kaluza-Klein", "compactified on a circle"],
    "dark_matter": ["dark-matter particles with source coupling", "concealed"],
    "three_species": ["source coupling +1, particles 10", "repulsive"],
    "ether": ["ether' field that exerts a body-force", "northward"],
    "hubble": ["Hubble-flow body-force", "r_crit"],
    "circle": ["fractional Laplacian operator -(-∇²)", "α = 0.75"],
    "oscillator": ["G(t) = G₀", "sinusoid"],
}


class TestWorldContentCoverage:
    def test_every_world_has_content(self):
        assert set(WORLD_CONTENT) == set(ALL_WORLDS)

    def test_every_world_has_a_known_topology(self):
        from discover_physics.prompts import TOPOLOGY_INSTRUCTIONS

        for world, content in WORLD_CONTENT.items():
            assert content.topology in TOPOLOGY_INSTRUCTIONS


class TestNoGroundTruthLeakage:
    def test_world_content_has_no_ground_truth_fields(self):
        # Structural guard: WorldContent literally cannot carry a leaked
        # optimal_explanation/true_law/rubric field, whatever gets added later.
        assert set(WorldContent.__dataclass_fields__) == {
            "topology",
            "law_stub",
            "experiment_format",
        }

    def test_description_excludes_forbidden_snippets(self):
        assert set(_FORBIDDEN_SNIPPETS) == set(ALL_WORLDS)
        for world, snippets in _FORBIDDEN_SNIPPETS.items():
            description = build_description(world)
            for snippet in snippets:
                assert snippet not in description, (
                    f"{world}: description leaks ground-truth snippet {snippet!r}"
                )

    def test_description_contains_law_stub_and_experiment_example(self):
        for world, content in WORLD_CONTENT.items():
            description = build_description(world)
            assert "discovered_law" in description
            assert content.experiment_format in description
