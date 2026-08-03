"""Agent-facing per-world content: mission text, discovered_law stub, and an
example run_experiment call. Values are copied verbatim from the `WORLDS`
registry / `get_world()` defaults in upstream `scienceagent.worlds` so the
task prompts match what the real benchmark shows an agent.

Deliberately excluded: upstream's `WORLDS[world]["description"]` field. It's
never shown to the agent by `run_discovery.py` (only used for the leaderboard
site) and it names the hidden operator (e.g. gravity's is "... mediated by a
2D Laplacian field") — including it here would leak the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from discover_physics.prompts import TOPOLOGY_INSTRUCTIONS

MISSION = (
    "You are an expert physics and AI research scientist tasked with discovering "
    "scientific laws in a simulated universe. Your goal is to propose experiments, "
    "analyse the data they return, and ultimately deduce the underlying scientific "
    "law. Note that the laws of physics in this universe may differ from those in "
    "our own. You can perform experiments using the `run_experiment` tool, but must "
    "follow the protocol strictly."
)

_DEFAULT_LAW_STUB = (
    "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
    "    # pos1: [x, y] -- always [0, 0], particle 1 is fixed at the origin\n"
    "    # pos2: [x, y] -- initial position of particle 2\n"
    "    # p1, p2: scalar properties of particle 1 and particle 2\n"
    "    # velocity2: [vx, vy] -- initial velocity of particle 2\n"
    "    # duration: float -- simulate from t=0 to t=duration\n"
    "    # return: (final_pos2, final_vel2)\n"
    "    return final_pos2, final_vel2\n"
)

_DEFAULT_EXPERIMENT_FORMAT = (
    '[{"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], '
    '"velocity2": [0.0, 0.0], "measurement_times": [0.5, 1.0, 2.0]}]'
)


@dataclass(frozen=True)
class WorldContent:
    topology: str  # key into TOPOLOGY_INSTRUCTIONS
    law_stub: str
    experiment_format: str


WORLD_CONTENT: dict[str, WorldContent] = {
    "gravity": WorldContent("two_particle", _DEFAULT_LAW_STUB, _DEFAULT_EXPERIMENT_FORMAT),
    "yukawa": WorldContent("two_particle", _DEFAULT_LAW_STUB, _DEFAULT_EXPERIMENT_FORMAT),
    "fractional": WorldContent("two_particle", _DEFAULT_LAW_STUB, _DEFAULT_EXPERIMENT_FORMAT),
    "oscillator": WorldContent("two_particle", _DEFAULT_LAW_STUB, _DEFAULT_EXPERIMENT_FORMAT),
    "extra_dimensions": WorldContent(
        "two_particle", _DEFAULT_LAW_STUB, _DEFAULT_EXPERIMENT_FORMAT
    ),
    "coulomb_easy": WorldContent(
        "two_particle",
        (
            "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
            "    # pos1: [x, y] position of fixed particle 1 (always [0, 0])\n"
            "    # pos2: [x, y] initial position of particle 2\n"
            "    # p1, p2: scalar charges\n"
            "    # velocity2: [vx, vy] initial velocity of particle 2\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: (final_pos2, final_vel2)\n"
            "    return final_pos2, final_vel2\n"
        ),
        (
            '[{"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], '
            '"velocity2": [0.0, 0.5], "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0]}]'
        ),
    ),
    "circle": WorldContent(
        "circle",
        (
            "def discovered_law(positions, velocities, duration):\n"
            "    # positions: list of 11 [x, y] coords relative to center\n"
            "    # velocities: list of 11 [vx, vy]\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 11 [x, y] final positions\n"
            "    return final_positions\n"
        ),
        (
            '[{"ring_radius": 5.0, "initial_tangential_velocity": 0.0, '
            '"measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}]'
        ),
    ),
    "dark_matter": WorldContent(
        "probes",
        (
            "def discovered_law(positions, velocities, duration):\n"
            "    # positions: list of 25 [x, y] coords relative to center\n"
            "    #   indices 0-19: visible background, 20-24: probes\n"
            "    # velocities: list of 25 [vx, vy]\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 25 [x, y] final positions\n"
            "    # NOTE: you are scored on the 5 PROBE trajectories (indices 20-24)\n"
            "    return final_positions\n"
        ),
        (
            '[{"probe_positions": [[5,0],[0,5],[-5,0],[0,-5],[7,7]], '
            '"probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]], '
            '"measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}]'
        ),
    ),
    "three_species": WorldContent(
        "probes",
        (
            "def discovered_law(positions, velocities, duration):\n"
            "    # positions: list of 35 [x, y] coords relative to center\n"
            "    # velocities: list of 35 [vx, vy]\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 35 [x, y] final positions\n"
            "    return final_positions\n"
        ),
        (
            '[{"probe_positions": [[5,0],[0,5],[-5,0],[0,-5],[7,7]], '
            '"probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]], '
            '"measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}]'
        ),
    ),
    "ether": WorldContent(
        "probes_with_masses",
        (
            "def discovered_law(positions, velocities, masses, duration):\n"
            "    # positions: list of 26 [x, y] coords relative to centre\n"
            "    # velocities: list of 26 [vx, vy]\n"
            "    # masses: list of 26 per-particle masses\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 26 [x, y] final positions\n"
            "    # NOTE: scoring focuses on the 5 PROBE trajectories (indices 21-25)\n"
            "    return final_positions\n"
        ),
        (
            '[{"probe_positions": [[8,0],[0,8],[-8,0],[0,-8],[10,10]], '
            '"probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]], '
            '"probe_masses": [1.0, 1.0, 2.0, 4.0, 1.0], '
            '"measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}]'
        ),
    ),
    "hubble": WorldContent(
        "probes_with_masses",
        (
            "def discovered_law(positions, velocities, masses, duration):\n"
            "    # positions: list of 26 [x, y] coords relative to centre\n"
            "    # velocities: list of 26 [vx, vy]\n"
            "    # masses: list of 26 per-particle masses\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 26 [x, y] final positions\n"
            "    # NOTE: scoring focuses on the 5 PROBE trajectories (indices 21-25)\n"
            "    return final_positions\n"
        ),
        (
            '[{"probe_positions": [[5,0],[10,0],[15,0],[18,0],[0,12]], '
            '"probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]], '
            '"probe_masses": [1.0, 1.0, 2.0, 4.0, 1.0], '
            '"measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}]'
        ),
    ),
}


def build_description(world: str) -> str:
    """Full agent-facing task description: mission + topology instructions."""
    content = WORLD_CONTENT[world]
    instructions = TOPOLOGY_INSTRUCTIONS[content.topology]
    return (
        f"{MISSION}\n\n"
        f"{instructions}\n\n"
        "### Submitting your answer\n\n"
        "When you are ready, submit your final answer as a single self-contained "
        "Python source string defining `discovered_law(...)` with the exact "
        "signature shown below. Hardcode any constants you inferred from your "
        "experiments directly in the code -- there is no further fitting step "
        "after submission, and you cannot run more experiments once you submit.\n\n"
        f"```python\n{content.law_stub}```\n\n"
        f"Example `run_experiment` call for this world:\n\n"
        f"`run_experiment({content.experiment_format})`"
    )
