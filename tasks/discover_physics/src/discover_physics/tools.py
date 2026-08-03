import json

from discover_physics.worlds import build_world

from corral.backend.tool import Tool, tool


@tool(hidden_args=["world_config"])
def run_experiment(
    experiments: str,
    world_config,
) -> str:
    """[BRIEF] Run a batch of experiments against the hidden physics simulator and return the observed trajectories. [/BRIEF]

    [DETAILED] Executes one or more experiments against the world's hidden simulator (a
    FieldSampler or NBody integrator running the true, undisclosed physics law) and
    returns the resulting particle trajectories. Each experiment specifies initial
    conditions (particle parameters, positions, velocities) and the times at which to
    measure the resulting trajectory. The exact set of keys each experiment dict must
    contain (e.g. `p1`, `p2`, `pos2`, `velocity2`, `measurement_times` for two-particle
    worlds; `ring_radius`, `initial_tangential_velocity`, `measurement_times` for the
    circle world) is world-specific and is given in the task description's
    `experiment_format` example and `law_stub`. Observed positions may include Gaussian
    measurement noise depending on the task; velocities are reported without noise. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use repeatedly, across many calls, to probe the hidden physics from different
      initial conditions before committing to a hypothesis.
    - Vary one quantity at a time (e.g. separation distance, charge/mass parameters,
      initial velocity) to isolate how the force/field law depends on it.
    - Use before submitting your final answer; you cannot query the simulator again
      after submission.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses each experiment dict in `experiments` and runs it through the world's
      hidden simulator for the requested `measurement_times`.
    - Returns one result dict per experiment with the observed trajectory (positions
      and velocities at each measurement time).
    - The simulator, its governing law, and any hidden parameters are never revealed
      directly; they can only be inferred from the trajectories this tool returns.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Read the task description's experiment format and law stub to
           learn this world's experiment schema. [/PREREQUISITE]
        2. [CURRENT] Call this tool with a batch of experiments to gather trajectory
           data. [/CURRENT]
        3. [FOLLOW_UP] Analyse the returned trajectories, refine your hypothesis, and
           either run more experiments or submit your discovered law. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - `run_experiment("[{\\"p1\\": 1.0, \\"p2\\": 1.0, \\"pos2\\": [3.0, 0.0], \\"velocity2\\": [0.0, 0.0], \\"measurement_times\\": [0.5, 1.0, 2.0]}]")`
    - `run_experiment("[{\\"ring_radius\\": 5.0, \\"initial_tangential_velocity\\": 0.3, \\"measurement_times\\": [2.0, 4.0]}]")`
    [/SYNTACTICAL]

    Args:
        experiments : [ARGS_BRIEF] JSON string encoding a list of experiment specifications to run. [/ARGS_BRIEF]
                      [ARGS_DETAILED] A JSON-encoded list of dicts; each element's required keys
                      depend on the current world (see the task's `experiment_format`), and every
                      experiment must include a `measurement_times` list of floats. [/ARGS_DETAILED]
                      [ARGS_SYNTACTICAL] Format: JSON string (with escaped quotes) - `"[{...}, {...}, ...]"` [/ARGS_SYNTACTICAL]
                      [ARGS_EXAMPLES] `"[{\\"p1\\": 1.0, \\"p2\\": 1.0, \\"pos2\\": [3.0, 0.0], \\"velocity2\\": [0.0, 0.0], \\"measurement_times\\": [0.5, 1.0]}]"` [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] JSON string with one result (observed trajectory) per experiment. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A JSON-encoded list, in the same order as `experiments`, of
             the observed trajectory data for each experiment (measurement times and the
             corresponding particle positions/velocities). [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] `[{"measurement_times": [0.5, 1.0], "pos1": [...], "pos2": [...], "velocity1": [...], "velocity2": [...]}]` [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERRORS]
            [ERROR_WHEN] When an experiment dict is missing a key required by this world's
            simulator. [/ERROR_WHEN]
            [ERROR_DETAILS] Each executor requires a specific set of keys; a missing one
            raises a KeyError/ValueError from the simulator. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Check the task's `experiment_format` example and ensure
            every required key is present. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - The simulator has finite numerical precision and (depending on the task) may add
      observation noise to reported positions.
    - Very long `measurement_times` or extreme initial conditions may be slow to
      simulate or numerically unstable.
    [/LIMITATIONS]
    """
    world = build_world(
        world_config["world"],
        engine=world_config.get("engine"),
        noise_std=world_config.get("noise_std", 0.0),
        noise_seed=world_config.get("noise_seed"),
    )
    results = world["executor"].run(json.loads(experiments))
    return json.dumps(results, indent=2)


def create_tools() -> dict[str, Tool]:
    """Build the tool pool for the discover_physics benchmark."""
    return {
        "run_experiment": run_experiment,
    }
