from __future__ import annotations

import json
from dataclasses import replace

import pytest
from wetlab.engine import ChemicalSystemSpec, WetlabEngine, WetlabState
from wetlab.env import QualitativeAnalysisEnvironment, QualitativeAnalysisTask
from wetlab.tools import create_tools

from corral.core import Action, State, execute_action, propose_action
from corral.core.environment import Toolset
from corral.core.task import InputRef


def _environment() -> QualitativeAnalysisEnvironment:
    task = QualitativeAnalysisTask(
        name="wetlab-state-test",
        description="Run a deterministic wetlab experiment.",
        tools=["mix_two_solutions", "measure_pH"],
        scoring_fn=lambda _answer: 1.0,
        submission_format="number",
        task_sys="K S(+6)",
        sample_list=[
            {
                "label": "sample",
                "vol": 20,
                "composition": {"K+": 0.1, "HSO4-": 0.1},
            }
        ],
        reagent_set="Basic",
        additional_reagents=[],
    )
    return QualitativeAnalysisEnvironment(
        "wetlab-state-test",
        task,
        toolset=Toolset(
            pool=create_tools(),
            workspace_factory=None,
        ),
        group_tasks={"wetlab-state-test": task},
    )


def _wetlab_payload(state: State) -> dict:
    return dict(state.environment["hidden_arguments"]["wetlab"])


def test_structured_inventory_round_trip_is_exact() -> None:
    spec = ChemicalSystemSpec(elements="Ag Cl N")
    engine = WetlabEngine(spec)
    silver_nitrate = engine.stock_solution(
        {"Ag+": 0.1, "NO3-": 0.1},
        description="AgNO3 0.1 M",
    )
    hydrochloric_acid = engine.stock_solution(
        {"H+": 0.1, "Cl-": 0.1},
        description="HCl 0.1 M",
    )
    test_solution = 5 * silver_nitrate + 5 * hydrochloric_acid
    test_solution.equilibrate()
    assert test_solution.has_precipitate
    original = engine.snapshot(
        {
            "AgNO3": silver_nitrate,
            "HCl": hydrochloric_acid,
            "test": test_solution,
        }
    )

    serialized = original.to_json()
    restored_state = WetlabState.from_dict(json.loads(serialized))
    restored_engine = WetlabEngine(restored_state.chemical_system)
    restored_inventory = restored_engine.restore(restored_state)

    assert restored_engine.snapshot(restored_inventory).to_json() == serialized


def test_incompatible_chemistry_checkpoint_fails_loudly() -> None:
    incompatible = replace(
        ChemicalSystemSpec(elements="K S(+6)"),
        reaktoro_version="incompatible-test-version",
    )

    with pytest.raises(RuntimeError, match="Reaktoro version does not match"):
        WetlabEngine(incompatible)


def test_action_replay_and_checkpoint_restore_are_deterministic() -> None:
    environment = _environment()
    initial, _ = environment.configure(environment.initial_state())
    action = Action(
        id="mix-1",
        name="mix_two_solutions",
        arguments={
            "test_label": "sample_koh",
            "sol1_label": "sample",
            "sol1_vol": 2,
            "sol2_label": "KOH",
            "sol2_vol": 1,
        },
        actor_id="agent_0",
    )

    first = execute_action(environment, propose_action(initial, action), action)
    replay = execute_action(environment, propose_action(initial, action), action)

    assert first.messages[-1]["content"] == replay.messages[-1]["content"]
    assert _wetlab_payload(first) == _wetlab_payload(replay)

    checkpoint = State.from_json(first.to_json())
    continuation = Action(
        id="measure-2",
        name="measure_pH",
        arguments={"label": "sample_koh"},
        actor_id="agent_0",
    )
    continued = execute_action(
        environment,
        propose_action(checkpoint, continuation),
        continuation,
    )

    assert continued.messages[-1]["metadata"]["status"] == "success"
    assert _wetlab_payload(continued) == _wetlab_payload(checkpoint)


def test_chained_task_receives_inventory_through_dependency_output() -> None:
    upstream_environment = _environment()
    upstream, _ = upstream_environment.configure(upstream_environment.initial_state())
    submit = Action(
        id="submit-1",
        name="submit_answer",
        arguments={"answer": "K+"},
        actor_id="agent_0",
    )
    submitted = execute_action(
        upstream_environment,
        propose_action(upstream, submit),
        submit,
    )
    output = upstream_environment.get_task_output(submitted)
    assert output is not None

    upstream_task = upstream_environment.current_task
    downstream_task = QualitativeAnalysisTask(
        name="wetlab-state-test-next",
        description="Continue the experiment.",
        tools=["measure_pH"],
        scoring_fn=lambda _answer: 1.0,
        submission_format="number",
        input_map={"wetlab-state-test": InputRef("wetlab-state-test")},
        task_sys="K S(+6)",
        sample_list=[
            {
                "label": "sample",
                "vol": 20,
                "composition": {"K+": 0.1, "HSO4-": 0.1},
            }
        ],
        reagent_set="Basic",
        additional_reagents=[],
        initial=False,
    )
    downstream_environment = QualitativeAnalysisEnvironment(
        "wetlab-state-test-next",
        downstream_task,
        toolset=Toolset(pool=create_tools(), workspace_factory=None),
        group_tasks={
            "wetlab-state-test": upstream_task,
            "wetlab-state-test-next": downstream_task,
        },
    )
    downstream = downstream_environment.initial_state(
        dependency_outputs={"wetlab-state-test": output}
    )
    configured, _ = downstream_environment.configure(downstream)

    assert _wetlab_payload(configured) == output.metadata["wetlab"]
