import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import litellm
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from corral import CorralRouter, CorralRunner
from corral.agents import ReActAgent
from corral.agents.hooks import AgentHooks, HookPoint
from corral.agents.hooks.core import HookContext
from corral.agents.hooks.intervention import create_intervention_hook
from corral.report import CorralWandbLogger

successful_partial = [
    {
        "role": "assistant",
        "content": 'I\'ll help you calculate the surface energy of Aluminum for the [0,0,1] direction. Let me start by understanding the task and gathering the necessary information.\n\n<thought>\nThe task requires calculating the surface energy of Aluminum using:\n1. Materials Project ID: mp-134\n2. Direction: [0, 0, 1]\n3. EAM potential\n4. 5x5x5 supercell replication\n5. CG minimization with specific tolerances\n\nSurface energy formula: \u03b3 = (E_surface - E_bulk) / (2 * A)\nwhere:\n- E_surface: energy of the slab with surface\n- E_bulk: energy of the bulk structure\n- A: surface area\n- Factor of 2 accounts for two surfaces created\n\nI need to:\n1. Get the structure from Materials Project\n2. Find the appropriate EAM potential for Aluminum\n3. Create bulk configuration and minimize\n4. Create surface configuration (remove PBC in z-direction) and minimize\n5. Calculate surface energy\n\nLet me start by listing available potentials.\n</thought>\n\n<action>list_files</action>\n<action_input>{"path": "/potentials"}</action_input>',
    },
    {
        "role": "user",
        "content": "Observation: {'tool_name': 'list_files', 'arguments': {'path': '/potentials'}, 'result': '{\\n  \"files\": [\\n    \"/potentials/EAM\",\\n    \"/potentials/SW\",\\n    \"/potentials/TERSOFF\"\\n  ]\\n}', 'status': 'success', 'error_message': None, 'duration': 23.081051360000856, 'timestamp': '2025-11-07T10:14:02.425911+00:00'}",
        "name": "list_files",
    },
    {
        "role": "assistant",
        "content": '<thought>\nGood, there\'s an EAM directory. Let me check what EAM potentials are available.\n</thought>\n\n<action>list_files</action>\n<action_input>{"path": "/potentials/EAM"}</action_input>',
    },
    {
        "role": "user",
        "content": "Observation: {'tool_name': 'list_files', 'arguments': {'path': '/potentials/EAM'}, 'result': '{\\n  \"files\": [\\n    \"/potentials/EAM/Al99.eam.alloy\",\\n    \"/potentials/EAM/Cu_Zhou04.eam.alloy\",\\n    \"/potentials/EAM/Fe-C_Hepburn_Ackland.eam.fs\",\\n    \"/potentials/EAM/Mg_Zhou04.eam.alloy\"\\n  ]\\n}', 'status': 'success', 'error_message': None, 'duration': 20.953012432000833, 'timestamp': '2025-11-07T10:14:27.868243+00:00'}",
        "name": "list_files",
    },
    {
        "role": "assistant",
        "content": '<thought>\nPerfect! There\'s an Al99.eam.alloy potential file for Aluminum. Let me get its metadata to understand the pair style needed.\n</thought>\n\n<action>get_potential_metadata</action>\n<action_input>{"file_path": "/potentials/EAM/Al99.eam.alloy"}</action_input>',
    },
    {
        "role": "user",
        "content": "Observation: {'tool_name': 'get_potential_metadata', 'arguments': {'file_path': '/potentials/EAM/Al99.eam.alloy'}, 'result': '{potential type : EAM, elements supported : Al (Aluminum), pair_style : eam/alloy}', 'status': 'success', 'error_message': None, 'duration': 0.0030104380020929966, 'timestamp': '2025-11-07T10:14:32.706123+00:00'}",
        "name": "get_potential_metadata",
    },
    {
        "role": "assistant",
        "content": '<thought>\nGreat! Now I need to:\n1. Get the Aluminum structure from Materials Project (mp-134)\n2. Convert it to LAMMPS data format\n3. Create two LAMMPS input scripts:\n   - One for bulk (periodic in all directions)\n   - One for surface (non-periodic in z-direction)\n4. Run both simulations\n5. Calculate surface energy\n\nLet me start by getting the structure.\n</thought>\n\n<action>get_structure_from_mp_text</action>\n<action_input>{"mp_id": "mp-134", "file_path": "/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/Al_bulk.cif"}</action_input>',
    },
    {
        "role": "user",
        "content": "Observation: {'tool_name': 'get_structure_from_mp_text', 'arguments': {'mp_id': 'mp-134', 'file_path': '/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/Al_bulk.cif'}, 'result': 'Structure saved successfully at /results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/Al_bulk.cif', 'status': 'success', 'error_message': None, 'duration': 22.45011810699725, 'timestamp': '2025-11-07T10:15:02.209169+00:00'}",
        "name": "get_structure_from_mp_text",
    },
    {
        "role": "assistant",
        "content": '<thought>\nGood! Now I need to convert this CIF to LAMMPS data format. I\'ll use atom_style "atomic" as specified in the task.\n</thought>\n\n<action>convert_structure_to_lammps_data</action>\n<action_input>{"structure_path": "/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/Al_bulk.cif", "output_file": "/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/Al_bulk.data", "atom_style": "atomic"}</action_input>',
    },
    {
        "role": "user",
        "content": "Observation: {'tool_name': 'convert_structure_to_lammps_data', 'arguments': {'structure_path': '/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/Al_bulk.cif', 'output_file': '/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/Al_bulk.data', 'atom_style': 'atomic'}, 'result': 'LAMMPS data file successfully written to: /results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/Al_bulk.data', 'status': 'success', 'error_message': None, 'duration': 1.5582352279998304, 'timestamp': '2025-11-07T10:15:08.221066+00:00'}",
        "name": "convert_structure_to_lammps_data",
    },
    {
        "role": "assistant",
        "content": '<thought>\nPerfect! Now I need to create two LAMMPS input scripts:\n\n1. Bulk configuration: 5x5x5 supercell with periodic boundaries in all directions\n2. Surface configuration: 5x5x5 supercell with non-periodic boundary in z-direction (creating surfaces)\n\nFor the surface calculation, I need to add vacuum in the z-direction to separate the surfaces.\n\nLet me create the bulk input script first.\n</thought>\n\n<action>write_file</action>\n<action_input>{"path": "/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/bulk.in", "content": "# LAMMPS input script for bulk Aluminum energy calculation\\n\\n# Initialization\\nunits metal\\natom_style atomic\\ndimension 3\\nboundary p p p\\n\\n# Read structure\\nread_data Al_bulk.data\\n\\n# Replicate 5x5x5\\nreplicate 5 5 5\\n\\n# Define potential\\npair_style eam/alloy\\npair_coeff * * /potentials/EAM/Al99.eam.alloy Al\\n\\n# Energy minimization\\nmin_style cg\\nminimize 1e-10 1e-10 100000 100000\\n\\n# Output final energy\\nvariable natoms equal \\"count(all)\\"\\nvariable etotal equal \\"pe\\"\\nvariable eatom equal \\"v_etotal/v_natoms\\"\\n\\nprint \\"Total atoms: ${natoms}\\"\\nprint \\"Total energy: ${etotal} eV\\"\\nprint \\"Energy per atom: ${eatom} eV/atom\\"\\n"}</action_input>',
    },
    {
        "role": "user",
        "content": "Observation: {'tool_name': 'write_file', 'arguments': {'path': '/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/bulk.in', 'content': '# LAMMPS input script for bulk Aluminum energy calculation\\n\\n# Initialization\\nunits metal\\natom_style atomic\\ndimension 3\\nboundary p p p\\n\\n# Read structure\\nread_data Al_bulk.data\\n\\n# Replicate 5x5x5\\nreplicate 5 5 5\\n\\n# Define potential\\npair_style eam/alloy\\npair_coeff * * /potentials/EAM/Al99.eam.alloy Al\\n\\n# Energy minimization\\nmin_style cg\\nminimize 1e-10 1e-10 100000 100000\\n\\n# Output final energy\\nvariable natoms equal \"count(all)\"\\nvariable etotal equal \"pe\"\\nvariable eatom equal \"v_etotal/v_natoms\"\\n\\nprint \"Total atoms: ${natoms}\"\\nprint \"Total energy: ${etotal} eV\"\\nprint \"Energy per atom: ${eatom} eV/atom\"\\n'}, 'result': 'Successfully wrote to /results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/bulk.in', 'status': 'success', 'error_message': None, 'duration': 1.8750712599976396, 'timestamp': '2025-11-07T10:15:20.471495+00:00'}",
        "name": "write_file",
    },
    {
        "role": "assistant",
        "content": '<thought>\nGood! Now let me create the surface input script. For the surface calculation, I need to:\n1. Create a 5x5x5 supercell\n2. Change boundary condition in z to non-periodic (shrink-wrapped)\n3. Add vacuum in z-direction to separate the two surfaces\n4. Minimize the structure\n\nThe vacuum should be large enough to avoid interaction between periodic images.\n</thought>\n\n<action>write_file</action>\n<action_input>{"path": "/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/surface.in", "content": "# LAMMPS input script for surface Aluminum energy calculation\\n\\n# Initialization\\nunits metal\\natom_style atomic\\ndimension 3\\nboundary p p p\\n\\n# Read structure\\nread_data Al_bulk.data\\n\\n# Replicate 5x5x5\\nreplicate 5 5 5\\n\\n# Change boundary condition in z to non-periodic\\nchange_box all boundary p p s\\n\\n# Add vacuum in z-direction (15 Angstroms on each side)\\nvariable zlo equal \\"zlo\\"\\nvariable zhi equal \\"zhi\\"\\nvariable vacuum equal 15.0\\nchange_box all z final $(v_zlo-v_vacuum) $(v_zhi+v_vacuum) units box\\n\\n# Define potential\\npair_style eam/alloy\\npair_coeff * * /potentials/EAM/Al99.eam.alloy Al\\n\\n# Energy minimization\\nmin_style cg\\nminimize 1e-10 1e-10 100000 100000\\n\\n# Output final energy and box dimensions\\nvariable natoms equal \\"count(all)\\"\\nvariable etotal equal \\"pe\\"\\nvariable eatom equal \\"v_etotal/v_natoms\\"\\nvariable lx equal \\"lx\\"\\nvariable ly equal \\"ly\\"\\nvariable area equal \\"v_lx*v_ly\\"\\n\\nprint \\"Total atoms: ${natoms}\\"\\nprint \\"Total energy: ${etotal} eV\\"\\nprint \\"Energy per atom: ${eatom} eV/atom\\"\\nprint \\"Surface area: ${area} Angstrom^2\\"\\n"}</action_input>',
    },
    # {
    #   "role": "user",
    #   "content": "Observation: {'tool_name': 'write_file', 'arguments': {'path': '/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/surface.in', 'content': '# LAMMPS input script for surface Aluminum energy calculation\\n\\n# Initialization\\nunits metal\\natom_style atomic\\ndimension 3\\nboundary p p p\\n\\n# Read structure\\nread_data Al_bulk.data\\n\\n# Replicate 5x5x5\\nreplicate 5 5 5\\n\\n# Change boundary condition in z to non-periodic\\nchange_box all boundary p p s\\n\\n# Add vacuum in z-direction (15 Angstroms on each side)\\nvariable zlo equal \"zlo\"\\nvariable zhi equal \"zhi\"\\nvariable vacuum equal 15.0\\nchange_box all z final $(v_zlo-v_vacuum) $(v_zhi+v_vacuum) units box\\n\\n# Define potential\\npair_style eam/alloy\\npair_coeff * * /potentials/EAM/Al99.eam.alloy Al\\n\\n# Energy minimization\\nmin_style cg\\nminimize 1e-10 1e-10 100000 100000\\n\\n# Output final energy and box dimensions\\nvariable natoms equal \"count(all)\"\\nvariable etotal equal \"pe\"\\nvariable eatom equal \"v_etotal/v_natoms\"\\nvariable lx equal \"lx\"\\nvariable ly equal \"ly\"\\nvariable area equal \"v_lx*v_ly\"\\n\\nprint \"Total atoms: ${natoms}\"\\nprint \"Total energy: ${etotal} eV\"\\nprint \"Energy per atom: ${eatom} eV/atom\"\\nprint \"Surface area: ${area} Angstrom^2\"\\n'}, 'result': 'Successfully wrote to /results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/surface.in', 'status': 'success', 'error_message': None, 'duration': 1.914784859000065, 'timestamp': '2025-11-07T10:15:32.449181+00:00'}",
    #   "name": "write_file"
    # },
    # {
    #   "role": "assistant",
    #   "content": "<thought>\nNow let me run the bulk simulation first to get the bulk energy.\n</thought>\n\n<action>run_lammps</action>\n<action_input>{\"input_file\": \"/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/bulk.in\"}</action_input>"
    # },
    # {
    #   "role": "user",
    #   "content": "Observation: {'tool_name': 'run_lammps', 'arguments': {'input_file': '/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/bulk.in'}, 'result': 'Simulation ran successfully using input: /results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/bulk.in, log saved at: bulk.log', 'status': 'success', 'error_message': None, 'duration': 4.183847636002611, 'timestamp': '2025-11-07T10:15:40.623065+00:00'}",
    #   "name": "run_lammps"
    # },
    # {
    #   "role": "assistant",
    #   "content": "<thought>\nGreat! The bulk simulation completed. Now let me run the surface simulation.\n</thought>\n\n<action>run_lammps</action>\n<action_input>{\"input_file\": \"/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/surface.in\"}</action_input>"
    # },
    # {
    #   "role": "user",
    #   "content": "Observation: {'tool_name': 'run_lammps', 'arguments': {'input_file': '/results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/surface.in'}, 'result': 'Simulation ran successfully using input: /results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/surface.in, log saved at: surface.log', 'status': 'success', 'error_message': None, 'duration': 4.6097025019989815, 'timestamp': '2025-11-07T10:15:48.903516+00:00'}",
    #   "name": "run_lammps"
    # },
    {
        "role": "user",
        "content": "Important instruction: files generated previously can be found at /results/6_Nov_Optimised/claude_45/react/surface_energy/subtask_false/level_1/workflow/aluminum_surface_energy_1_trial_0/, which is independent from the working directory.",
    },
]
successful_hooks = [
    create_intervention_hook(
        {"aluminum_surface_energy_1": cont["content"]}, execute_tools=False
    )
    for cont in successful_partial
]

# successful_map_1 = {
#     "aluminum_surface_energy_1": successful_partial_2,
# }

# successful_map_2 = {
#     "aluminum_surface_energy_1": successful_partial_2,
# }

# failed_map = {"aluminum_surface_energy_1": failed_partial}


# successful_hook = create_intervention_hook(successful_map, execute_tools=False)
# failed_hook = create_intervention_hook(failed_map, execute_tools=False)


def logprobs_hook(context: HookContext) -> None:
    """Extract, convert to dict, and save logprobs."""
    llm_response = context.iteration_data.get("llm_response")
    if not llm_response:
        return

    log_probs_obj = getattr(llm_response, "logprobs", None)
    if not log_probs_obj:
        return

    # Convert Pydantic object to a standard Python Dict once
    try:
        if hasattr(log_probs_obj, "model_dump"):
            lp_dict = log_probs_obj.model_dump()
        elif hasattr(log_probs_obj, "dict"):
            lp_dict = log_probs_obj.dict()
        else:
            lp_dict = log_probs_obj

        # Store in metadata so other hooks can use the clean DICT
        context.metadata["current_logprobs_dict"] = lp_dict

    except Exception as e:
        logger.error(f"Failed to convert logprobs to dict: {e}")
        return

    # Save to disk as an artifact (optional, but good for records)
    lp_data = {
        "id": llm_response.id,
        "task_id": context.task_id,
        "iteration": context.iteration,
        "logprobs": lp_dict,
    }

    out_dir = Path("logprobs")
    out_dir.mkdir(exist_ok=True, parents=True)
    log_prob_hook_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    context.metadata["logprob_hook_timestamp"] = log_prob_hook_timestamp
    file_path = out_dir / f"{llm_response.id}_{log_prob_hook_timestamp}.json"

    with file_path.open("w") as f:
        json.dump(lp_data, f, indent=2)
    logger.info(f"Saved logprobs to {file_path}")


def compute_logprob_metrics(
    logprobs: dict, log_prob_hook_timestamp
) -> dict[str, float]:
    if not logprobs or "content" not in logprobs:
        return {"avg_logprob": 0, "entropy": 0, "variance": 0, "avg_top_entropy": 0}

    token_logprobs = [tok["logprob"] for tok in logprobs["content"] if "logprob" in tok]
    if not token_logprobs:
        return {"avg_logprob": 0, "entropy": 0, "variance": 0, "avg_top_entropy": 0}

    avg_logprob = np.mean(token_logprobs)
    probs = np.exp(token_logprobs)
    entropy = -np.sum(probs * np.log(probs + 1e-10)) / len(probs)
    variance = np.var(token_logprobs)

    # Optional: Avg entropy over top_logprobs per token (for alternative sharpness)
    top_entropies = []
    for tok in logprobs["content"]:
        if "top_logprobs" in tok:
            top_lps = [tl["logprob"] for tl in tok["top_logprobs"]]
            if top_lps:
                top_probs = np.exp(top_lps) / np.sum(np.exp(top_lps))  # Normalize
                top_ent = -np.sum(top_probs * np.log(top_probs + 1e-10))
                top_entropies.append(top_ent)
    avg_top_entropy = np.mean(top_entropies) if top_entropies else 0

    return {
        "avg_logprob": avg_logprob,
        "entropy": entropy,
        "variance": variance,
        "avg_top_entropy": avg_top_entropy,
        "path_length": len(token_logprobs),
        "log_prob_hook_timestamp": log_prob_hook_timestamp,
    }


def save_metrics_to_file(
    task_id: str,
    iteration: int,
    metrics: dict[str, float],
    log_prob_hook_timestamp=None,
) -> None:
    """
    Save computed metrics to a JSON file.

    Args:
        task_id: The current task identifier
        iteration: The iteration number
        metrics: Dictionary of metrics (e.g., {'avg_logprob': -0.5, 'entropy': 1.2, ...})
    """
    out_dir = Path("metrics")
    out_dir.mkdir(exist_ok=True, parents=True)

    file_path = (
        out_dir / f"{task_id}_iteration_{iteration}_{log_prob_hook_timestamp}.json"
    )
    try:
        with file_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")


def metrics_hook(context: HookContext) -> None:
    """Compute and save metrics using the dict already in memory."""
    # 1. Try to get the pre-converted dict from the previous hook
    lp_dict = context.metadata.get("current_logprobs_dict")

    # 2. Fallback: If logprobs_hook didn't run, try to get it from the response object
    if not lp_dict:
        llm_response = context.iteration_data.get("llm_response")
        if not llm_response or not getattr(llm_response, "logprobs", None):
            return

        # (Conversion logic if needed)
        obj = llm_response.logprobs
        lp_dict = obj.model_dump() if hasattr(obj, "model_dump") else obj

    # 3. Compute metrics using the in-memory dictionary
    log_prob_hook_timestamp = context.metadata["logprob_hook_timestamp"]
    metrics = compute_logprob_metrics(lp_dict, log_prob_hook_timestamp)

    # Store metrics in context for logging or final report
    context.metadata[f"iteration_{context.iteration}_metrics"] = metrics

    # Save metrics to file
    save_metrics_to_file(
        context.task_id, context.iteration, metrics, log_prob_hook_timestamp
    )


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def move_work_dir_files(work_dir: str, target_subdir: str):
    """
    Moves all contents of work_dir into a new subdirectory within work_dir,
    and creates the target subdirectory if it doesn't exist.
    """
    if not work_dir:
        logger.warning("CORRAL_WORK_DIR is not set. Skipping file movement.")
        return

    target_path = Path(work_dir) / target_subdir
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Moving work directory contents to: {target_path}")

    # Move all files and folders *except* the target subdirectory itself
    for item_path in Path(work_dir).iterdir():
        if item_path.name != target_subdir:
            try:
                shutil.move(item_path, target_path)
            except Exception as e:
                logger.error(f"Failed to move {item_path}: {e!s}")

    logger.success("Work directory files moved successfully.")


def run_benchmark(
    model: str = "claude-3-5-sonnet-20241022",
    _task_ids: list | None = None,
    temperature: float = 0.0,
    run_name: str = "corral_benchmark_run",
    verbose: str = "brief",
    hooks=None,
):
    """Run the benchmark with specified model and tasks"""

    interface = CorralRouter(base_url="http://localhost:8000")
    wandblogger = CorralWandbLogger(
        project="corral_md_oss",
        group="gpt_oss",
        name=run_name,
    )
    agent = ReActAgent(
        model=model,
        logprobs=True,
        top_logprobs=20,
        max_iterations=20,
        temperature=temperature,
        api_endpoint="https://api.helmholtz-blablador.fz-juelich.de/v1",
    )
    runner = CorralRunner(
        interface,
        agent,
        logger=wandblogger,
    )

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        task_ids=[
            "aluminum_surface_energy_1",
        ],
        trials_per_task=5,
        k_values=[1, 2, 3, 4, 5],
        verbose=True,
        tool_verbosity=verbose,
        hooks=hooks,
    )
    result.generate_report(f"{run_name}_try.json")
    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()
    os.environ["OPENAI_API_KEY"] = os.getenv("BLABLADOR_API_KEY_TEST", "")

    # corral_work_dir = os.getenv("CORRAL_WORK_DIR")
    corral_work_dir = "./"
    if not corral_work_dir:
        logger.error("CORRAL_WORK_DIR environment variable is not set!")

    verbose = "workflow"
    conditions = ["successful"]
    for condition in conditions:
        hooks = AgentHooks()  # Fresh hooks per condition
        if condition == "successful":
            for _hook in successful_hooks:
                hooks.register(HookPoint.BEFORE_TASK, _hook)
        hooks.register(HookPoint.AFTER_ITERATION, logprobs_hook)
        hooks.register(HookPoint.AFTER_ITERATION, metrics_hook)  # Always on

        run_name = (
            f"gpt_oss_120-react-corral_md_{condition}-{verbose}_verbosity-single_12"
        )
        model = (
            "openai/1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
        )
        run_benchmark(model=model, run_name=run_name, verbose=verbose, hooks=hooks)
        move_work_dir_files(corral_work_dir, verbose)
