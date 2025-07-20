import json
import random
from pathlib import Path

# Path to the input JSON file
input_path = Path("./tasks_open.json")

prompts = [
    "Based on the characterization data in the following, output only the SMILES string of the matching compound: {task['spectra']}.",
    "From the following spectral data {task['spectra']}, derive and return the compound's SMILES—nothing else.",
    "Using these characterization data {task['spectra']}, give the SMILES string of the matching compound alone.",
    "Return just the SMILES for the compound that matched the following characterization data {task['spectra']}.",
    "Provide the SMILES corresponding to compound matching the spectra {task['spectra']}; no additional text.",
    "Output solely the SMILES representation for the compound with these spectral details: {task['spectra']}.",
    "Based on the following spectra {task['spectra']}, return only the compound's SMILES.",
    "Deliver the SMILES string for the molecule characterized by {task['spectra']}, with no extra information.",
    "Give the SMILES representation of the compound characterized by the spectra here: {task['spectra']}.",
    "Report only the SMILES string derived from these characterization data: {task['spectra']}.",
]

prompts_2 = [
    "Based on the spectral information in {task['spectra']}, enumerate the SMILES strings of all fragments that the spectra reveal.",
    "From the spectra provided ({task['spectra']}), deduce every possible fragment and output each as a SMILES representation.",
    "Inspect the dataset {task['spectra']} and return the SMILES codes corresponding to every fragment indicated by the spectra.",
    "Using the spectral data in {task['spectra']}, identify each distinct fragment and list its SMILES notation.",
    "Given {task['spectra']}, derive all fragment structures implied by the spectra and supply their SMILES strings.",
    "Examine the spectra contained in {task['spectra']} and output the SMILES for each fragment you can confidently infer.",
    "With reference to the spectral dataset {task['spectra']}, extract every deducible fragment and provide its SMILES code.",
    "Parse the spectral readings in {task['spectra']} and enumerate the SMILES representations of the constituent fragments.",
    "Review {task['spectra']} and produce a list of SMILES strings for all fragments suggested by the spectral features.",
    "From the spectral evidence in {task['spectra']}, infer all fragment structures and output them as SMILES strings, one per line.",
]

prompt_3 = [
    "Using the provided fragments {task['fragments']} and the spectral data {task['spectra']}, output the SMILES string of the molecule responsible for the spectrum.",
    "Determine the SMILES representation of the compound that generated {task['spectra']} given its molecular fragments {task['fragments']}.",
    "From the fragment set {task['fragments']} and the accompanying spectral dataset {task['spectra']}, identify and return the corresponding molecule in SMILES format.",
    "Based on the fragments {task['fragments']} along with spectra {task['spectra']}, deduce the SMILES of the originating molecule.",
    "Infer the chemical structure as a SMILES string that matches the spectra {task['spectra']} using the fragments {task['fragments']}.",
    "Given fragment information {task['fragments']} plus spectral details {task['spectra']}, produce the SMILES code of the molecule.",
    "Identify the molecule (SMILES) responsible for the spectral profile {task['spectra']} utilizing the fragment list {task['fragments']}.",
    "With access to {task['fragments']} and spectra {task['spectra']}, compute the SMILES representation of the compound.",
    "Find the SMILES string corresponding to the spectra data {task['spectra']} using the supplied molecular fragments {task['fragments']}.",
    "Return the SMILES of the molecule that matches the given spectra {task['spectra']} and fragment set {task['fragments']}.",
]


# Output directory for the new JSON files
output_dir = input_path.parent / "tasks_json"
output_dir.mkdir(parents=True, exist_ok=True)


# Load the input JSON data
with input_path.open(encoding="utf-8") as f:
    tasks = json.load(f)

keywords = ["spectra", "spectra-elucidation", "experimental-spectra"]
metrics = ["binary"]
scoring_fn = "score_molecule_similarity"
scoring_subtasks = "score_molecule_fragments_subtasks"

for i, task in enumerate(tasks, 1):
    prompt_template = random.choice(prompts)
    prompt = prompt_template.replace("{task['spectra']}", str(task["spectra"]))

    # Subtask: fragment elucidation
    subtask_prompt_template = random.choice(prompts_2)
    subtask_prompt = subtask_prompt_template.replace(
        "{task['spectra']}", str(task["spectra"])
    )
    subtask = {
        "id": f"{task['name']}_fragment",
        "name": f"task_{i}_fragment",
        "keywords": [*keywords, "fragment-elucidation"],
        "metrics": metrics,
        "input": {
            "prompt": subtask_prompt,
            "input_from_task": False,
            "input_for_task": f"{task['name']}_final",
        },
        "output": [
            {"type": "string", "target": task.get("fragments", []), "threshold": None}
        ],
        "scoring_fn": scoring_subtasks,
        "subtasks": [],
    }
    subtask_2_prompt_template = random.choice(prompt_3)
    # Keep {task['fragments']} as a fillable input, only replace {task['spectra']}
    subtask_2_prompt = subtask_2_prompt_template.replace(
        "{task['spectra']}", str(task["spectra"])
    )
    subtask_2 = {
        "id": f"{task['name']}_final",
        "name": f"task_{i}_final",
        "keywords": [*keywords, "fragment2molecule"],
        "metrics": metrics,
        "input": {
            "prompt": subtask_2_prompt,
            "input_from_task": f"{task['name']}_fragment",
            "input_for_task": False,
        },
        "output": [{"type": "string", "target": task["smiles"], "threshold": None}],
        "scoring_fn": scoring_fn,
        "subtasks": [],
    }

    new_task = {
        "id": task["name"],
        "name": f"task_{i}",
        "keywords": keywords,
        "metrics": metrics,
        "input": {"prompt": prompt, "input_from_task": False, "input_for_task": False},
        "output": [{"type": "string", "target": task["smiles"], "threshold": None}],
        "scoring_fn": scoring_fn,
        "subtasks": [subtask, subtask_2],
    }
    output_path = output_dir / f"task_{i}.json"
    with output_path.open("w", encoding="utf-8") as out_f:
        json.dump(new_task, out_f, indent=2, ensure_ascii=False)
