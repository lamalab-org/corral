import json
import random
from pathlib import Path

# Path to the input JSON file
input_path = Path("./tasks_open.json")

prompts = [
    "You're in a laboratory setting with a sample of an organic compound. Analyze the sample and return its SMILES representation, minimizing the use of costly and time-intensive resources.",
    "Given an organic compound sample in a lab environment, your objective is to determine its SMILES string while conserving resources due to the high cost of analysis.",
    "In the lab, you've received a sample of an organic compound. Perform a careful analysis and provide the SMILES code, using only the necessary time and materials.",
    "You are operating in a lab with a sample of an organic compound. Extract its SMILES string efficiently, keeping in mind that the analysis is both expensive and slow.",
    "Analyze the provided organic compound sample in a lab environment and output the SMILES string, while minimizing resource consumption due to the costly nature of the process.",
    "Working in a laboratory, you're tasked with identifying the SMILES string of an organic sample. Be efficient, as the analysis is resource-intensive and time-consuming.",
    "Within a laboratory context, you have a sample of an organic molecule. Determine its SMILES notation carefully, using minimal but sufficient analysis.",
    "You're tasked with analyzing an organic compound in a lab. Return its SMILES string while ensuring that resource use is justified given the high cost of each analysis.",
    "Inside the lab, you have a sample of an organic compound. Identify its SMILES string using only essential resources, as analysis is both costly and time-heavy.",
    "In this laboratory scenario, an organic sample needs to be analyzed to retrieve its SMILES string. Each test is expensive, so proceed with resource-awareness.",
]


question_1 = "What is the molecular formula of the compound of the sample at hand?"
question_1_rephrased = [
    "What is the chemical formula for the sample compound?",
    "Can you provide the molecular formula of the compound present in the sample?",
    "Identify the molecular formula of the compound in this sample.",
    "What formula represents the molecule in the sample?",
    "State the molecular formula for the compound analyzed.",
]
question_2 = "What is the double bond equivalent (DBE) or the number of insaturations of the compound of the sample at hand?"
question_2_rephrased = [
    "How many double bond equivalents (DBE) does the sample compound have?",
    "What is the degree of unsaturation in the compound from the sample?",
    "Indicate the number of insaturations (DBE) for the sample's compound.",
    "What is the DBE value for the compound in this sample?",
    "How many unsaturations are present in the compound analyzed?",
]
question_3 = "Which fragment ions in the mass spectrum can be assigned or confirmed based on their isotope distribution? If None reply with 'None'."
question_3_rephrased = [
    "Which mass spectrum fragment ions are identifiable by their isotope patterns?",
    "Can any fragment ions be confirmed using isotope distribution in the mass spectrum?",
    "List the fragment ions in the spectrum that can be assigned via isotope distribution.",
    "Are there fragment ions in the mass spectrum confirmed by isotope distribution?",
    "Which ions in the mass spectrum are validated by their isotope distribution?",
]
question_4 = "What are the number of chemically equivalent carbon atoms in the molecule of the sample at hand?"
question_4_rephrased = [
    "How many chemically equivalent carbon atoms are present in the sample's molecule?",
    "State the count of equivalent carbon atoms in the compound.",
    "What is the number of carbon atoms with chemical equivalence in the molecule?",
    "How many carbon atoms in the sample's molecule are chemically equivalent?",
    "Indicate the number of chemically equivalent carbons in the compound.",
]
question_5 = "What are the number of chemically equivalent protons in the molecule of the sample at hand?"
question_5_rephrased = [
    "How many chemically equivalent protons are found in the sample's molecule?",
    "State the number of equivalent protons in the compound.",
    "What is the count of protons with chemical equivalence in the molecule?",
    "How many protons in the sample's molecule are chemically equivalent?",
    "Indicate the number of chemically equivalent protons in the compound.",
]
question_6 = "How many aromatic carbon atoms are in the molecule of the sample at hand? If there are no aromatic carbon atoms, reply with '0'."
question_6_rephrased = [
    "What is the number of aromatic carbons in the sample's molecule?",
    "How many aromatic carbon atoms does the compound contain?",
    "State the count of aromatic carbons in the molecule.",
    "How many aromatic carbon atoms in the sample's compound? If not, reply '0'.",
    "Indicate the number of aromatic carbon atoms present in the molecule.",
]
question_7 = "What is the number of CH3 groups in the molecule of the sample at hand? If there are no CH3 groups, reply with '0'."
question_7_rephrased = [
    "How many methyl (CH3) groups are present in the sample's molecule?",
    "State the number of CH3 groups in the compound.",
    "What is the count of methyl groups in the molecule?",
    "How many CH3 groups in the sample's compound? If not, reply '0'.",
    "Indicate the number of CH3 groups found in the molecule.",
]
question_8 = "How many cabonyl groups (C=O) are in the molecule of the sample at hand? If there are no carbonyl groups, reply with '0'."
question_8_rephrased = [
    "What is the number of carbonyl (C=O) groups in the sample's molecule?",
    "How many C=O groups does the compound contain?",
    "State the count of carbonyl groups in the molecule.",
    "How many carbonyl groups in the sample's compound? If not, reply '0'.",
    "Indicate the number of C=O groups present in the molecule.",
]
question_9 = "Link as many fragments as possible of the molecule of the sample at hand and return the SMILES of the different fragments."
question_9_rephrased = [
    "Connect as many molecular fragments as possible and provide their SMILES representations.",
    "Link the fragments of the sample's molecule and list their SMILES strings.",
    "Identify and link fragments from the molecule, returning their SMILES.",
    "Provide SMILES for all possible linked fragments of the sample's molecule.",
    "Return the SMILES strings for the connected fragments of the compound.",
]
question_10 = "What is the SMILES string of the compound of the sample at hand?"
question_10_rephrased = [
    "What is the SMILES notation for the sample's compound?",
    "Provide the SMILES string representing the compound in the sample.",
    "State the SMILES for the molecule analyzed.",
    "What SMILES string corresponds to the compound in this sample?",
    "Indicate the SMILES representation of the sample's compound.",
]

sub_keywords = [
    "molecular-formula",
    "double-bond-equivalent",
    "isotopic-distribution",
    "hydrogen-symmetry-classes",
    "carbon-symmetry-classes",
    "aromatic-carbons",
    "ch3-groups",
    "carbonyl-groups",
    "hydroxyl-and-amine-groups",
    "fragment-elucidation",
    "smiles",
]

# Output directory for the new JSON files
output_dir = input_path.parent / "tasks_json"
output_dir.mkdir(parents=True, exist_ok=True)

# Load the input JSON data
with input_path.open(encoding="utf-8") as f:
    tasks = json.load(f)

keywords = ["spectra", "spectra-elucidation", "experimental-spectra"]
metrics = ["binary"]

for i, task in enumerate(tasks, 1):
    prompt = random.choice(prompts)

    # Subtask: fragment elucidation
    subtasks = []
    for j in range(1, 11):
        if j == 2:
            input_from_task = [f"{task['name']}_subtask_1"]
        elif j == 9:
            input_from_task = [f"{task['name']}_subtask_{k}" for k in range(1, 9)]
        elif j == 10:
            input_from_task = [f"{task['name']}_subtask_{k}" for k in range(1, 10)]
        else:
            input_from_task = False

        input_for_task = []
        if j == 1:
            input_for_task = [
                f"{task['name']}_subtask_2",
                f"{task['name']}_subtask_9",
                f"{task['name']}_subtask_10",
            ]
        elif j == 9:
            input_for_task = [f"{task['name']}_subtask_10"]
        elif j == 10:
            input_for_task = False
        else:
            input_for_task = [f"{task['name']}_subtask_9", f"{task['name']}_subtask_10"]

        tools = []
        if j == 1:
            tools = ["mass_spectrometry_spectra"]
        elif j == 2:
            tools = ["retrieve_dbe_formula"]
        elif j == 3:
            tools = ["mass_spectrometry_spectra", "retrieve_isotope_distribution"]
        elif j == 10:
            tools = []
        elif j == 4:
            tools = ["carbon_nmr_spectra", "retrieve_carbon_shifts"]
        elif j == 5:
            tools = [
                "proton_nmr_spectra",
                "retrieve_protons_shifts",
                "retrieve_aromatic_protons_shifts",
            ]
        elif j == 6 or j == 7:
            tools = [
                "carbon_nmr_spectra",
                "proton_nmr_spectra",
                "hsqc_nmr_spectra",
                "retrieve_aromatic_protons_shifts",
                "retrieve_carbon_shifts",
                "retrieve_protons_shifts",
            ]
        elif j == 8:
            tools = [
                "carbon_nmr_spectra",
                "proton_nmr_spectra",
                "hsqc_nmr_spectra",
                "ir_spectra",
                "retrieve_carbon_shifts",
                "retrieve_protons_shifts",
            ]
        elif j == 9:
            tools = [
                "hsqc_nmr_spectra",
                "mass_spectrometry_spectra",
                "retrieve_carbon_shifts",
                "retrieve_protons_shifts",
                "retrieve_aromatic_protons_shifts",
            ]

        question_rephrased_list = globals()[f"question_{j}_rephrased"]
        subtask_prompt = random.choice(question_rephrased_list)
        subtask = {
            "id": f"{task['name']}_subtask_{j}",
            "name": f"task_{i}_subtask_{j}",
            "keywords": [*keywords, sub_keywords[j - 1]],
            "metrics": metrics,
            "input": {
                "prompt": subtask_prompt,
                "input_from_task": input_from_task,
                "input_for_task": input_for_task,
            },
            "tools": tools,
            "output": [{"type": "string", "target": task["smiles"], "threshold": None}],
            "scoring_fn": j,
        }
        subtasks.append(subtask)

    new_task = {
        "id": task["name"],
        "name": f"task_{i}",
        "keywords": keywords,
        "metrics": metrics,
        "input": {"prompt": prompt, "input_from_task": False, "input_for_task": False},
        "output": [{"type": "string", "target": task["smiles"], "threshold": None}],
        "scoring_fn": "score_molecule",
        "subtasks": subtasks,
    }
    output_path = output_dir / f"task_{i}.json"
    with output_path.open("w", encoding="utf-8") as out_f:
        json.dump(new_task, out_f, indent=2, ensure_ascii=False)
