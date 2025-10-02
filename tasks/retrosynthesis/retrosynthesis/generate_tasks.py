import json
from pathlib import Path

from loguru import logger

MOLECULES = [
    "C=C(C1=CC=CC=C1)[C@@H]2CCC[C@@]2(O)C(F)(F)F",  # Carbony-En Reaction
    "COC([C@]12CC=CC[C@H]1C(C2)=O)=O",  # Diels-Alder
    "COC(C1(C=C1[Si](C)(C)C)/C=C/C2=CC=CC=C2)=O",  # Cycloaddition
    "C=C(C1=CC=C(OC)C=C1)C2=CC=C(OC)C=C2",  # Peterson olefination
    "C=C(C1=CC([C@H]([C@@]1(CC(C(CO[Si](C)(C(C)(C)C)C)=C)=O)[H])C)=O)C",  # Nozaki Hiyama Kishi Reaction
    # Difficult ones
    "O[C@H]1C[C@@H](O[C@@H]([C@@H]1C)/C=C(CO)/C)C/C=C/C=C/C(O)=O",  # Still-Gennari + Horner-Wadsworth-Emmons
    "O[C@H]1C2=CC3=CC=CC=C3O[C@H]2CCC1",  # Baylis-Hillman Reaction + cyclic stereocontrol
    "CC(c1c(CCCC2)c2c(OS(=O)(C(F)(F)F)=O)cc1)=O",
    "COc1ccc2c(c1)cc(-c1ccccc1)n2Cc1cccc(-c2noc(=O)[nH]2)n1",
    "O=S(NC1=CC2=C(OC3(CC2)CCC3)C(N4CCNCC4)=C1)(C5=C(F)C=CC=C5)=O",
]
PRIZES = [
    300.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    9999.0,
]


def main():
    tasks_path = Path(__file__).parent / "tasks_json"
    for i, molecule in enumerate(MOLECULES):
        task = {
            "id": f"make_{i}",
            "name": f"Synthesize)_{molecule}",
            "keywords": ["chemistry", "synthesis", "retrosynthesis"],
            "metrics": ["binary"],
            "input": {
                "prompt": f"Propose a retrosynthesis route to synthesize the molecule with SMILES {molecule}. The final precursors must be buyable and the total cost must not exceed ${PRIZES[i]}.",
                "input_from_task": False,
                "input_for_task": False,
            },
            "output": [
                {
                    "type": "integer",
                    "target": PRIZES[i],
                    "threshold": None,
                }
            ],
            "scoring_fn": "final_score",
            "submission_format": """Submit a JSON object representing the retrosynthesis route. It must follow the next JSON format: `{\n  "type": "mol",\n  "smiles": "CO",\n  "children": [\n    {\n      "type": "reaction",\n      "template_id": "template_x",\n      "children": [\n        {\n          "type": "mol",\n          "smiles": "BrC"\n        },\n        {\n          "type": "mol",\n          "smiles": "[OH-]"\n        }\n      ]\n    }\n  ]\n}`.""",
        }
        task_file = tasks_path / f"make_{i}.json"
        with task_file.open("w") as f:
            json.dump(task, f, indent=4)
        logger.info(f"Generated task file: {task_file}")


if __name__ == "__main__":
    main()
