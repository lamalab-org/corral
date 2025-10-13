import json
from pathlib import Path

from loguru import logger

MOLECULES = [
    "C=C(c1ccccc1)[C@@H]1CCC[C@@]1(O)C(F)(F)F",  # Carbony-En Reaction
    "COC([C@]12CC=CC[C@H]1C(C2)=O)=O",  # Diels-Alder
    "COC(C1(C=C1[Si](C)(C)C)/C=C/C2=CC=CC=C2)=O",  # Cycloaddition
    "C=C(C1=CC=C(OC)C=C1)C2=CC=C(OC)C=C2",  # Peterson olefination
    "C=C(C1=CC([C@H]([C@@]1(CC(C(CO[Si](C)(C(C)(C)C)C)=C)=O)[H])C)=O)C",  # Nozaki Hiyama Kishi Reaction + Deprotection of OTBS into OH
    # Difficult ones
    "O[C@H]1C[C@@H](O[C@@H]([C@@H]1C)/C=C(CO)/C)C/C=C/C=C/C(O)=O",  # Still-Gennari + Horner-Wadsworth-Emmons
    "O[C@H]1C2=CC3=CC=CC=C3O[C@H]2CCC1",  # Baylis-Hillman Reaction + cyclic stereocontrol
    "COc1cccc(NC(=O)c2nnn(Cc3ccc(CN4CC(F)C4)cc3)c2N)c1",
    "Cc1ccc(NS(=O)(=O)c2ccc(/C=C/C(=O)Nc3ccccc3N)cc2)cc1",
    "O=S(NC1=CC(N2CCN(C(OC(C)(C)C)=O)CC2)=C3C(CCC4(CCC4)O3)=C1)(C5=C(F)C=CC=C5)=O",
]
PRIZES = [
    ["1914396"],
    ["1914397"],
    ["1914398"],
    ["1679759"],
    ["36006", "1914399", "1914400"],
    ["29648", "1914401", "1914402", "1914403"],
    ["1914404", "337284"],
    ["1914405", "1914406", "1914407"],
    ["324328", "1914408", "733"],
    ["20810", "2895", "1914409", "1914410", "1914411", "74060"],
]


def main():
    tasks_path = Path(__file__).parent.parent / "environments" / "level_3" / "tasks"
    for i, molecule in enumerate(MOLECULES):
        task = {
            "id": f"make_{i+1}_lvl1",
            "name": f"synthesize_{molecule}",
            "keywords": ["chemistry", "synthesis", "retrosynthesis"],
            "metrics": ["binary"],
            "input": {
                "prompt": f"Propose a retrosynthesis route to synthesize the molecule with SMILES {molecule}. The leaves on the retrosynthesis tree should be commercially available chemicals.",
                "input_from_task": False,
                "input_for_task": False,
            },
            "output": [
                {
                    "type": "integer",
                    "target": 9999.0,
                    "threshold": None,
                }
            ],
            "scoring_fn": "score_final",
            "submission_format": """Submit a JSON object representing the retrosynthesis route. It must follow the same JSON format as the next example: `{\n  "type": "mol",\n  "smiles": "CO",\n  "children": [\n    {\n      "type": "reaction",\n      "template_id": "template_x",\n      "children": [\n        {\n          "type": "mol",\n          "smiles": "BrC"\n        },\n        {\n          "type": "mol",\n          "smiles": "[OH-]"\n        }\n      ]\n    }\n  ]\n}`.""",
        }
        task_file = tasks_path / f"make_{i+1}.json"
        with task_file.open("w") as f:
            json.dump(task, f, indent=4)
        logger.info(f"Generated task file: {task_file}")


if __name__ == "__main__":
    main()
