import json
from pathlib import Path

from loguru import logger

MOLECULES = [
    "[CH2:1]=[C:2]([c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1)[C@@H:9]1[CH2:10][CH2:11][CH2:12][C@@:13]1([OH:14])[C:15]([F:16])([F:17])[F:18]",  # Carbony-En Reaction
    "COC([C@]12CC=CC[C@H]1C(C2)=O)=O",  # Diels-Alder
    "COC(C1(C=C1[Si](C)(C)C)/C=C/C2=CC=CC=C2)=O",  # Cycloaddition
    "C=C(C1=CC=C(OC)C=C1)C2=CC=C(OC)C=C2",  # Peterson olefination
    # Difficult ones
    "O[C@H]1C[C@@H](O[C@@H]([C@@H]1C)/C=C(CO)/C)C/C=C/C=C/C(O)=O",  # Still-Gennari + Horner-Wadsworth-Emmons
    "COc1cccc(NC(=O)c2nnn(Cc3ccc(CN4CC(F)C4)cc3)c2N)c1",
    "Cc1ccc(NS(=O)(=O)c2ccc(/C=C/C(=O)Nc3ccccc3N)cc2)cc1",
    "O=S(NC1=CC(N2CCN(C(OC(C)(C)C)=O)CC2)=C3C(CCC4(CCC4)O3)=C1)(C5=C(F)C=CC=C5)=O",
]
TEMPLATES = [
    ["1914384"],
    ["1914385"],
    ["1914386"],
    ["1679747"],
    ["29646", "1914389", "1914390", "149040", "1914391"],
    ["1914393", "1914394", "1914395"],
    ["324324", "1914396", "733"],
    ["20810", "2895", "1914397", "1914398", "1914399", "74054"],
]
PRIZES = [
    250.0,
    405.0,
    125.0,
    150.0,
    9999.0,
    9999.0,
    9999.0,
    9999.0,
]


def main():
    tasks_path = Path(__file__).parent / "tasks"
    for i, molecule in enumerate(MOLECULES):
        task = [
            {
                "id": f"make_{i+1}_lvl3",
                "name": f"make_{i+1}_lvl3",
                "keywords": ["chemistry", "synthesis", "retrosynthesis"],
                "metrics": ["binary"],
                "input": {
                    "prompt": f"Propose a retrosynthesis route to synthesize the molecule with SMILES {molecule}. The leaves on the retrosynthesis tree should be commercially available chemicals. A molecule's price is its frozen estimated cost in USD for 1 g, and the route cost is the sum across all leaf molecules. The route cost cannot exceed ${PRIZES[i]}. The final route should not surpass {len(TEMPLATES[i])} steps.",
                    "input_from_task": False,
                    "input_for_task": False,
                },
                "output": [
                    {
                        "type": "integer",
                        "target": {"prize": PRIZES[i], "max_steps": len(TEMPLATES[i])},
                        "threshold": None,
                    }
                ],
                "scoring_fn": "score_final",
                "submission_format": """Submit a JSON object representing the retrosynthesis route. It must follow the same JSON format as the next example: `{\n  "type": "mol",\n  "smiles": "CO",\n  "children": [\n    {\n      "type": "reaction",\n      "template_id": 1234,\n      "children": [\n        {\n          "type": "mol",\n          "smiles": "BrC"\n        },\n        {\n          "type": "mol",\n          "smiles": "[OH-]"\n        }\n      ]\n    }\n  ]\n}`.""",
                "tools": [
                    "check_smiles_reaction_template_matching",
                    "search_template_catalog_by_criteria",
                    "get_template",
                    "get_available_functional_groups",
                    "apply_template",
                    "verify_step",
                    "verify_route",
                    "search_catalog_by_smiles",
                    "is_buyable",
                    "deprotect_molecule",
                    "detect_functional_groups",
                    "detect_protection_groups",
                ],
            }
        ]
        task_file = tasks_path / f"make_{i+1}.json"
        with task_file.open("w") as f:
            json.dump(task, f, indent=4)
        logger.info(f"Generated task file: {task_file}")


if __name__ == "__main__":
    main()
