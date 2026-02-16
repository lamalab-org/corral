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

TARGETS = [
    [["C/C(C1=CC=CC=C1)=C/CCCC(C(F)(F)F)=O"]],
    [
        [
            "[CH2:6]=[CH:7][CH:8]=[CH2:9]",
            "[CH3:1][O:2][C:3](=[O:4])[C:5]1=[CH:10][C:11](=[O:12])[CH2:13]1",
        ]
    ],
    [
        [
            "O=[C:5]([C:3]([O:2][CH3:1])=[O:4])[CH2:6][CH2:7][c:8]1[cH:9][cH:10][cH:11][cH:12][cH:13]1",
            "[CH:14]#[C:15][Si:16]([CH3:17])([CH3:18])[CH3:19]",
        ]
    ],
    [
        [
            "O=[C:2]([c:3]1[cH:4][cH:5][c:6]([O:7][CH3:8])[cH:9][cH:10]1)[c:11]1[cH:12][cH:13][c:14]([O:15][CH3:16])[cH:17][cH:18]1",
            "[Li][CH2:1][Si](C)(C)C",
        ]
    ],
    [
        [
            "C[C@@H]1[C@H](C[C@@H](O[C@@H]1/C=C(CO)/C)C/C=C/C=C/C(O)=O)O[Si](C)(C(C)(C)C)C"
        ],
        [
            "O=P(C/C=C/C(OCC)=O)(OCC)OCC",
            "C[C@@H]1[C@@H](O[Si](C)(C)C(C)(C)C)C[C@H](CC=O)O[C@@H]1/C=C(C)\\CO",
        ],
        [
            "C[C@@H]1[C@@H](O[Si](C)(C)C(C)(C)C)C[C@H](CC(OCC)=O)O[C@@H]1/C=C(C)\\C(OC)=O"
        ],
        [
            "C[C@@H]1[C@@H](O)C[C@H](CC(OCC)=O)O[C@@H]1/C=C(C)\\C(OC)=O",
            "CC(C)(C)[Si](C)(OS(C(F)(F)F)(=O)=O)C",
        ],
        [
            "C[CH:16](P(=O)(OCC(F)(F)F)OCC(F)(F)F)[C:15]#[C:17][O:19][CH3:20]",
            "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH2:6][C@H:7]1[CH2:8][C@H:9]([OH:10])[C@@H:11]([CH3:12])[C@@H:13]([CH:14]=[O:18])[O:21]1",
        ],
    ],
    [
        ["FC(C1)CN1Cc2ccc(CCl)cc2", "COc1cc(NC(c2nn[nH]c2N)=O)ccc1"],
        [
            "FC1CN(C1)Cc2ccccc2",
            "ClCCl",
        ],
        [
            "O[CH:2]1[CH2:3][N:4]([CH2:5][c:6]2[cH:7][cH:8][cH:9][cH:10][cH:11]2)[CH2:12]1"
        ],
    ],
    [
        ["Nc1ccccc1N", "Cc1ccc(NS(=O)(c2ccc(/C=C/C(O)=O)cc2)=O)cc1"],
        ["O=C(CC(O)=O)O", "O=Cc1ccc(S(Nc2ccc(C)cc2)(=O)=O)cc1"],
        [
            "Cl[S:7](=[O:8])(=[O:9])[c:10]1[cH:11][cH:12][c:13]([CH:14]=[O:15])[cH:16][cH:17]1",
            "[CH3:1][c:2]1[cH:3][cH:4][c:5]([NH2:6])[cH:18][cH:19]1",
        ],
    ],
    [
        [
            "FC1=C(S(Cl)(=O)=O)C=CC=C1",
            "NC1=CC(N2CCN(C(OC(C)(C)C)=O)CC2)=C3C(CCC4(CCC4)O3)=C1",
        ],
        ["O=[N+](C1=CC(N2CCN(C(OC(C)(C)C)=O)CC2)=C3C(CCC4(CCC4)O3)=C1)[O-]"],
        ["CC(OC(N1CCNCC1)=O)(C)C", "BrC1=C2C(CCC3(CCC3)O2)=CC([N+]([O-])=O)=C1"],
        ["O=[N+](C1=CC=C2C(CCC3(CCC3)O2)=C1)[O-]"],
        ["O=C1C2=CC([N+]([O-])=O)=CC=C2OC3(CCC3)C1"],
        [
            "O=[C:4]1[CH2:5][CH2:6][CH2:7]1",
            "[O:1]=[C:2]([CH3:3])[c:17]1[c:9]([OH:8])[cH:10][cH:11][c:12]([N+:13](=[O:14])[O-:15])[cH:16]1",
        ],
    ],
]


def main():
    tasks_path = Path(__file__).parent / "subtasks"
    for i, molecule in enumerate(MOLECULES):
        tasks = []
        final_inputs = []
        for j, template in enumerate(TEMPLATES[i]):
            if j == 0:
                input_from_task = [f"make_{i+1}_lvl1-template_search-{j+1}"]
                initial_inputs = {"initial_molecule": molecule}
            else:
                input_from_task = [
                    f"make_{i+1}_lvl1-apply_template-{j}",
                    f"make_{i+1}_lvl1-template_search-{j+1}",
                ]
                initial_inputs = {}
            task = {
                "id": f"make_{i+1}_lvl1-template_search-{j+1}",
                "name": f"make_{i+1}_lvl1-template_search-{j+1}",
                "keywords": [
                    "chemistry",
                    "synthesis",
                    "retrosynthesis",
                    "template_search",
                ],
                "metrics": ["binary"],
                "input": {
                    "prompt": f"Can you return the `mapped_rxn` associated with the template {template}?",
                    "input_from_task": False,
                    "input_for_task": [
                        f"make_{i+1}_lvl1-apply_template-{j+1}",
                        f"make_{i+1}_lvl1-build_complete_route",
                    ],
                },
                "output": [
                    {
                        "type": "string",
                        "target": template,
                        "threshold": None,
                    }
                ],
                "scoring_fn": "check_template",
                "submission_format": "Return a dict with the `mapped_rxn` as a string and the `template_id` (the one used to search the database not the hash) as an integer in JSON format, e.g., {'mapped_rxn': 'Cc1ccccc1.Br>>Cc1ccccc1Br', 'template_id': 12345}.",
                "tools": [
                    "check_smiles_reaction_template_matching",
                    "search_template_catalog_by_criteria",
                    "get_template",
                    "get_available_functional_groups",
                    "deprotect_molecule",
                    "detect_functional_groups",
                    "detect_protection_groups",
                ],
            }
            final_inputs.append(f"make_{i+1}_lvl1-template_search-{j+1}")
            if j == len(TEMPLATES[i]) - 1:
                input_for_task = [
                    f"make_{i+1}_lvl1-build_complete_route",
                ]
            else:
                input_for_task = [
                    f"make_{i+1}_lvl1-apply_template-{j+2}",
                    f"make_{i+1}_lvl1-build_complete_route",
                ]
            tasks.append(task)
            previous_task = f"make_{i+1}_lvl1-template_search-{j+1}"
            task = {
                "id": f"make_{i+1}_lvl1-apply_template-{j+1}",
                "name": f"make_{i+1}_lvl1-apply_template-{j+1}",
                "keywords": [
                    "chemistry",
                    "synthesis",
                    "retrosynthesis",
                    "apply_template",
                ],
                "metrics": ["binary"],
                "input": {
                    "prompt": f"Can you return all possible precursors of applying the template as the result of the task {previous_task} to the molecule below?",
                    "input_from_task": input_from_task,
                    "input_for_task": [input_for_task],
                },
                "output": [
                    {
                        "type": "list",
                        "target": [TARGETS[i][j]],
                        "threshold": None,
                    }
                ],
                "initial_inputs": initial_inputs,
                "scoring_fn": "check_list_molecules",
                "submission_format": "Return a list of lists where each inner list contain each possible combination in which each item of the inner list corresponds to the SMILES of each of the precursors as a string. For example, `[['CCO', 'O'], ['CC=O', 'CCO']]`.",
                "tools": [
                    "get_template",
                    "apply_template",
                    "verify_step",
                ],
            }
            final_inputs.append(f"make_{i+1}_lvl1-apply_template-{j+1}")
            tasks.append(task)

        final_targets = []
        for k in TARGETS[i]:
            final_targets = k
        task = {
            "id": f"make_{i+1}_lvl1-build_complete_route",
            "name": f"make_{i+1}_lvl1-build_complete_route",
            "keywords": ["chemistry", "synthesis", "retrosynthesis", "route_building"],
            "metrics": ["binary"],
            "input": {
                "prompt": "Can you build and return the complete route of chemicals based on the inputs?",
                "input_from_task": final_inputs,
                "input_for_task": False,
            },
            "output": [
                {
                    "type": "list",
                    "target": final_targets,
                    "threshold": None,
                }
            ],
            "initial_inputs": {},
            "scoring_fn": "check_reactants",
            "submission_format": """Submit a JSON object representing the retrosynthesis route. It must follow the same JSON format as the next example: `{\n  \"type\": \"mol\",\n  \"smiles\": \"CO\",\n  \"children\": [\n    {\n      \"type\": \"reaction\",\n      \"template_id\": 1234,\n      \"children\": [\n        {\n          \"type\": \"mol\",\n          \"smiles\": \"BrC\"\n        },\n        {\n          \"type\": \"mol\",\n          \"smiles\": \"[OH-]\"\n        }\n      ]\n    }\n  ]\n}`.""",
            "tools": [
                "verify_step",
                "verify_route",
                "search_catalog_by_smiles",
                "is_buyable",
            ],
        }
        tasks.append(task)
        task_file = tasks_path / f"make_{i+1}.json"
        with task_file.open("w") as f:
            json.dump(tasks, f, indent=4)
        logger.info(f"Generated subtask file: {task_file}")


if __name__ == "__main__":
    main()
