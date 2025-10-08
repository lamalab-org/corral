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

TARGETS = [
    [
        "[CH3:1]/[C:2]([c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1)=[CH:9]/[CH2:10][CH2:11][CH2:12][C:13](=[O:14])[C:15]([F:16])([F:17])[F:18]"
    ],
    [
        "[CH2:6]=[CH:7][CH:8]=[CH2:9]",
        "[CH3:1][O:2][C:3](=[O:4])[C:5]1=[CH:10][C:11](=[O:12])[CH2:13]1",
    ],
    [
        "O=[C:5]([C:3]([O:2][CH3:1])=[O:4])[CH2:6][CH2:7][c:8]1[cH:9][cH:10][cH:11][cH:12][cH:13]1",
        "[CH:14]#[C:15][Si:16]([CH3:17])([CH3:18])[CH3:19]",
    ],
    [
        "O=[C:2]([c:3]1[cH:4][cH:5][c:6]([O:7][CH3:8])[cH:9][cH:10]1)[c:11]1[cH:12][cH:13][c:14]([O:15][CH3:16])[cH:17][cH:18]1",
        "[Li][CH2:1][Si](C)(C)C",
    ],
    [
        "I[C:14](=[CH2:15])[CH2:16][O:17][Si:18]([CH3:19])([CH3:20])[C:21]([CH3:22])([CH3:23])[CH3:24]",
        "[CH2:1]=[C:2]([CH3:3])[C:4]1=[CH:5][C@H:6]([OH:7])[C@@H:8]([CH3:9])[C@@H:10]1[CH2:11][CH:12]=[O:13]",
    ],
    [
        "C[CH:16](P(=O)(OCC(F)(F)F)OCC(F)(F)F)[C:15]#[C:17][O:19][CH3:20]",
        "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH2:6][C@H:7]1[CH2:8][C@H:9]([OH:10])[C@@H:11]([CH3:12])[C@@H:13]([CH:14]=[O:18])[O:21]1",
    ],
    [
        "O=[CH:14][c:13]1[c:8]([OH:7])[cH:9][cH:10][cH:11][cH:12]1",
        "[O:1]=[C:2]1[CH2:3][CH2:4][CH2:5][CH:6]=[CH:15]1",
    ],
    ["O[CH:2]1[CH2:3][N:4]([CH2:5][c:6]2[cH:7][cH:8][cH:9][cH:10][cH:11]2)[CH2:12]1"],
    [
        "Cl[S:7](=[O:8])(=[O:9])[c:10]1[cH:11][cH:12][c:13]([CH:14]=[O:15])[cH:16][cH:17]1",
        "[CH3:1][c:2]1[cH:3][cH:4][c:5]([NH2:6])[cH:18][cH:19]1",
    ],
    [
        "O=[C:4]1[CH2:5][CH2:6][CH2:7]1",
        "[O:1]=[C:2]([CH3:3])[c:17]1[c:9]([OH:8])[cH:10][cH:11][c:12]([N+:13](=[O:14])[O-:15])[cH:16]1",
    ],
]


def main():
    tasks_path = Path(__file__).parent.parent / "environments" / "level_1" / "tasks"
    for i, molecule in enumerate(MOLECULES):
        task = {
            "id": f"make_{i+1}_lvl1",
            "name": f"synthesize_{molecule}",
            "keywords": ["chemistry", "synthesis", "retrosynthesis"],
            "metrics": ["binary"],
            "input": {
                "prompt": f"Propose a retrosynthesis route to synthesize the molecule with SMILES {molecule}. The route must have at least one reaction.  You should use the template/s {PRIZES[i]} in this order.",
                "input_from_task": False,
                "input_for_task": False,
            },
            "output": [
                {
                    "type": "integer",
                    "target": [TARGETS[i]],
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
