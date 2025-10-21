import json
from pathlib import Path

from loguru import logger

MOLECULES = [
    "[CH2:1]=[C:2]([c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1)[C@@H:9]1[CH2:10][CH2:11][CH2:12][C@@:13]1([OH:14])[C:15]([F:16])([F:17])[F:18]",  # Carbony-En Reaction
    "COC([C@]12CC=CC[C@H]1C(C2)=O)=O",  # Diels-Alder
    "COC(C1(C=C1[Si](C)(C)C)/C=C/C2=CC=CC=C2)=O",  # Cycloaddition
    "C=C(C1=CC=C(OC)C=C1)C2=CC=C(OC)C=C2",  # Peterson olefination
    "C[C@H]1[C@](CC(C(CO)=C)=O)(C(C(C)=C)=CC1=O)[H]",  # Nozaki Hiyama Kishi Reaction + Deprotection of OTBS into OH
    # Difficult ones
    "O[C@H]1C[C@@H](O[C@@H]([C@@H]1C)/C=C(CO)/C)C/C=C/C=C/C(O)=O",  # Still-Gennari + Horner-Wadsworth-Emmons
    "O[C@H]1C2=CC3=CC=CC=C3O[C@H]2CCC1",  # Baylis-Hillman Reaction + cyclic stereocontrol
    "COc1cccc(NC(=O)c2nnn(Cc3ccc(CN4CC(F)C4)cc3)c2N)c1",
    "Cc1ccc(NS(=O)(=O)c2ccc(/C=C/C(=O)Nc3ccccc3N)cc2)cc1",
    "O=S(NC1=CC(N2CCN(C(OC(C)(C)C)=O)CC2)=C3C(CCC4(CCC4)O3)=C1)(C5=C(F)C=CC=C5)=O",
]
TEMPLATES = [
    ["1914396"],
    ["1914397"],
    ["1914398"],
    ["1679759"],
    ["36006", "1914399", "1914400"],
    ["29648", "1914401", "1914414", "149046", "1914403"],
    ["1914404", "337284"],
    ["1914405", "1914406", "1914407"],
    ["324328", "1914408", "733"],
    ["20810", "2895", "1914409", "1914410", "1914411", "74060"],
]
HINTS = [
    "In a carbonyl-ene reaction, an alkene reacts with an allylic hydrogen and a carbonyl group to form a new carbon-carbon bondm. This reaction can take place between parts of the same molecule. The reaction typically requires a Lewis acid catalyst and proceeds via a concerted mechanism. The stereochemistry of the product is influenced by the geometry of the alkene and the carbonyl group. The result of the reaction is the formation of a new carbon-carbon bond and a new stereocenter at the site of the allylic hydrogen, in which one of the substituents is going to be a hydroxyl group.",
    "A Dies-Alder reaction is a [4+2] cycloaddition between a conjugated diene and a dienophile, resulting in the formation of a six-membered ring. The reaction is stereospecific, meaning that the stereochemistry of the reactants is preserved in the product. The reaction typically proceeds via a concerted mechanism, where the pi electrons of the diene and dienophile interact to form new sigma bonds. The reaction can be catalyzed by heat or Lewis acids, and the regioselectivity of the product can be influenced by substituents on the diene and dienophile.",
    "In a cycloaddition reaction, two unsaturated molecules (or parts of the same molecule) combine to form a cyclic product. The reaction typically involves the formation of new sigma bonds between the reacting species, resulting in the creation of a ring structure.",
    "The Peterson Reaction allows the preparation of alkenes from alpha-silylcarbanions. An intermediate beta-hydroxy silane is formed, following an elimination step to yield the alkene.",
    "tert-Butyldimethylsilyl (TBS or TBDMS) is a common protecting group for alcohols in organic synthesis. The TBS group is stable under a variety of reaction conditions, including acidic and basic environments, making it useful for multi-step syntheses. It can be removed (deprotected) using fluoride sources such as tetrabutylammonium fluoride (TBAF) or by acidic hydrolysis, regenerating the free alcohol.\n\nThe nozaki-hiyama-kishi reaction is a nickel/chromium-catalyzed coupling reaction between an aldehyde and an allyl, vinyl, or aryl halide to form a new carbon-carbon bond. The reaction typically proceeds via the formation of an organochromium intermediate, which then reacts with the aldehyde to form the desired product. The reaction is stereoselective, meaning that the stereochemistry of the product is influenced by the geometry of the starting materials and the reaction conditions. The resulting product is an alcohol (that can be easily oxidized into a ketone) with a new carbon-carbon bond formed between the aldehyde and the halide.",
    "tert-Butyldimethylsilyl (TBS or TBDMS) is a common protecting group for alcohols in organic synthesis. It is introduced using reagents such as tert-butyldimethylsilyl chloride (TBDMSCl) or tert-butyldimethylsilyl trifluoromethanesulfonate (TBDMSOTf) in the presence of a base like imidazole or pyridine. The TBS group is stable under a variety of reaction conditions, including acidic and basic environments, making it useful for multi-step syntheses. It can be removed (deprotected) using fluoride sources such as tetrabutylammonium fluoride (TBAF) or by acidic hydrolysis, regenerating the free alcohol.\n\nIn the Horner-Wadsworth-Emmons reaction, the reaction of aldehydes or ketones with stabilized phosphorus ylides (phosphonate carbanions) leads to olefins with excellent E-selectivity.\n\nDIBAL-H is a reducing agent used in organic synthesis, particularly for the selective reduction of esters and nitriles to aldehydes. It is a complex of diisobutylaluminum hydride and is typically used in low temperatures to minimize side reactions.\n\nTMSOTf can be used to protect alcohols by converting them into their corresponding trimethylsilyl (TMS) ethers. The reaction typically involves the treatment of the alcohol with TMSOTf in the presence of a base, such as triethylamine or pyridine.\n\nThe Still-Gennari reaction is a modification of the Horner-Wadsworth-Emmons (HWE) reaction that allows for the selective formation of (Z)-alkenes from aldehydes and phosphonate esters. The reaction typically involves the use of a phosphonate ester with electron-withdrawing groups, such as bis(trifluoroethyl) or bis(2,2,2-trifluoroethyl) groups, which helps to stabilize the carbanion intermediate formed during the reaction.",
    "The Baylis-Hillman reaction is a carbon-carbon bond-forming reaction between an activated alkene (such as an acrylate or vinyl ketone) and an aldehyde or ketone, catalyzed by a nucleophilic catalyst (often a tertiary amine or phosphine). The reaction proceeds via the formation of a zwitterionic intermediate, which then undergoes nucleophilic attack on the carbonyl compound to form the desired product.\n\nKetones can be reduced to secondary alcohols using various reducing agents. Common reagents for this transformation include sodium borohydride (NaBH4) and lithium aluminum hydride (LiAlH4).",
    "Friedel-Crafts alkylation is a type of electrophilic aromatic substitution reaction that introduces an alkyl group onto an aromatic ring. The reaction typically involves the use of an alkyl halide and a Lewis acid catalyst, such as aluminum chloride (AlCl3) or ferric chloride (FeCl3). The Lewis acid activates the alkyl halide, generating a carbocation or a related electrophilic species that can then attack the aromatic ring, forming a new carbon-carbon bond.\n\nA substitution reaction can be used to replace a leaving group (such as a hydroxyl group) with an halogens (like chlorine, bromine, or iodine). This can be achieved using reagents such as hydrogen halides (HF), thionyl chloride (SOCl2), phosphorus tribromide (PBr3), or phosphorus triiodide (PI3).",
    "A carboxylic acid can be converted to an amide by reaction with an amine in the presence of a coupling agent. Common coupling agents include carbodiimides (like DCC or EDC) or uronium salts (like HATU or TBTU). The reaction typically proceeds via the formation of an activated ester intermediate, which then reacts with the amine to form the desired amide bond.\n\nThe Doebner Modification is a reaction in which an aromatic aldehyde reacts with malonic acid (HOOC-CH2-COOH) under base (e.g., piperidine/pyridine or ammonium acetate) to give an alpha,beta-unsaturated carboxylic acid (the new alkene) with decarboxylation.",
    "An aromatic amine attacks the electrophilic sulfur of phenylsulfonyl chloride, then the intermediate collapses to expel Cl⁻ and base deprotonates the N-H to give the sulfonamide\n\nThe reduction of nitro groups to amines can be achieved using various reducing agents. Common methods include catalytic hydrogenation (using hydrogen gas and a metal catalyst such as palladium on carbon, Pt, or Raney nickel) or chemical reduction using reagents like iron and hydrochloric acid (Fe/HCl), tin and hydrochloric acid (Sn/HCl), or zinc and ammonium chloride (Zn/NH4Cl). These methods effectively convert the nitro group (-NO2) to an amino group (-NH2) while preserving other functional groups in the molecule.\n\nDuring an Nucleophilic Aromatic Substitution (S_NAr) reaction, a nucleophile replaces a leaving group (such as a halogen) on an aromatic ring. This reaction typically occurs when the aromatic ring is activated by electron-withdrawing groups (like nitro groups) that stabilize the negative charge in the intermediate Meisenheimer complex. The nucleophile attacks the carbon atom bearing the leaving group, leading to the formation of a new carbon-nucleophile bond and the departure of the leaving group.\n\nThe Electrophilic Bromination of an aromatic ring involves the substitution of a hydrogen atom on the aromatic ring with a bromine atom. This reaction is typically carried out using bromine (Br2) in the presence of a Lewis acid catalyst, such as iron(III) bromide (FeBr3) or aluminum bromide (AlBr3). The Lewis acid activates the bromine molecule, generating a more electrophilic species that can attack the aromatic ring, leading to the formation of a new carbon-bromine bond.\n\nFor fully reducing a ketone into an alkane, the Wolff-Kishner reduction can be employed. This reaction involves the conversion of the ketone into a hydrazone intermediate using hydrazine (NH2NH2) under basic conditions, followed by heating with a strong base (like KOH) to eliminate nitrogen gas and form the corresponding alkane. During the entire process, the O-H bond of the ketone is broken, and obviously a 'ketones' functional group as well.\n\nIn the cyclization of chalcones to flavanones, an intramolecular Michael addition occurs where the nucleophilic enolate of the ketone attacks the electrophilic beta-carbon of the alpha,beta-unsaturated carbonyl system. This reaction is typically catalyzed by a base or acid and results in the formation of a new carbon-carbon bond, leading to the cyclic flavanone structure. During the reaction, a C-C and a C-O bond are formed, while a C-O bonds is broken. About the functional groups, a ether oxygens are formed.",
]

TARGETS = [
    ["C/C(C1=CC=CC=C1)=C/CCCC(C(F)(F)F)=O"],
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
    tasks_path = Path(__file__).parent / "tasks"
    for i, molecule in enumerate(MOLECULES):
        task = {
            "id": f"make_{i+1}_lvl2",
            "name": f"make_{i+1}_lvl2",
            "keywords": ["chemistry", "synthesis", "retrosynthesis"],
            "metrics": ["binary"],
            "input": {
                "prompt": f"Propose a retrosynthesis route to synthesize the molecule with SMILES {molecule}. The route must have at least {len(TEMPLATES[i])} reactions. The leaves on the retrosynthesis tree should be commercially available chemicals.\n\nHere are some hints to help you (the hints are enumerated in the order that they should be applied): {HINTS[i]}",
                "input_from_task": False,
                "input_for_task": False,
            },
            "output": [
                {
                    "type": "integer",
                    "target": TARGETS[i],
                    "threshold": None,
                }
            ],
            "scoring_fn": "check_reactants",
            "submission_format": """Submit a JSON object representing the retrosynthesis route. It must follow the same JSON format as the next example: `{\n  "type": "mol",\n  "smiles": "CO",\n  "children": [\n    {\n      "type": "reaction",\n      "template_id": "template_x",\n      "children": [\n        {\n          "type": "mol",\n          "smiles": "BrC"\n        },\n        {\n          "type": "mol",\n          "smiles": "[OH-]"\n        }\n      ]\n    }\n  ]\n}`.""",
            "tools": [
                "search_template_catalog_by_criteria",
                "get_template",
                "get_available_functional_groups",
                "apply_template",
                "verify_step",
                "verify_route",
                "search_catalog_by_smiles",
                "is_buyable",
                "suggest_protecting_groups",
                "deprotect_molecule",
                "detect_functional_groups",
                "detect_protection_groups",
            ],
        }
        task_file = tasks_path / f"make_{i+1}.json"
        with task_file.open("w") as f:
            json.dump(task, f, indent=4)
        logger.info(f"Generated task file: {task_file}")


if __name__ == "__main__":
    main()
