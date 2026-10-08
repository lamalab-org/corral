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
    # make_9: https://patents.google.com/patent/AU2023303060A1/en
    "Cc1nc2c(-c3ccc(Cl)cc3F)nc(N3CCOC(c4cnn(C5CC5)c4)C3)nc2c(=O)n1C",
    # make_10: https://patents.google.com/patent/WO2026086892A1/en
    "CC(F)(F)c1nc(Nc2ccccc2)c2c(OC(/C=C/S(C)(=O)=O)C3CC3)cccc2n1",
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
    ["279", "178068", "401532", "91372", "127119", "1914400", "485553", "71001", "913", "32251", "140", "177", "809", "194171", "294717", "44599"],
    ["1914402", "1745", "89471", "4324", "152", "2270", "13161", "408", "479", "1914401", "8515"],
]
HINTS = [
    "In a carbonyl-ene reaction, an alkene reacts with an allylic hydrogen and a carbonyl group to form a new carbon-carbon bondm. This reaction can take place between parts of the same molecule. The reaction typically requires a Lewis acid catalyst and proceeds via a concerted mechanism. The stereochemistry of the product is influenced by the geometry of the alkene and the carbonyl group. The result of the reaction is the formation of a new carbon-carbon bond and a new stereocenter at the site of the allylic hydrogen, in which one of the substituents is going to be a hydroxyl group. Additionally, the reaction involves the change in the order of a C=C bond to a C-C bond, and a C=O bond to a C-O bond. 'ketones' and 'C-C double bonds' are gone, while a 'aliphatic hydroxyl' and a 'five-membered rings' are formed. A Carbon-Carbon bond is formed.",
    "A Dies-Alder reaction is a [4+2] cycloaddition between a conjugated diene and a dienophile, resulting generally in the formation of a six-membered ring. The reaction is stereospecific, meaning that the stereochemistry of the reactants is preserved in the product. The reaction typically proceeds via a concerted mechanism, where the pi electrons of the diene and dienophile interact to form new sigma bonds. The reaction can be catalyzed by heat or Lewis acids, and the regioselectivity of the product can be influenced by substituents on the diene and dienophile. As a result of the reaction, a new carbon-carbon bond is formed, and the order changes from C=C to C-C, and from C=C to C-C as well.",
    "In a cycloaddition reaction, two unsaturated molecules (or parts of the same molecule) combine to form a cyclic product. The reaction typically involves the formation of new sigma bonds between the reacting species, resulting in the creation of a ring structure. During the reaction a carbon-carbon bond is formed, while a carbon-oxygen is broken. As a result of these transformations, a 'ketones' and 'C-C triple bonds' functional groups are lost.",
    "The Peterson Reaction allows the preparation of alkenes from alpha-silylcarbanions. An intermediate beta-hydroxy silane is formed, following an elimination step to yield the alkene. Thus, a new carbon-carbon double bond is formed, while a carbon-silicon bond and a carbon-oxygen bond are broken. As a result, a 'C-C double bonds' functional group is formed, while 'ketones' and 'trimethylsilyl' groups are lost.",
    "tert-Butyldimethylsilyl (TBS or TBDMS) is a common protecting group for alcohols in organic synthesis. It is introduced using reagents such as tert-butyldimethylsilyl chloride (TBDMSCl) or tert-butyldimethylsilyl trifluoromethanesulfonate (TBDMSOTf) in the presence of a base like imidazole or pyridine. The TBS group is stable under a variety of reaction conditions, including acidic and basic environments, making it useful for multi-step syntheses. It can be removed (deprotected) using fluoride sources such as tetrabutylammonium fluoride (TBAF) or by acidic hydrolysis, regenerating the free alcohol. As a result of this reaction, a bond between oxygen and silicon is broken, and a 'aliphatic hydroxyls' functional group is formed, and a 't-butyldimethylsilyl' gone.\n\nIn the Horner-Wadsworth-Emmons reaction, the reaction of aldehydes or ketones with stabilized phosphorus ylides (phosphonate carbanions) leads to olefins with excellent E-selectivity. The overall of the reaction is a carbon-carbon bond formed, while a carbon-phosphorus bond and a carbon-oxygen are broken. As a result, an 'aliphatic carboxylic acid', and  'C-C double bonds' groups are formed.\n\nDIBAL-H is a reducing agent used in organic synthesis, particularly for the selective reduction of esters and nitriles to aldehydes. It is a complex of diisobutylaluminum hydride and is typically used in low temperatures to minimize side reactions. Overall a carbon-oxygen bond is broken, and a another carbon-oxygen bond order changes from 2 to 1, forming a 'aliphatic hydroxyls' functional group.\n\nTMSOTf can be used to protect alcohols by converting them into their corresponding trimethylsilyl (TMS) ethers. The reaction typically involves the treatment of the alcohol with TMSOTf in the presence of a base, such as triethylamine or pyridine. During the reaction a oxygen-silicon bond is broken and another one (O-Si) is formed. No functional groups are formed in the transformation, but the most notable broken is 'aliphatic hydroxyls'.\n\nThe Still-Gennari reaction is a modification of the Horner-Wadsworth-Emmons (HWE) reaction that allows for the selective formation of (Z)-alkenes from aldehydes and phosphonate esters. The reaction typically involves the use of a phosphonate ester with electron-withdrawing groups, such as bis(trifluoroethyl) or bis(2,2,2-trifluoroethyl) groups, which helps to stabilize the carbanion intermediate formed during the reaction. Thus, a carbon-carbon double bond, and a carbon-oxygen are formed, while a carbon-phosphorus bond, a carbon-carbon bond and a carbon-oxygen bond are broken. Additionally, the triple bond of an alkyne is changed to a single bond. As a result 'C-C double bonds' and  'carboxylic esters' functional groups are formed.",
    "The Sn2 reaction is a bimolecular nucleophilic substitution reaction where a nucleophile attacks an electrophilic carbon atom, leading to the displacement of a leaving group. The reaction proceeds via a single transition state, resulting in the inversion of stereochemistry at the carbon center. Common nucleophiles used in Sn2 reactions include hydroxide ions (OH-), alkoxide ions (RO-), and cyanide ions (CN-). The reaction typically occurs in polar aprotic solvents, which help to stabilize the transition state and enhance the nucleophilicity of the attacking species. As a result of the reaction, a new carbon-nucleophile bond is formed while a carbon-leaving group bond is broken, e.g. C-N formed and C-Cl gone breaking an 'halide' functional group.\n\nThe Friedel-Crafts acylation is a type of electrophilic aromatic substitution reaction that introduces an acyl group onto an aromatic ring. The reaction typically involves the use of an acyl chloride (RCOCl) in the presence of a Lewis acid catalyst, such as aluminum chloride (AlCl3) or ferric chloride (FeCl3). The Lewis acid activates the acylating agent, generating a more electrophilic species that can attack the aromatic ring, leading to the formation of a new carbon-carbon bond and breaking a carbon-chlorine bond. An 'halide' functional group is gone during the transformation.\n\nA substitution reaction can be used to replace a leaving group (such as a hydroxyl group) with an halogens (like chlorine, bromine, or iodine). This can be achieved using reagents such as hydrogen halides (HF), thionyl chloride (SOCl2), phosphorus tribromide (PBr3), or phosphorus triiodide (PI3). For the example with hidroxyl as leaving group, a Carbon-Oxygen bond is broken, and a Carbon-Halogen bond is formed, breaking an 'aliphatic hydroxyl' functional group.",
    "A carboxylic acid can be converted to an amide by reaction with an amine in the presence of a coupling agent. Common coupling agents include carbodiimides (like DCC or EDC) or uronium salts (like HATU or TBTU). The reaction typically proceeds via the formation of an activated ester intermediate, which then reacts with the amine to form the desired amide bond. Therefore, a carbon-nitrogen bond is formed while a carbon-oxygen is broken as the result of breaking the 'aliphatic carboxylic acid'.\n\nThe Doebner Modification is a reaction in which an aromatic aldehyde reacts with malonic acid (HOOC-CH2-COOH) under base (e.g., piperidine/pyridine or ammonium acetate) to give an alpha,beta-unsaturated carboxylic acid (the new alkene) with decarboxylation. During this reaction, a carbon-carbon double bond is formed, while a carbon-carbon bond and a carbon-oxygen bond are broken. Additionally, a 'C-C double bond' functional group is formed.\n\nSulfonamide formation occurs when an amine reacts with a sulfonyl chloride. During the reaction, a bond between sulfur and nitrogen is formed, forming a 'sulfonamides' functional group.",
    "An aromatic amine attacks the electrophilic sulfur of phenylsulfonyl chloride, then the intermediate collapses to expel Cl⁻ and base deprotonates the N-H to give the sulfonamide. 'sulfonamides' functional group is formed, (N-S bond) while a (S-Cl bond) is broken.\n\nThe reduction of nitro groups to amines can be achieved using various reducing agents. Common methods include catalytic hydrogenation (using hydrogen gas and a metal catalyst such as palladium on carbon, Pt, or Raney nickel) or chemical reduction using reagents like iron and hydrochloric acid (Fe/HCl), tin and hydrochloric acid (Sn/HCl), or zinc and ammonium chloride (Zn/NH4Cl). These methods effectively convert the 'nitro' group (-NO2) to an 'primary amines' group (-NH2) while preserving other functional groups in the molecule, and a bond between O and N being broken during the process.\n\nDuring an Nucleophilic Aromatic Substitution (S_NAr) reaction, a nucleophile replaces a leaving group (such as a halogen) on an aromatic ring. This reaction typically occurs when the aromatic ring is activated by electron-withdrawing groups (like nitro groups) that stabilize the negative charge in the intermediate Meisenheimer complex. The nucleophile attacks the carbon atom bearing the leaving group, leading to the formation of a new carbon-nucleophile bond and the departure of the leaving group. For example, when the nucleophile is an amine, a carbon-nitrogen bond is formed while a carbon-halogen bond is broken, breaking an 'halide' functional group.\n\nThe Electrophilic Bromination of an aromatic ring involves the substitution of a hydrogen atom on the aromatic ring with a bromine atom. This reaction is typically carried out using bromine (Br2) in the presence of a Lewis acid catalyst, such as iron(III) bromide (FeBr3) or aluminum bromide (AlBr3). The Lewis acid activates the bromine molecule, generating a more electrophilic species that can attack the aromatic ring, leading to the formation of a new carbon-bromine bond, and an 'halide' functional group is formed.\n\nFor fully reducing a ketone into an alkane, the Wolff-Kishner reduction can be employed. This reaction involves the conversion of the ketone into a hydrazone intermediate using hydrazine (NH2NH2) under basic conditions, followed by heating with a strong base (like KOH) to eliminate nitrogen gas and form the corresponding alkane. During the entire process, the O-H bond of the ketone is broken, and obviously a 'ketones' functional group as well.\n\nIn the cyclization of chalcones to flavanones, an intramolecular Michael addition occurs where the nucleophilic enolate of the ketone attacks the electrophilic beta-carbon of the alpha,beta-unsaturated carbonyl system. This reaction is typically catalyzed by a base or acid and results in the formation of a new carbon-carbon bond, leading to the cyclic flavanone structure. During the reaction, a C-C and a C-O bond are formed, while a C-O bonds is broken. About the functional groups, a 'ether oxygens' are formed, and 'aromatic hydroxyls' and 'ketones' are broken.",
    (
        "1. The free morpholine secondary amine displaces the remaining chloride on the electron-poor fused pyrimidopyrimidinone by nucleophilic aromatic substitution. A heteroaryl C-N bond replaces C-Cl and joins the core and side-chain branches.\n\n"
        "2. A Suzuki coupling joins the heterocyclic chloride and 4-chloro-2-fluorophenylboronic acid. A C-C bond is formed while the heterocyclic C-Cl and aryl C-B bonds are broken. The other heterocyclic chloride is retained for the final amination.\n\n"
        "3. Phosphorus oxychloride converts two of the heterocyclic oxo positions to chlorides. Two C-O bonds are replaced by C-Cl bonds, leaving the N-methyl lactam carbonyl intact.\n\n"
        "4. Methylamine opens the oxazine ring and cyclocondensation replaces its ring oxygen with N-methyl nitrogen, giving the fused pyrimidopyrimidine trione. Two C-N bonds replace the ring C-O bonds.\n\n"
        "5. Acetic anhydride acylates the amino group of 5-aminoorotic acid and promotes cyclodehydration to the fused oxazine trione. The new ring contains the acetyl-derived carbon and methyl group; C-N and C-O bonds are formed while the carboxylic acid is consumed.\n\n"
        "6. Acid removes the Boc carbamate from the morpholine nitrogen, releasing the free secondary amine. The N-C(=O) bond is cleaved and the Boc group is lost, while the N-cyclopropyl pyrazole is retained.\n\n"
        "7. Copper-mediated N-cyclopropylation couples cyclopropylboronic acid with the pyrazole N-H. A pyrazole N-C bond replaces the cyclopropyl C-B bond; the morpholine nitrogen stays Boc protected.\n\n"
        "8. Di-tert-butyl dicarbonate protects the morpholine secondary amine as a Boc carbamate. An N-C(=O) bond forms while the pyrazole N-H remains available for the next transformation.\n\n"
        "9. Hydrogenolysis removes the remaining benzyl protecting group from the morpholine nitrogen. Cleavage of its N-CH2Ph bond gives the secondary amine. The patent performs both debenzylations together; this route represents them as two template applications.\n\n"
        "10. Hydrogenolysis removes the benzyl protecting group from the pyrazole nitrogen. The pyrazole N-CH2Ph bond is cleaved to restore N-H; the morpholine nitrogen remains benzyl protected in this explicit intermediate.\n\n"
        "11. Acid-promoted cyclodehydration of the amino diol closes the six-membered morpholine ring. An intramolecular C-O bond forms with loss of water, converting the two alcohol groups into a cyclic ether.\n\n"
        "12. Borohydride reduction converts the ketone into a secondary alcohol. The carbonyl C=O bond becomes C-O, giving the amino diol needed for morpholine formation.\n\n"
        "13. 2-(Benzylamino)ethanol displaces bromide from the alpha-bromoketone by nucleophilic substitution. A C-N bond replaces C-Br to give a tertiary amino ketone bearing a pendant alcohol.\n\n"
        "14. N-Bromosuccinimide converts the ethoxyvinyl group to an alpha-bromoketone under hydrolytic conditions. The enol ether becomes a ketone, a C-Br bond forms at the terminal carbon, and the ethoxy group is removed.\n\n"
        "15. A Stille coupling of the bromopyrazole with tributyl(1-ethoxyvinyl)tin installs an ethoxyvinyl substituent. A C-C bond forms while the pyrazole C-Br and vinyl C-Sn bonds break.\n\n"
        "16. Benzyl bromide protects the pyrazole nitrogen by N-alkylation. A pyrazole N-CH2 bond forms and the benzylic C-Br bond breaks; the bromine attached to the pyrazole carbon remains available for coupling."
    ),
    (
        "1. A Horner-Wadsworth-Emmons olefination couples the aldehyde with diethyl [(methylsulfonyl)methyl]phosphonate. A C=C bond joins the aldehyde carbon to the phosphonate methylene carbon as C-O and C-P bonds are broken, giving the E-alkenyl sulfone.\n\n"
        "2. Dess-Martin oxidation converts the primary alcohol to an aldehyde. The C-O single bond becomes C=O without changing the side-chain carbon skeleton.\n\n"
        "3. Lithium aluminium hydride reduces the ethyl ester to a primary alcohol. The ester alkoxy C-O bond is cleaved and its carbonyl C=O becomes C-OH, while the quinazoline and aryl ether remain intact.\n\n"
        "4. The phenoxide displaces bromide from ethyl 2-bromo-2-cyclopropylacetate. A C-O bond replaces C-Br to install the cyclopropyl-substituted ester side chain as an aryl ether.\n\n"
        "5. Hydrolysis of the phenolic acetate cleaves its O-C(=O) bond and regenerates the phenol. The patent combines aniline substitution and deacetylation in one operation; this route represents them as two template applications.\n\n"
        "6. Aniline displaces chloride at C4 of the quinazoline by nucleophilic aromatic substitution. A heteroaryl C-N bond replaces C-Cl, producing the N-phenylamino substituent while retaining the acetate protecting group.\n\n"
        "7. Thionyl chloride converts the quinazolin-4-one carbonyl to the corresponding 4-chloroquinazoline. The heterocyclic C-O bond is replaced by C-Cl, while the phenolic acetate remains intact.\n\n"
        "8. Acetic anhydride protects the phenol as its acetate ester. An O-C(=O) bond forms, replacing the phenolic O-H with an acetyl group before chlorination of the heterocycle.\n\n"
        "9. Boron tribromide cleaves the aryl methyl ether to reveal the phenol. The O-CH3 bond breaks while the aryl C-O bond is retained, forming an aromatic hydroxyl group.\n\n"
        "10. Cyclocondensation of the anthranilamide with ethyl 2,2-difluoropropionate forms the quinazolin-4-one ring. The ester carbonyl carbon becomes the ring carbon bearing the 1,1-difluoroethyl group. Two C-N bonds form with loss of ethanol and water.\n\n"
        "11. Activation of 2-amino-6-methoxybenzoic acid followed by reaction with ammonia converts the carboxylic acid to a primary benzamide. A carbonyl C-N bond replaces the acid C-OH bond; the aromatic amino and methoxy groups are retained."
    ),
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
        "C[CH:16](P(=O)(OCC(F)(F)F)OCC(F)(F)F)[C:15]#[C:17][O:19][CH3:20]",
        "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH2:6][C@H:7]1[CH2:8][C@H:9]([OH:10])[C@@H:11]([CH3:12])[C@@H:13]([CH:14]=[O:18])[O:21]1",
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
    [
        "Nc1c(C(=O)O)[nH]c(=O)[nH]c1=O",
        "CC(=O)OC(C)=O",
        "CN",
        "O=P(Cl)(Cl)Cl",
        "OB(O)c1ccc(Cl)cc1F",
        "Brc1cn[nH]c1",
        "BrCc1ccccc1",
        "C=C(OCC)[Sn](CCCC)(CCCC)CCCC",
        "O=C1CCC(=O)N1Br",
        "OCCNCc1ccccc1",
        "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",
        "OB(O)C1CC1",
    ],
    [
        "COc1cccc(N)c1C(=O)O",
        "N",
        "CCOC(=O)C(C)(F)F",
        "CC(=O)OC(C)=O",
        "O=S(Cl)Cl",
        "Nc1ccccc1",
        "CCOC(=O)C(Br)C1CC1",
        "CCOP(=O)(CS(C)(=O)=O)OCC",
    ],
]


def main(tasks_path: Path | None = None):
    tasks_path = tasks_path or Path(__file__).parent / "tasks_json"
    tasks_path.mkdir(parents=True, exist_ok=True)
    for i, molecule in enumerate(MOLECULES):
        task = [
            {
                "id": f"make_{i+1}_lvl2",
                "name": f"make_{i+1}_lvl2",
                "keywords": ["chemistry", "synthesis", "retrosynthesis"],
                "metrics": ["binary"],
                "input": {
                    "prompt": f"Propose a retrosynthesis route to synthesize the molecule with SMILES {molecule}. The route must have {len(TEMPLATES[i])} reactions.\n\nHere are some guidance to help you (the reactions are enumerated in the order that they must be applied): {HINTS[i]}. Note that each of the hint describes the forward reactions for each step (enumerated in the order that they should be applied), but you have to perform the retrosynthetic reactions.",
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
                "scoring_function": "check_reactants",
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
