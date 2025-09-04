ALL_IONS = [
    # Common cations
    'Ag+',
    'Al3+',
    'Ba2+',
    'Ca2+',
    'Cd2+',
    'Co2+',
    'Cr3+',
    'Cs+',
    'Cu2+',
    'Fe2+',
    'Fe3+',
    'Ga3+',
    'Hg2+',
    'Hg2^2+',
    'K+',
    'Li+',
    'Mg2+',
    'Mn2+',
    'Na+',
    'NH4+',
    'Ni2+',
    'Pb2+',
    'Rb+',
    'Sn2+',
    'Sr2+',
    'Zn2+',

    # Common anions
    'Br-',
    'Cl-',
    'CN-',
    'CO3^2-',
    'CrO4^2-',
    'F-',
    'I-',
    'NO2-',
    'NO3-',
    'OH-',
    'PO4^3-',
    'S2-'
    'SO3^2-',
    'SO4^2-',
    'SCN-',
]

def score_ion_list(prediction: str, ground_truth: str) -> float:

    pred_list = prediction.split(", ")
    gt_list = ground_truth.split(", ")

    for ion in gt_list:
        if ion not in ALL_IONS:
            raise ValueError(f"Invalid ion in the ground truth: {ion}")
        
    if set(pred_list) == set(gt_list):
        return 1.0
    else:
        return 0.0
