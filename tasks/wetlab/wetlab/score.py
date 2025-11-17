from tools import CATIONS, ANIONS

ALL_IONS = CATIONS + ANIONS

CARBONATES = ["CO3-2", "HCO3-"]
SULFATES = ["SO4-2", "HSO4-"]
PHOSPHATES = ["PO4-3", "HPO4-2", "H2PO4-"]
OXALATES = ["C2O4-2", "HC2O4-", "Ox-2", "HOx-"]
CHROMATES = ["CrO4-2", "HCrO4-"]
SULFIDES = ["S-2", "HS-"]


def _normalize_ion(ion: str) -> str:
    if ion not in ALL_IONS:
        raise ValueError(f"Undefined Ion: {ion}")
    elif ion in CATIONS:
        return ion
    elif ion in CARBONATES:
        return "CO3-2"
    elif ion in SULFATES:
        return "SO4-2"
    elif ion in PHOSPHATES:
        return "PO4-3"
    elif ion in OXALATES:
        return "Ox-2"
    elif ion in CHROMATES:
        return "CrO4-2"
    elif ion in SULFIDES:
        return "S-2"
    else:
        return ion
        

def score_ion_list(prediction: str, ground_truth: str, binarize=True) -> float:

    gt_set = set([_normalize_ion(x.strip()) for x in ground_truth.split(",")])
    
    try:
        pred_set = set([_normalize_ion(x.strip()) for x in prediction.split(",")])
    except ValueError:
        return 0.0
        
    iou = len(gt_set.intersection(pred_set))/len(gt_set.union(pred_set))

    if binarize:
        return 1.0 if (iou == 1) else 0.0
    else:
        return iou
