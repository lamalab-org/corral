from tools import CATIONS, ANIONS
from typing import Dict
import json

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
        

def none_checker(prediction: str, ground_truth: str) -> float:
    assert ground_truth.lower() == 'none', "none_checker used but ground_truth is not None!"
    return prediction.strip().lower() == 'none'

def score_ion_list(prediction: str, ground_truth: str, binarize=True) -> float:

    gt_set = set([_normalize_ion(x.strip()) for x in ground_truth.split(",")])
    try:
        pred_set = set([_normalize_ion(x.strip()) for x in prediction.split(",")])
    except ValueError:
        return 0.0
    
    iou = len(gt_set.intersection(pred_set)) / len(gt_set.union(pred_set))
    
    if binarize:
        return 1.0 if (iou == 1) else 0.0
    else:
        return iou


def score_salt(prediction: str, ground_truth: Dict, binarize=True) -> float:
    if prediction.startswith("`"):
        prediction = prediction.strip("`json")

    try:
        pred_dict = json.loads(prediction)
    except json.JSONDecodeError:
        return 0.0

    if set(pred_dict.keys()) != {'cation', 'anion'}:
        return 0.0
    
    true_cation = ground_truth['cation']
    true_anion = ground_truth['anion']
    pred_cation = pred_dict['cation']
    pred_anion = pred_dict['anion']

    cation_score = 1.0 if (pred_cation == true_cation) else 0.0

    if pred_anion == true_anion:
        anion_score = 1.0
    else:
        normal_true_anion = _normalize_ion(true_anion)
        try:
            normal_pred_anion = _normalize_ion(pred_anion)
        except ValueError:
            normal_pred_anion = ""

        anion_score = 0.5 if (normal_pred_anion == normal_true_anion) else 0.0
    
    final_score = (cation_score + anion_score) / 2

    if binarize:
        return 1.0 if (final_score == 1.0) else 0.0
    else:
        return final_score


    

