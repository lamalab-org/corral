"""Type definitions for the retrosynthesis package."""

from typing import Literal

# Define the supported functional groups for protecting group suggestions
FunctionalGroup = Literal[
    "1,2-Aminoalcohol",
    "1,2-Diol",
    "1,3-Diol",
    "Acetylene",
    "Alcohol",
    "Aldehyde, Ketone",
    "Amide, Carbamate",
    "Amine",
    "Carboxylic acid",
    "Indole",
    "Phenol",
    "Sulfonamide",
]
