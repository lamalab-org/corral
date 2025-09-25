import json
from pathlib import Path
from typing import Any

from retrosynthesis.types import FunctionalGroup
from retrosynthesis.utils import (
    _is_buyable,
    apply_template_forward,
    apply_template_retro,
    search_catalog,
    species_match,
)

from corral.backend.tool import Tool, tool
from corral.utils.modal import remote_call

# @tool
# def template_catalog(
#     resulting_group: str | None, reaction_type: str | None
# ) -> dict[str, Any]:
#     """Returns a template catalog."""
#     return resulting_group, reaction_type


@tool
def apply_template(molecule_smiles: str, template_id: str) -> list[str]:
    """Applies a reaction template to a molecule."""
    return apply_template_retro(molecule_smiles, template_id)


@tool
def verify_step(molecule_smiles: str, template: str, precursors: list[str]) -> bool:
    """Verifies if a retrosynthetic step is valid."""
    reactants = ".".join(precursors)
    real_product = apply_template_forward(reactants, template)
    return species_match([molecule_smiles], [real_product]) if real_product else False


@tool
def verify_route(route: str) -> tuple[bool, str]:
    """
    Verifies that a route string follows the hierarchical chemical synthesis schema.

    Expected schema:
    - Root level: molecule with "type": "mol", "smiles": str, optional "children"
    - Children can be reactions or molecules
    - Reactions must have "type": "reaction", "template_id": str, "children": list
    - Molecules must have "type": "mol", "smiles": str, optional "children"

    Args:
        route (str): JSON string representing the synthesis route

    Returns:
        tuple[bool, str]: (is_valid, error_message)
            - is_valid: True if the route follows the schema, False otherwise
            - error_message: Empty string if valid, error description if invalid
    """

    def validate_molecule(mol: dict[str, Any], path: str = "root") -> tuple[bool, str]:
        """Validate a molecule node"""
        # Check required fields
        if not isinstance(mol, dict):
            return False, f"Node at {path} is not a dictionary"

        if mol.get("type") != "mol":
            actual_type = mol.get("type", "missing")
            return False, f"Molecule at {path} has type '{actual_type}', expected 'mol'"

        if "smiles" not in mol:
            return False, f"Molecule at {path} is missing required 'smiles' field"

        if not isinstance(mol["smiles"], str):
            return False, f"Molecule at {path} has non-string 'smiles' field"

        # SMILES should not be empty
        if not mol["smiles"].strip():
            return False, f"Molecule at {path} has empty SMILES string"

        # Check optional children
        if "children" in mol:
            if not isinstance(mol["children"], list):
                return False, f"Molecule at {path} has non-list 'children' field"
            # Children should contain exactly one reaction for synthesis routes
            if len(mol["children"]) != 1:
                return (
                    False,
                    f"Molecule at {path} has {len(mol['children'])} children, expected exactly 1 reaction",
                )
            is_valid, error = validate_reaction(
                mol["children"][0], f"{path}.children[0]"
            )
            if not is_valid:
                return False, error

        return True, ""

    def validate_reaction(
        reaction: dict[str, Any], path: str = "reaction"
    ) -> tuple[bool, str]:
        """Validate a reaction node"""
        if not isinstance(reaction, dict):
            return False, f"Reaction at {path} is not a dictionary"

        if reaction.get("type") != "reaction":
            actual_type = reaction.get("type", "missing")
            return (
                False,
                f"Reaction at {path} has type '{actual_type}', expected 'reaction'",
            )

        if "template_id" not in reaction:
            return False, f"Reaction at {path} is missing required 'template_id' field"

        if not isinstance(reaction["template_id"], str):
            return False, f"Reaction at {path} has non-string 'template_id' field"

        if "children" not in reaction:
            return False, f"Reaction at {path} is missing required 'children' field"

        if not isinstance(reaction["children"], list):
            return False, f"Reaction at {path} has non-list 'children' field"

        # Reaction must have at least 2 children (reactants)
        if len(reaction["children"]) < 2:
            return (
                False,
                f"Reaction at {path} has {len(reaction['children'])} children, expected at least 2 reactants",
            )

        # All children must be valid molecules
        for i, child in enumerate(reaction["children"]):
            is_valid, error = validate_molecule(child, f"{path}.children[{i}]")
            if not is_valid:
                return False, error

        return True, ""

    try:
        # Parse JSON string
        data = json.loads(route)

        # Root must be a molecule
        is_valid, error = validate_molecule(data, "root")
        if not is_valid:
            return False, error

        return True, "The route is valid"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {e!s}"
    except Exception as e:
        return False, f"Unexpected error: {e!s}"


@tool
def search_catalog_by_cas(cas: str) -> list[dict[str, Any]] | str:
    """Searches a catalog for available precursors. Returns a list of chemical info dicts or a not-found message."""
    chemicals = search_catalog(cas)
    if len(chemicals) > 1:
        return chemicals[:1]
    return chemicals if chemicals else "No results found"


@tool
def is_buyable(molecule_smiles: str) -> bool:
    """Checks if a molecule is available for purchase."""
    return _is_buyable(molecule_smiles)


@tool
def suggest_protecting_groups(functional_group: FunctionalGroup) -> list[str]:
    """Suggests protecting groups for a given functional group."""
    _path = Path(__file__).parent / "protecting_groups.json"
    with _path.open("r") as f:
        protecting_groups_map = json.load(f)

    return [
        pg
        for pg in protecting_groups_map
        if pg.get("functional_group") == functional_group
    ]


@tool
def smiles_to_cas(molecule_smiles: str) -> str:
    """Converts a SMILES string to a CAS number."""
    return remote_call(function_name="return_cas_number", env_name="chemenv")(
        compound=molecule_smiles
    )


@tool
def cas_to_smiles(cas_number: str) -> str:
    """Converts a CAS number to a SMILES string."""
    return remote_call(function_name="get_isomeric_smiles_pubchem", env_name="chemenv")(
        compound=cas_number
    )


def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    return {
        # "template_catalog": template_catalog,
        "apply_template": apply_template,
        "verify_step": verify_step,
        "verify_route": verify_route,
        "search_catalog": search_catalog_by_cas,
        "is_buyable": is_buyable,
        "suggest_protecting_groups": suggest_protecting_groups,
        "smiles_to_cas": smiles_to_cas,
        "cas_to_smiles": cas_to_smiles,
    }
