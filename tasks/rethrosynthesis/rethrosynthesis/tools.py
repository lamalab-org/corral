from typing import Any

from corral.backend.tool import Tool, tool


@tool
def template_catalog(
    reacting_group: str | None, resulting_group: str | None, reaction_type: str | None
) -> dict[str, Any]:
    """Returns a template catalog."""
    return reacting_group, resulting_group, reaction_type


@tool
def apply_template(molecule_smiles: str, template: str) -> list[str]:
    """Applies a reaction template to a molecule."""
    return molecule_smiles, template


@tool
def verify_step(molecule_smiles: str, template: str, precursors: list[str]) -> bool:
    """Verifies if a retrosynthetic step is valid."""
    return molecule_smiles, template, precursors


@tool
def verify_route(target_smiles: str, reaction_tree: dict[str, Any]) -> bool:
    """Verifies if a retrosynthetic route is valid."""
    return target_smiles, reaction_tree


@tool
def search_catalog(molecule_smiles: str) -> list[str]:
    """Searches a catalog for available precursors."""
    return molecule_smiles


@tool
def is_buyable(molecule_smiles: str) -> bool:
    """Checks if a molecule is available for purchase."""
    return molecule_smiles


@tool
def molecule_properties(molecule_smiles: str) -> dict[str, Any]:
    """Returns properties of a molecule."""
    return molecule_smiles


def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    return {
        "template_catalog": template_catalog,
        "apply_template": apply_template,
        "verify_step": verify_step,
        "verify_route": verify_route,
        "search_catalog": search_catalog,
        "is_buyable": is_buyable,
        "molecule_properties": molecule_properties,
    }
