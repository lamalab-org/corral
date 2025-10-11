import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdDeprotect
from rdkit.Chem.rdDeprotect import Deprotect
from retrosynthesis.constants import FUNCTIONAL_GROUPS, PG
from retrosynthesis.types import FunctionalGroup
from retrosynthesis.utils import (
    _is_buyable,
    apply_template_forward,
    apply_template_retro,
    detect_functional_groups_in_molecule,
    get_molecule_summary,
    search_by_template,
    search_catalog,
    search_reactions_by_criteria,
    species_match,
)

from corral.backend.tool import Tool, tool
from corral.utils.modal import remote_call


# TODO: Include reaction type in the search_template_catalog tool
@tool
def search_template_catalog_by_criteria(
    molecule_smiles: str,
    functional_groups_broken: list[str] | None = None,
    functional_groups_formed: list[str] | None = None,
    bonds_formed: list[str] | None = None,
    bonds_broken: list[str] | None = None,
    bonds_order_changed: list[str] | None = None,
) -> list[dict[str, Any]]:
    """[BRIEF] Searches the retrosynthetic template database based on specified criteria. [/BRIEF]

    [DETAILED] This function allows users to search a retrosynthetic template database using various chemical criteria, including functional groups that are broken or formed, as well as specific bonds that are formed, broken, or have their order changed.
    This changes refer to the forward reaction, meaning that if you are looking for a retrosynthetic template that breaks the alcohol in the current molecule to form an alkene, you should specify "alcohol" in `functional_groups_formed` and "alkene" in `functional_groups_broken`.
    It returns a list of templates that match the given criteria, each represented as a dictionary containing relevant information, and ranked by Tanimoto similarity with respect the reference molecule. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to find retrosynthetic templates that involve specific functional group transformations.
    - When you want to explore templates based on bond changes in a target molecule.
    - When planning retrosynthetic routes and looking for applicable templates based on chemical features. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify the chemical features (functional groups, bonds) relevant to your retrosynthetic analysis. You can use the tool `get_available_functional_groups` to see the list of functional groups that can be used for searching. [/PREREQUISITE]
    2. [CURRENT] Use `search_template_catalog_by_criteria` to find templates that match your specified criteria. [/CURRENT]
    3. [FOLLOW_UP] Review the returned templates and select those that are most relevant to your synthesis planning. You can then apply these templates using the `apply_template` tool or validate retrosynthetic steps with `verify_step`. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a SMILES string representing the target molecule and optional criteria for functional groups and bonds.
    - It analyzes the molecule to identify its functional groups and bond structure.
    - It searches the retrosynthetic template database for templates that match the specified criteria.
    - It ranks the matching templates based on their Tanimoto similarity to the reference molecule.
    - It checks that all returned templates can be applied to the target molecule.
    - The search results are returned as a list of dictionaries, each containing details about a matching template. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `search_template_catalog_by_criteria("CCO", functional_groups_broken=["alcohol"], functional_groups_formed=["alkene"])`,
        `search_template_catalog_by_criteria("c1ccccc1O", bonds_broken=["C-O"], bonds_formed=["C-C"])`,
        `search_template_catalog_by_criteria("C1=CC=CC=C1", functional_groups_formed=["carboxylic_acid"])`,
        `search_template_catalog_by_criteria("C1=CC=CC=C1", bonds_order_changed=["C=C"])`,
        `search_template_catalog_by_criteria("C1=CC=CC=C1C(=O)O", functional_groups_broken=["carboxylic_acid"], bonds_broken=["C=O"])`,
    ]
    [/SYNTACTICAL]

    Args:
        molecule_smiles (str):
            [BRIEF] SMILES string of the target molecule. [/BRIEF]
            [DETAILED] SMILES string of the molecule to be analyzed for retrosynthetic template matching. The SMILES string must be valid and represent a real chemical structure. [/DETAILED]
            [SYNTACTICAL] Valid SMILES string [/SYNTACTICAL]
            [EXAMPLES] "CCO", "c1ccccc1O", "C1=CC=CC=C1" [/EXAMPLES]

        functional_groups_broken (list[str] | None):
            [BRIEF] List of functional groups that are broken in the forward reaction. None will result in not filtering with this criterion. [/BRIEF]
            [DETAILED] A list of functional groups (from a predefined set) that are expected to be broken during the forward reaction. This helps to filter templates that involve the cleavage of these groups. [/DETAILED]
            [SYNTACTICAL] List of valid functional group strings or None [/SYNTACTICAL]
            [EXAMPLES] ["alcohol", "amine"], None [/EXAMPLES]

        functional_groups_formed (list[str] | None):
            [BRIEF] List of functional groups that are formed in the forward reaction. None will result in not filtering with this criterion. [/BRIEF]
            [DETAILED] A list of functional groups (from a predefined set) that are expected to be formed during the forward reaction. This helps to filter templates that involve the creation of these groups. [/DETAILED]
            [SYNTACTICAL] List of valid functional group strings or None [/SYNTACTICAL]
            [EXAMPLES] ["alkene", "carboxylic_acid"], None [/EXAMPLES]

        bonds_formed (list[str] | None):
            [BRIEF] List of bonds that are formed in the forward reaction. None will result in not filtering with this criterion. [/BRIEF]
            [DETAILED] A list of bond types (e.g., "6-6", "6-8") that are expected to be formed during the forward reaction. This helps to filter templates that involve the formation of these bonds. [/DETAILED]
            [SYNTACTICAL] List of valid bond type strings or None [/SYNTACTICAL]
            [EXAMPLES] ["6-6", "6-8"], None [/EXAMPLES]

        bonds_broken (list[str] | None):
            [BRIEF] List of bonds that are broken in the forward reaction. None will result in not filtering with this criterion. [/BRIEF]
            [DETAILED] A list of bond types (e.g., "6-6", "6-8") that are expected to be broken during the forward reaction. This helps to filter templates that involve the cleavage of these bonds. [/DETAILED]
            [SYNTACTICAL] List of valid bond type strings or None [/SYNTACTICAL]
            [EXAMPLES] ["6-6", "6-8"], None [/EXAMPLES]

        bonds_order_changed (list[str] | None):
            [BRIEF] List of bonds whose order is changed in the forward reaction. None will result in not filtering with this criterion. [/BRIEF]
            [DETAILED] A list of bond types (e.g., "6-6", "6-8") whose order is expected to change during the forward reaction. This helps to filter templates that involve changes in bond order. [/DETAILED]
            [SYNTACTICAL] List of valid bond type strings or None [/SYNTACTICAL]
            [EXAMPLES] ['6-6 (1.0->2.0)'], None [/EXAMPLES]

    Returns:
        list[dict[str, Any]]:
            [BRIEF] List of dictionaries representing matching retrosynthetic templates. [/BRIEF]
            [DETAILED] Each dictionary in the returned list contains details about a retrosynthetic template that matches the specified criteria, including its SMARTS representation and other relevant information. If no templates match the criteria, an empty list is returned. [/DETAILED]
            [SYNTACTICAL] List of dictionaries or an empty list [/SYNTACTICAL]
            [EXAMPLES] [{"template_id": "123", "smarts": "..."}], [] [/EXAMPLES]

    [LIMITATIONS] Known limitations:
        - The function relies on the completeness and accuracy of the retrosynthetic template catalog. If the catalog is incomplete or contains errors, the search results may be affected.
        - The criteria provided must be specific enough to yield meaningful results; overly broad criteria may return too many templates, while overly narrow criteria may return none.
        - The function may not handle all edge cases in chemical structures, such as unusual bonding patterns or rare functional groups.
        - The accuracy of the search results is dependent on the quality of the underlying reaction templates and algorithms used in the retrosynthetic analysis.
    [/LIMITATIONS]
    """
    return search_reactions_by_criteria(
        functional_groups_broken=functional_groups_broken,
        functional_groups_formed=functional_groups_formed,
        bonds_formed=bonds_formed,
        bonds_broken=bonds_broken,
        bonds_order_changed=bonds_order_changed,
        reference_smiles=molecule_smiles,
        limit=10,
    )


@tool
def get_template(template_id: str) -> str:
    """[BRIEF] Retrieves a retrosynthetic template and other information by its ID. [/BRIEF]

    [DETAILED] This function takes a template ID as input and returns the corresponding retrosynthetic template in SMARTS format.
    If the template ID is not found, it raises a ValueError.
    It also retrieves the canonical SMARTS template (template for the forward reaction), and an example reaction for that example reaction.
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to retrieve a specific retrosynthetic template for analysis or application.
    - When you want to explore the details of a known retrosynthetic transformation. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify the template ID you want to retrieve, you can search the template catalog with `search_template_catalog`. [/PREREQUISITE]
    2. [CURRENT] Use `get_template` to obtain the SMARTS representation of the template. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved template with `apply_template` to explore possible precursors for target molecules or validate retrosynthetic steps with `verify_step`. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a string representing the template ID as input.
    - It looks up the template ID in a predefined database or dictionary of retrosynthetic templates.
    - If the template ID exists, it retrieves and returns the corresponding SMARTS string.
    - If the template ID does not exist, it raises a ValueError indicating that the template was not found. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_template("123")`,
        `get_template("45")`,
        `get_template("7892")`,
        `get_template("001999")`,
        `get_template("9999999")`,
    ]
    [/SYNTACTICAL]

    Args:
        template_id (str):
            [BRIEF] Identifier of the retrosynthetic template to retrieve. [/BRIEF]
            [DETAILED] The template ID corresponds to a specific retrosynthetic transformation that can be applied to molecules. The templates ids can be found in the template catalog. [/DETAILED]
            [SYNTACTICAL] Valid template ID string [/SYNTACTICAL]
            [EXAMPLES] "template_123", "template_456", "template_789" [/EXAMPLES]
    Returns:
        str: The SMARTS representation of the requested retrosynthetic template.
            [BRIEF] SMARTS string of the retrosynthetic template. [/BRIEF]
            [DETAILED] The SMARTS string defines the chemical transformation represented by the retrosynthetic template. It can be used in various cheminformatics applications to apply the transformation to target molecules. [/DETAILED]
            [SYNTACTICAL] Valid SMARTS string [/SYNTACTICAL]
            [EXAMPLES] "[C:1][O:2]>>[C:1][C:2]", "[C:1][C:2]>>[C:1][O:2]" [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] Raised when the template ID is not found. [/ERROR_WHEN]
            [ERROR_DETAILS] This occurs if the provided template ID does not correspond to any known retrosynthetic template in the database. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the template ID is correct and exists in the template catalog. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
        - The function relies on the availability of the specified template in the template catalog.
        - The SMARTS string must be valid and represent a real chemical transformation.
        - The function may not account for all possible molecular variations and edge cases.
    [/LIMITATIONS]
    """
    return str(search_by_template(template_id))


@tool
def get_available_functional_groups() -> str:
    """
    [BRIEF] Returns a list of available functional groups for querying the database. [/BRIEF]

    [DETAILED] This function provides a list of predefined functional groups that can be used to filter molecules in the database.
    These functional groups are based on common chemical motifs and can aid in the identification and selection of relevant compounds for synthesis or analysis. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to identify specific functional groups in a set of molecules.
    - When you want to filter molecules based on their functional group content.
    - When you are interested in exploring the chemical space around certain functional motifs. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Understand the functional groups relevant to your chemical analysis or synthesis planning. [/PREREQUISITE]
    2. [CURRENT] Use `get_available_functional_groups` to retrieve the list of functional groups. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved functional groups to filter or search for molecules in the database using other tools or functions. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function accesses a predefined list of functional groups stored in the retrosynthesis package.
    - It returns this list as a simple Python list of strings, each representing a functional group.
    - The functional groups are standardized and commonly used in cheminformatics for molecular characterization. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_available_functional_groups()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        list[str]:
            [BRIEF] List of available functional groups. [/BRIEF]
            [DETAILED] A list of strings, each representing a functional group that can be used for querying the database. These functional groups are based on common chemical motifs and are useful for filtering and identifying relevant compounds. [/DETAILED]
            [SYNTACTICAL] List of valid functional group strings [/SYNTACTICAL]
            [EXAMPLES] ["alcohol", "amine", "carboxylic_acid"] [/EXAMPLES]

    [LIMITATIONS] Known limitations:
        - The list of functional groups is predefined and may not cover all possible functional groups found in chemical compounds.
        - The function does not provide additional information about each functional group, such as its chemical properties or reactivity.
        - The functional groups are based on common motifs and may not account for all variations or derivatives of these groups.
    [/LIMITATIONS]
    """

    return str(FUNCTIONAL_GROUPS)


@tool
def apply_template(molecule_smiles: str, template_id: str) -> list[str]:
    """[BRIEF] Applies a retrosynthetic template to a molecule. [/BRIEF]

    [DETAILED] Given a molecule in SMILES format and a template ID, this function applies the retrosynthetic template to the molecule and returns a list of precursor SMILES strings.
    If the template cannot be applied, it returns an empty list.
    This function is useful for retrosynthetic analysis in computational chemistry and drug discovery. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have a target molecule and want to explore possible precursors using a specific retrosynthetic template.
    - When you need to generate a list of potential starting materials for a given molecule based on known reaction templates.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have the template of the reaction you want to apply, you can search the template catalog with `search_template_catalog`. [/PREREQUISITE]
    2. [CURRENT] Use `apply_template` to get possible precursors for your target molecule. [/CURRENT]
    3. [FOLLOW_UP] Validate the proposed precursors using `verify_step` to ensure the retrosynthetic step is chemically valid. Use `is_buyable` to check if the proposed precursors are commercially available. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a SMILES string representing the target molecule and a template ID as inputs.
    - It applies the retrosynthetic template associated with the given ID to the molecule using the `rxnutils` package.
    - If the template is successfully applied, it returns a list of SMILES strings representing the precursor molecules.
    - If the template cannot be applied (e.g., due to incompatibility with the molecule structure), it returns an empty list.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `apply_template("CCO", "template_123")`,
        `apply_template("c1ccccc1O", "template_456")`,
        `apply_template("C1=CC=CC=C1", "template_789")`,
        `apply_template("C1=CC=CC=C1", "template_zzz")`,
        `apply_template("C1=CC=CC=C1C(=O)O", "template_131415")`,
    ]
    [/SYNTACTICAL]

    Args:
        molecule_smiles (str):
            [BRIEF] SMILES string of the target molecule. [/BRIEF]
            [DETAILED] SMILES string of the molecule to which the template will be applied. Note that the SMILES string must be valid and represent a real chemical structure which can be processed by the retrosynthetic route used. [/DETAILED]
            [SYNTACTICAL] Valid SMILES string [/SYNTACTICAL]
            [EXAMPLES] "CCO", "c1ccccc1O", "C1=CC=CC=C1" [/EXAMPLES]

        template_id (str):
            [BRIEF] Identifier of the retrosynthetic template to apply. [/BRIEF]
            [DETAILED] The template ID corresponds to a specific retrosynthetic transformation that can be applied to the target molecule. The templates ids can be found in the template catalog. [/DETAILED]
            [SYNTACTICAL] Valid template ID string [/SYNTACTICAL]
            [EXAMPLES] "template_123", "template_456", "template_789" [/EXAMPLES]

    Returns:
        list[str]:
            [BRIEF] List of SMILES strings representing the precursor molecules. [/BRIEF]
            [DETAILED] Each SMILES string in the returned list corresponds to a precursor molecule that can be used to synthesize the target molecule using the specified retrosynthetic template. If the template cannot be applied, the list will be empty. [/DETAILED]
            [SYNTACTICAL] List of valid SMILES strings or an empty list [/SYNTACTICAL]
            [EXAMPLES] ["CCBr", "CO"], [] [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] Raised when the template cannot be found. [/ERROR_WHEN]
            [ERROR_DETAILS] This occurs if the provided template ID does not correspond to any known retrosynthetic template in the database. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the template ID is correct and exists in the template catalog. [/ERROR_RECOVERY]

        Exception:
            [ERROR_WHEN] Raised for any unexpected errors during template application. [/ERROR_WHEN]
            [ERROR_DETAILS] This will be most likely due to issues with the input SMILES string. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Check that the SMILES string being used is correct. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - The function relies on the availability and correctness of the retrosynthetic templates. If the template database is incomplete or contains errors, the results may be affected.
    - The function may not handle all edge cases in chemical structures, such as unusual bonding patterns or rare functional groups.
    - The accuracy of the retrosynthetic predictions is dependent on the quality of the underlying reaction templates and algorithms used in the `rxnutils` package.
    [/LIMITATIONS]
    """
    return apply_template_retro(molecule_smiles, template_id)


@tool
def verify_step(molecule_smiles: str, template_id: str, precursors: list[str]) -> bool:
    """
    [BRIEF] Verifies if a retrosynthetic step is valid. [/BRIEF]

    [DETAILED] This function checks whether applying a given retrosynthetic template to a set of precursor molecules results in the target molecule.
    It is used to validate retrosynthetic steps in a synthesis route. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have a proposed retrosynthetic step and want to confirm its validity.
    - When validating a synthesis route to ensure each step correctly leads to the intended product. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify the rethrosynthetic template to use and use `apply_template` to generate the precursors. [/PREREQUISITE]
    2. [CURRENT] Use `verify_step` to check if the precursors lead to the target molecule when the template is applied. [/CURRENT]
    3. [FOLLOW_UP] If the step is valid, proceed with the next steps in the synthesis route. If invalid, reconsider the choice of precursors or template. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a target molecule in SMILES format, a retrosynthetic template, and a list of precursor SMILES strings.
    - It applies the retrosynthetic template to the precursors to generate a product.
    - It then compares the generated product to the target molecule to determine if they match.
    - If the generated product matches the target molecule, the function returns True, indicating a valid retrosynthetic step. Otherwise, it returns False. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `verify_step("CCO", "template_123", ["CCBr", "CO"])`,
        `verify_step("c1ccccc1O", "template_456", ["c1ccccc1Br", "CO"])`,
        `verify_step("C1=CC=CC=C1", "template_789", ["C1=CC=CC=C1Br", "H2O"])`,
        `verify_step("C1=CC=CC=C1", "template_zzz", ["C1=CC=CC=C1Br", "H2O"])`,
        `verify_step("C1=CC=CC=C1C(=O)O", "template_131415", ["C1=CC=CC=C1C(=O)Cl", "H2O"])`,
    ]
    [/SYNTACTICAL]

    Args:
        molecule_smiles (str):
            [BRIEF] SMILES string of the target molecule. [/BRIEF]
            [DETAILED] The SMILES string representation of the target molecule. It must be the molecule that one already known in the retrosynthetic context. [/DETAILED]
            [SYNTACTICAL] Valid SMILES string [/SYNTACTICAL]
            [EXAMPLES] "CCO", "c1ccccc1O", "C1=CC=CC=C1" [/EXAMPLES]

        template_id (str):
            [BRIEF] Retrosynthetic template in SMARTS format. [/BRIEF]
            [DETAILED] The retrosynthetic template is a SMARTS string that defines the transformation to be applied to the precursors. It should be one of the templates from the dataset. [/DETAILED]
            [SYNTACTICAL] Valid SMARTS string [/SYNTACTICAL]
            [EXAMPLES] "[C:1][O:2]>>[C:1][C:2]", "[C:1][C:2]>>[C:1][O:2]" [/EXAMPLES]

        precursors (list[str]):
            [BRIEF] List of SMILES strings representing the precursor molecules. [/BRIEF]
            [DETAILED] A list of SMILES strings, each representing a precursor molecule that, when combined and transformed by the template, should yield the target molecule. [/DETAILED]
            [SYNTACTICAL] List of valid SMILES strings [/SYNTACTICAL]
            [EXAMPLES] ["CCBr", "CO"], ["c1ccccc1Br", "CO"] [/EXAMPLES]

    Returns:
        bool:
            [BRIEF] True if the retrosynthetic step is valid, False otherwise. [/BRIEF]
            [DETAILED] The function returns True if applying the retrosynthetic template to the provided precursors results in the target molecule. If the generated product does not match the target molecule, it returns False. [/DETAILED]
            [SYNTACTICAL] Boolean value (True or False) [/SYNTACTICAL]
            [EXAMPLES] True, False [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] Raised when the template is invalid or cannot be applied. [/ERROR_WHEN]
            [ERROR_DETAILS] This occurs if the provided template is not a valid SMARTS string or if it cannot be applied to the given precursors. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the template is correct and compatible with the precursors. [/ERROR_RECOVERY]
    """
    reactants = ".".join(precursors)
    real_products = apply_template_forward(reactants, template_id)
    return species_match([molecule_smiles], real_products) if real_products else False


@tool
def verify_route(route: str) -> tuple[bool, str]:
    """
    [BRIEF] Verifies if a synthesis route follows the expected schema. [/BRIEF]

    [DETAILED] This function checks whether a given synthesis route, represented as a JSON string, adheres to a predefined hierarchical schema.
    The schema defines the structure and required fields for molecules and reactions in the route. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have a synthesis route and want to ensure it is correctly formatted before further processing.
    - When you need to validate a route before submitting it to a retrosynthesis planning system. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Construct or obtain a synthesis route in JSON format. [/PREREQUISITE]
    2. [CURRENT] Use `verify_route` to validate the route structure. [/CURRENT]
    3. [FOLLOW_UP] If the route is valid, proceed with retrosynthesis planning, analysis or submit if the route is correct. You can check if the reactants are valid by checking if they are buyable using the tool `is_buyable`. If invalid, correct the structure based on the error message provided. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a JSON string representing a synthesis route as input.
    - It parses the JSON and checks that it conforms to a hierarchical schema where:
      - The root is a molecule with required fields "type" and "smiles".
      - Molecules can have children that are either reactions or other molecules.
      - Reactions must have required fields "type", "template_id", and "children".
      - Each reaction must have at least two children (reactants).
    - The function recursively validates each node in the hierarchy to ensure all required fields are present and correctly typed.
    - If the route adheres to the schema, it returns (True, "The route is valid"). If not, it returns (False, error_message) with a description of the first encountered error. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `verify_route('{"type": "mol", "smiles": "CCO", "children": [{"type": ...}]}')`,
        `verify_route('{"type": "mol", "smiles": "c1ccccc1O", "children": [{"type": ...}]}')`,
        `verify_route('{"type": "mol", "smiles": "C1=CC=CC=C1", "children": [{"type": ...}]}')`,
        `verify_route('{"type": "mol", "smiles": "C1=CC=CC=C1", "children": [{"type": ...}]}')`,
        `verify_route('{"type": "mol", "smiles": "C1=CC=CC=C1C(=O)O", "children": [{"type": ...}]}')`,
    ]
    [/SYNTACTICAL]

    Args:
        route (str):
            [BRIEF] JSON string representing the synthesis route. [/BRIEF]
            [DETAILED] A JSON-formatted string that encodes the hierarchical structure of a synthesis route, including molecules and reactions with their respective fields. [/DETAILED]
            [SYNTACTICAL] Valid JSON string [/SYNTACTICAL]
            [EXAMPLES] '{"type": "mol", "smiles": "CCO", "children": [{"type": ...}]}', '{"type": "mol", "smiles": "c1ccccc1O", "children": [{"type": ...}]}' [/EXAMPLES]

    Returns:
        tuple[bool, str]:
            [BRIEF] (is_valid, error_message) where is_valid is True if the route is valid, False otherwise. [/BRIEF]
            [DETAILED] A tuple where the first element is a boolean indicating whether the route is valid, and the second element is a string containing an error message if the route is invalid, or a success message if it is valid. [/DETAILED]
            [SYNTACTICAL] (Boolean, String) [/SYNTACTICAL]
            [EXAMPLES] (True, "The route is valid"), (False, "Invalid JSON format: ...") [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] Raised when the input JSON is invalid or cannot be parsed. [/ERROR_WHEN]
            [ERROR_DETAILS] This occurs if the provided string is not valid JSON format. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the input string is correctly formatted JSON. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - The function assumes that the input JSON string is well-formed and does not handle deeply nested structures beyond typical synthesis routes.
    - The validation focuses on structural correctness and does not verify the chemical validity of the molecules or reactions.
    - The function may not catch all possible schema violations, especially in complex or unconventional route representations.
    - Error messages may not always pinpoint the exact location of the error in deeply nested structures.
    [/LIMITATIONS]
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
def search_catalog_by_cas(cas: str, limit: int = 10) -> list[dict[str, Any]] | str:
    """
    [BRIEF] Searches a catalog for available precursors. [/BRIEF]

    [DETAILED] This function searches a chemical catalog using a CAS number to find available precursor chemicals.
    It returns a list of chemical information dictionaries if matches are found, or a message indicating no results were found. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have a CAS number and want to find corresponding chemicals in the catalog.
    - When validating the availability of a chemical precursor for synthesis planning. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Obtain the CAS number of the chemical you want to search for. You can obtain the CAS number for a chemical by using the tool `smiles_to_cas`. [/PREREQUISITE]
    2. [CURRENT] Use `search_catalog_by_cas` to look up the CAS number in the catalog. [/CURRENT]
    3. [FOLLOW_UP] If matches are found, review the chemical information for potential use in synthesis. If no matches are found, consider alternative chemicals or suppliers. You can also check if the molecule is buyable using the tool `is_buyable`. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a CAS number as input and queries a chemical catalog for matching entries.
    - It retrieves a list of chemicals that match the provided CAS number, each represented as a dictionary containing relevant chemical information.
    - If multiple matches are found, it returns only the first `limit` matches to avoid overwhelming the user.
    - If no matches are found, it returns a message indicating that no results were found. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `search_catalog_by_cas("50-00-0")`,
        `search_catalog_by_cas("64-17-5")`,
        `search_catalog_by_cas("67-56-1")`,
        `search_catalog_by_cas("000-00-0")`,
        `search_catalog_by_cas("999-99-9")`,
    ]
    [/SYNTACTICAL]

    Args:
        cas (str):
            [BRIEF] CAS number of the chemical to search for. [/BRIEF]
            [DETAILED] The CAS number is a unique numerical identifier assigned to every chemical substance described in the open scientific literature. It is used to provide a unique, unmistakable identifier for chemical substances. [/DETAILED]
            [SYNTACTICAL] Valid CAS number string [/SYNTACTICAL]
            [EXAMPLES] "50-00-0", "64-17-5", "67-56-1" [/EXAMPLES]

        limit (int):
            [BRIEF] Maximum number of matches to return. Defaults to 10. [/BRIEF]
            [DETAILED] This parameter sets an upper limit on the number of chemical matches to return from the catalog search. It helps to manage the amount of data returned and ensures that the user is not overwhelmed with too many results. The default value is 10. [/DETAILED]
            [SYNTACTICAL] Positive integer [/SYNTACTICAL]
            [EXAMPLES] 5, 10, 20 [/EXAMPLES]

    Returns:
        list[dict[str, Any]] | str:
            [BRIEF] List of chemical info dicts or a not-found message. [/BRIEF]
            [DETAILED] If matches are found, a list of dictionaries containing chemical information is returned. Each dictionary represents a chemical and includes details such as name, CAS number, and availability. If no matches are found, a message indicating no results were found is returned. [/DETAILED]
            [SYNTACTICAL] List of dictionaries or a string message [/SYNTACTICAL]
            [EXAMPLES] [{"name": "Formaldehyde", "cas": "50-00-0", ...}], "No results found" [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] Raised for any unexpected errors during the catalog search. [/ERROR_WHEN]
            [ERROR_DETAILS] This could be due to connectivity issues, invalid CAS number format, or server errors in the catalog service. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify the CAS number format if the error has to do with the CAS number. If the error comes from the catalog service, inform the user to try again later. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - The function limits the number of returned matches to prevent overwhelming the user with too many results.
    - The accuracy and completeness of the search results depend on the underlying chemical catalog being queried.
    - The function does not handle partial matches or synonyms; it strictly searches by the exact CAS number provided.
    - If the catalog service is down or unreachable, the function will not be able to return results.
    [/LIMITATIONS]
    """
    chemicals = search_catalog(cas)
    if len(chemicals) > limit:
        return chemicals[:limit]
    return chemicals if chemicals else "No results found"


@tool
def is_buyable(cas: str) -> bool:
    """
    [BRIEF] Checks if a molecule is commercially available. [/BRIEF]

    [DETAILED] This function determines whether a given molecule, represented by its CAS number, is commercially available for purchase.
    It returns True if the molecule can be bought, and False otherwise. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to verify the availability of a chemical precursor for synthesis planning.
    - When deciding whether to include a specific molecule in a synthesis route based on its commercial availability. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Obtain the CAS number of the molecule you want to check. You can obtain the CAS number for a chemical by using the tool `smiles_to_cas`. [/PREREQUISITE]
    2. [CURRENT] Use `is_buyable` to check if the molecule is commercially available. [/CURRENT]
    3. [FOLLOW_UP] If the molecule is buyable, consider it for inclusion in your synthesis route. If not, look for alternative molecules, precursors or routes. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a CAS number as input and queries a database or service that tracks the commercial availability of chemicals.
    - It checks if the molecule associated with the provided CAS number is listed as available for purchase.
    - If the molecule is found to be commercially available, the function returns True. If it is not available, it returns False. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `is_buyable("50-00-0")`,
        `is_buyable("64-17-5")`,
        `is_buyable("67-56-1")`,
        `is_buyable("000-00-0")`,
        `is_buyable("999-99-9")`,
    ]
    [/SYNTACTICAL]

    Args:
        cas (str):
            [BRIEF] CAS number of the molecule to check. [/BRIEF]
            [DETAILED] The CAS number is a unique numerical identifier assigned to every chemical substance described in the open scientific literature. It is used to provide a unique, unmistakable identifier for chemical substances. [/DETAILED]
            [SYNTACTICAL] Valid CAS number string [/SYNTACTICAL]
            [EXAMPLES] "50-00-0", "64-17-5", "67-56-1", "000-00-0", "999-99-9" [/EXAMPLES]

    Returns:
        bool:
            [BRIEF] True if the molecule is commercially available, False otherwise. [/BRIEF]
            [DETAILED] The function returns True if the molecule associated with the provided CAS number is listed as available for purchase. If the molecule is not available, it returns False. [/DETAILED]
            [SYNTACTICAL] Boolean value (True or False) [/SYNTACTICAL]
            [EXAMPLES] True, False [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] Raised for any unexpected errors during the availability check. [/ERROR_WHEN]
            [ERROR_DETAILS] This could be due to connectivity issues, invalid CAS number format, or server errors in the availability service. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify the CAS number format if the error has to do with the CAS number. If the error comes from the availability service, inform the user to try again later, and you should workaround by checking alternative routes. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - The function relies on the accuracy and completeness of the underlying database or service used to check commercial availability.
    - The availability status may change over time, so the function's results may not always reflect the most current market conditions.
    - The function does not provide information on pricing, suppliers, or quantities available for purchase.
    - If the availability service is down or unreachable, the function will not be able to return results.
    [/LIMITATIONS]
    """
    return _is_buyable(cas)


@tool
def suggest_protecting_groups(functional_group: FunctionalGroup) -> list[str]:
    """
    [BRIEF] Suggests protecting groups for a given functional group.[/BRIEF]

    [DETAILED] This function provides a list of suitable protecting groups for a specified functional group. Protecting groups are used in synthetic chemistry to temporarily mask reactive sites on molecules during multi-step synthesis processes. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When planning a synthetic route and needing to protect a functional group from unwanted reactions.
    - When selecting appropriate protecting groups based on the functional group present in the target molecule. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify the functional group in your target molecule. You can use the tool `detect_functional_groups` to identify functional groups in a molecule. [/PREREQUISITE]
    2. [CURRENT] Use `suggest_protecting_groups` to get a list of suitable protecting groups for the identified functional group. [/CURRENT]
    3. [FOLLOW_UP] Evaluate the suggested protecting groups and select the most appropriate one for your synthesis plan. Consider factors such as ease of installation and removal, stability under reaction conditions, and compatibility with other functional groups in the molecule. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a functional group as input, which is an enumeration representing common functional groups in organic chemistry (e.g., alcohol, amine, carboxylic acid).
    - It looks up a predefined mapping of functional groups to their corresponding protecting groups.
    - The function returns a list of protecting groups that are suitable for the specified functional group. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `suggest_protecting_groups('1,2-Aminoalcohol')`,
        `suggest_protecting_groups('Aldehyde, Ketone')`,
        `suggest_protecting_groups('Amine')`,
        `suggest_protecting_groups('Carboxylic acid')`,
        `suggest_protecting_groups('Phenol')`,
    ]
    [/SYNTACTICAL]

    Args:
        functional_group (FunctionalGroup):
            [BRIEF] The functional group for which to suggest protecting groups. It must be one of the following: '1,2-Aminoalcohol', '1,2-Diol', '1,3-Diol', 'Acetylene', 'Alcohol', 'Aldehyde, Ketone', 'Amide, Carbamate', 'Amine', 'Carboxylic acid', 'Indole', 'Phenol', 'Sulfonamide'. [/BRIEF]
            [DETAILED] An enumeration value representing a common functional group in organic chemistry. The functional group should be selected from the predefined list to ensure accurate suggestions. [/DETAILED]
            [SYNTACTICAL] One of the predefined FunctionalGroup enum values. [/SYNTACTICAL]
            [EXAMPLES] "Sulfonamide", "Amine", "Alcohol" [/EXAMPLES]

    Returns:
        list[str]:
            [BRIEF] List of protecting groups suitable for the given functional group and the reactions and conditions to protect and deprotect with them. [/BRIEF]
            [DETAILED] A list of strings, each representing a protecting group that can be used to protect the specified functional group during synthesis. The list may include common protecting groups as well as their abbreviations. [/DETAILED]
            [SYNTACTICAL] List of strings [/SYNTACTICAL]
            [EXAMPLES] '[{"reagents": "CuSO₄", "solvents": "Acetone", "temperature": "RT", "time": "36 h", "yield": "83%"},...,]', '[][/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] Raised when the provided functional group is not recognized. [/ERROR_WHEN]
            [ERROR_DETAILS] This occurs if the functional group does not match any of the predefined enumeration values. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the functional group is one of the recognized types. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - The function relies on a predefined mapping of functional groups to protecting groups, which may not cover all possible scenarios or the latest developments in synthetic chemistry.
    - The suitability of a protecting group may depend on specific reaction conditions and the overall synthetic route, which are not considered in this function.
    - The function does not provide information on the installation or removal procedures for the suggested protecting groups.
    - The function may not account for steric or electronic effects that could influence the choice of protecting group in complex molecules.
    [/LIMITATIONS]
    """
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
    """
    [BRIEF] Converts a SMILES string to a CAS number. [/BRIEF]

    [DETAILED] This function takes a SMILES (Simplified Molecular Input Line Entry System) string as input and converts it to the corresponding CAS (Chemical Abstracts Service) number. The CAS number is a unique numerical identifier assigned to every chemical substance described in the open scientific literature. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have a SMILES representation of a molecule and need to find its CAS number.[/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Obtain the SMILES string of the molecule you want to convert. [/PREREQUISITE]
    2. [CURRENT] Use `smiles_to_cas` to convert the SMILES string to a CAS number. [/CURRENT]
    3. [FOLLOW_UP] Use the obtained CAS number for further chemical information lookup, procurement, or documentation. You can also use the tool `search_catalog_by_cas` to find available precursors or `is_buyable` to check if the molecule is commercially available. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a SMILES string as input and queries a chemical database or service that maps SMILES strings to CAS numbers.
    - The function first validates the SMILES string to ensure it represents a valid molecular structure.
    - It retrieves the CAS number associated with the provided SMILES string.
    - If a matching CAS number is found, it is returned as a string. If no match is found, an appropriate message or exception may be raised. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `smiles_to_cas("CCO")`,
        `smiles_to_cas("c1ccccc1O")`,
        `smiles_to_cas("C1=CC=CC=C1")`,
        `smiles_to_cas("C1=CC=CC=C1C(=O)O")`,
        `smiles_to_cas("C1=CC=CC=C1C(=O)Cl")`,
    ]
    [/SYNTACTICAL]

    Args:
        molecule_smiles (str):
            [BRIEF] SMILES string of the molecule to convert. [/BRIEF]
            [DETAILED] A valid SMILES string of the molecule that you need the CAS number for. [/DETAILED]
            [SYNTACTICAL] Valid SMILES string [/SYNTACTICAL]
            [EXAMPLES] "CCO", "c1ccccc1O", "C1=CC=CC=C1" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] The corresponding CAS number as a string. [/BRIEF]
            [DETAILED] The function returns the CAS number associated with the provided SMILES string. The CAS number is a unique identifier for chemical substances and is widely used in chemical databases and literature. [/DETAILED]
            [SYNTACTICAL] String representing a CAS number [/SYNTACTICAL]
            [EXAMPLES] "50-00-0", "64-17-5", "67-56-1" [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] Raised for any unexpected errors during the conversion process. [/ERROR_WHEN]
            [ERROR_DETAILS] This could be due to connectivity issues, or server errors in the conversion service. [/ERROR_DETAILS]
            [ERROR_RECOVERY] If the error comes from the conversion service, being a server error or similar, inform the user to try again later, and you should go through a different route. [/ERROR_RECOVERY]

        ValueError:
            [ERROR_WHEN] Raised when the provided SMILES string is invalid. [/ERROR_WHEN]
            [ERROR_DETAILS] This occurs if the SMILES string cannot be parsed into a valid molecular structure. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the SMILES string is correctly formatted. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - The function relies on the accuracy and completeness of the underlying database or service used for the conversion.
    - Not all SMILES strings may have a corresponding CAS number, especially for novel or less common compounds.
    - The function does not handle stereochemistry or isotopic variations in the SMILES string.
    - If the conversion service is down or unreachable, the function will not be able to return results.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(molecule_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {molecule_smiles}")
    return remote_call(function_name="return_cas_number", env_name="chemenv")(
        compound=molecule_smiles
    )


@tool
def cas_to_smiles(cas_number: str) -> str:
    """
    [BRIEF] Converts a CAS number to an isomeric SMILES string. [/BRIEF]

    [DETAILED] This function takes a CAS (Chemical Abstracts Service) number as input and converts it to the corresponding isomeric SMILES (Simplified Molecular Input Line Entry System) string.
    The isomeric SMILES representation includes stereochemical information, making it more specific than standard SMILES. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have a CAS number and need to find the corresponding isomeric SMILES representation of the molecule. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Obtain the CAS number of the molecule you want to convert. You can obtain the CAS number for a chemical by using the tool `smiles_to_cas`. [/PREREQUISITE]
    2. [CURRENT] Use `cas_to_smiles` to convert the CAS number to an isomeric SMILES string. [/CURRENT]
    3. [FOLLOW_UP] Use the obtained isomeric SMILES string for further chemical analysis, modeling, or synthesis planning. You can also use the tool `detect_functional_groups` to identify functional groups in the molecule. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a CAS number as input and queries a chemical database or service that maps CAS numbers to isomeric SMILES strings.
    - It retrieves the isomeric SMILES representation associated with the provided CAS number.
    - If a matching isomeric SMILES string is found, it is returned as a string. If no match is found, an appropriate message or exception may be raised. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `cas_to_smiles("50-00-0")`,
        `cas_to_smiles("64-17-5")`,
        `cas_to_smiles("67-56-1")`,
        `cas_to_smiles("000-00-0")`,
        `cas_to_smiles("999-99-9")`,
    ]
    [/SYNTACTICAL]

    Args:
        cas_number (str):
            [BRIEF] CAS number of the molecule to convert. [/BRIEF]
            [DETAILED] Valid CAS number string that corresponds to a specific chemical substance that you want to convert to SMILES. [/DETAILED]
            [SYNTACTICAL] Valid CAS number string [/SYNTACTICAL]
            [EXAMPLES] "50-00-0", "64-17-5", "67-56-1" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] The corresponding isomeric SMILES string. [/BRIEF]
            [DETAILED] The function returns the isomeric SMILES representation associated with the provided CAS number. The isomeric SMILES includes stereochemical information, making it more specific than standard SMILES. [/DETAILED]
            [SYNTACTICAL] String representing an isomeric SMILES [/SYNTACTICAL]
            [EXAMPLES] "C(CO)O", "CCO", "C1=CC=CC=C1" [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] Raised for any unexpected errors during the conversion process. [/ERROR_WHEN]
            [ERROR_DETAILS] This could be due to connectivity issues, invalid CAS number format, or server errors in the conversion service. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify the CAS number format if the error has to do with the CAS number. If the error comes from the conversion service, inform the user to try again later. [/ERROR_RECOVERY]
    [/RAISES]
    """
    return remote_call(function_name="get_isomeric_smiles_pubchem", env_name="chemenv")(
        compound=cas_number
    )


@tool
def deprotect_molecule(molecule_smiles: str) -> str:
    """
    [BRIEF] Removes protecting groups from a molecule represented by a SMILES string. [/BRIEF]

    [DETAILED] This function takes a SMILES (Simplified Molecular Input Line Entry System) string representing a molecule with protecting groups and removes those protecting groups to yield the deprotected molecule.
    Protecting groups are commonly used in synthetic chemistry to temporarily mask reactive sites on molecules during multi-step synthesis processes. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have a molecule with protecting groups and need to obtain the deprotected version of the molecule.
    - When analyzing a synthetic route and needing to understand the structure of the molecule without protecting groups. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Obtain the SMILES string of the molecule with protecting groups. You can use the tool `detect_protection_groups` to identify protecting groups in a molecule. [/PREREQUISITE]
    2. [CURRENT] Use `deprotect_molecule` to remove the protecting groups and obtain the deprotected molecule. [/CURRENT]
    3. [FOLLOW_UP] Use the deprotected SMILES string for further chemical analysis, modeling, or synthesis planning. You can also use the tool `detect_functional_groups` to identify functional groups in the deprotected molecule. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a SMILES string as input and uses RDKit's rdDeprotect module to identify and remove common protecting groups.
    - It processes the molecular structure to eliminate the protecting groups while preserving the core structure of the molecule.
    - The resulting deprotected molecule is then converted  back to a SMILES string and returned. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `deprotect_molecule("CC(C)(OC(NCCOc1ccccc1)=O)C")`,
        `deprotect_molecule("CC(C)(OC(NCCOc1ccccc1)=O)C")`,
        `deprotect_molecule("CC(C)(OC(NCCOc1ccccc1)=O)C")`,
        `deprotect_molecule("CC(C)(OC(NCCOc1ccccc1)=O)C")`,
        `deprotect_molecule("CC(C)(OC(NCCOc1ccccc1)=O)C")`,
    ]
    [/SYNTACTICAL]

    Args:
        molecule_smiles (str):
            [BRIEF] SMILES string of the molecule with protecting groups. [/BRIEF]
            [DETAILED] Valid SMILES string of the molecule that contains the protective groups that you want to detect and remove. Ensure that is a valid SMILES, otherwise an exception will be raised. [/DETAILED]
            [SYNTACTICAL] Valid SMILES string [/SYNTACTICAL]
            [EXAMPLES] "CC(C)(OC(NCCOc1ccccc1)=O)C", "CC(C)(OC(NCCOc1ccccc1)=O)C" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] The SMILES string of the deprotected molecule. [/BRIEF]
            [DETAILED] The function returns the SMILES representation of the molecule after removing the protecting groups. This deprotected SMILES string represents the core structure of the molecule without any temporary modifications. [/DETAILED]
            [SYNTACTICAL] String representing a SMILES [/SYNTACTICAL]
            [EXAMPLES] "CCO", "CCO" [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] Raised when the provided SMILES string is invalid. [/ERROR_WHEN]
            [ERROR_DETAILS] This occurs if the SMILES string cannot be parsed into a valid molecular structure. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the SMILES string is correctly formatted. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - The function relies on RDKit's rdDeprotect module, which may not recognize all possible protecting groups.
    - The accuracy of the deprotection process depends on the quality of the input SMILES string.
    - The function may not handle complex molecules with multiple or overlapping protecting groups effectively.
    - The function does not provide information on the specific protecting groups that were removed or their positions in the original molecule.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(molecule_smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    return Deprotect(mol)


@tool
def detect_protection_groups(
    smiles: str,
) -> dict[str, Any]:
    """
    [BRIEF] Detects protecting groups in a molecule and their positions. [/BRIEF]

    [DETAILED] This function identifies protecting groups present in a molecule represented by a SMILES (Simplified Molecular Input Line Entry System) string.
    It returns a list of detected protecting groups along with their abbreviations, full names, classes, and the positions of the atoms involved in each protecting group. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to identify protecting groups in a molecule for synthesis planning or analysis.
    - When you want to understand the locations of protecting groups within a molecular structure. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Obtain the SMILES string of the molecule you want to analyze. [/PREREQUISITE]
    2. [CURRENT] Use `detect_protection_groups` to identify protecting groups and their positions in the molecule. [/CURRENT]
    3. [FOLLOW_UP] Use the information about protecting groups for synthesis planning, modification of the molecule, or further analysis. You can also use the tool `deprotect_molecule` to remove the identified protecting groups. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a SMILES string as input and converts it to a molecular structure using RDKit.
    - It uses RDKit's rdDeprotect module to identify common protecting groups based on predefined reaction SMARTS patterns.
    - For each detected protecting group, it records the abbreviation, full name, class, and the positions of the atoms involved.
    - Uses other custom SMARTS patterns to enhance the detection process, searching also for those in the molecule.
    - The results are compiled into a structured dictionary that includes the input SMILES, a mapped SMILES for reference, and a list of detected protecting groups with their details. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `detect_protection_groups("CC(C)(OC(NCCOc1ccccc1)=O)C")`,
        `detect_protection_groups("CCO")`,
        `detect_protection_groups("c1ccccc1O")`,
        `detect_protection_groups("C1=CC=CC=C1")`,
        `detect_protection_groups("C1=CC=CC=C1C(=O)O")`,
    ]
    [/SYNTACTICAL]

    Args:
        smiles (str):
            [BRIEF] SMILES string of the molecule to analyze. [/BRIEF]
            [DETAILED] Valid SMILES string of the molecule that contains the protective groups that you want to detect. [/DETAILED]
            [SYNTACTICAL] Valid SMILES string [/SYNTACTICAL]
            [EXAMPLES] "CC(C)(OC(NCCOc1ccccc1)=O)C", "CCO", "c1ccccc1O" [/EXAMPLES]

    Returns:
        dict[str, Any]:
            [BRIEF] Dictionary containing input SMILES, mapped SMILES, and detected protecting groups. [/BRIEF]
            [DETAILED] The function returns a dictionary with the following keys:
                - "input_smiles": The original SMILES string provided as input.
                - "mapped_smiles": A version of the SMILES string with atom-map numbers for reference.
                - "protecting_groups": A list of dictionaries, each representing a detected protecting group with details such as abbreviation, full name, class, and atom positions.
            This structured output provides comprehensive information about the protecting groups present in the molecule. [/DETAILED]
            [SYNTACTICAL] Dictionary with specific keys and values [/SYNTACTICAL]
            [EXAMPLES] {
                "input_smiles": "CC(C)(OC(NCCOc1ccccc1)=O)C",
                "mapped_smiles": "CC(C)(OC(NCCOc1ccccc1)=O)C",
                "protecting_groups": [
                    {"abbrev": "Boc", "name": "tert-Butyloxycarbonyl", "class": "amine", "positions": [(0, 1, 2)]},...
                ]
            } [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] Raised when the provided SMILES string is invalid. [/ERROR_WHEN]
            [ERROR_DETAILS] This occurs if the SMILES string cannot be parsed into a valid molecular structure. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the SMILES string is correctly formatted. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - The function relies on RDKit's rdDeprotect module, which may not recognize all possible protecting groups.
    - The function may not handle complex molecules with multiple or overlapping protecting groups effectively.
    - Custom protecting group SMARTS patterns must be valid; invalid SMARTS will be skipped without notification.
    - The function does not provide information on the specific reactions or conditions used to install or remove the protecting groups.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # Make a copy with atom-map numbers equal to (index+1) for a mapped SMILES view
    mol_mapped = Chem.Mol(mol)
    for i, a in enumerate(mol_mapped.GetAtoms()):
        a.SetAtomMapNum(i + 1)
    mapped_smiles = Chem.MolToSmiles(mol_mapped, isomericSmiles=True)

    results = []

    for dep in rdDeprotect.GetDeprotections():
        rxn = AllChem.ReactionFromSmarts(dep.reaction_smarts)
        reactant_templates = [
            rxn.GetReactantTemplate(i) for i in range(rxn.GetNumReactantTemplates())
        ]

        group_matches = set()
        for rt in reactant_templates:
            for match in mol.GetSubstructMatches(rt, uniquify=True):
                group_matches.add(tuple(match))

        if group_matches:
            results.append(
                {
                    "abbrev": dep.abbreviation,  # e.g., "Boc"
                    "name": dep.full_name,  # human-readable name
                    "class": dep.deprotection_class,  # e.g., "amine", "alcohol", ...
                    "positions": sorted(
                        group_matches
                    ),  # tuples of 0-based atom indices
                }
            )

    if PG is not None:
        for full_name, smarts in PG:
            try:
                query = Chem.MolFromSmarts(smarts)
            except Exception:
                query = None
            if not query:
                # Skip invalid SMARTS silently; you can log if desired
                continue

            matches = mol.GetSubstructMatches(query, uniquify=True)
            if matches:
                # Deduplicate tuples
                group_matches = sorted({tuple(m) for m in matches})
                results.append(
                    {
                        "abbrev": None,  # no abbreviation provided
                        "name": full_name,  # from the PG pair
                        "class": "custom",  # generic label
                        "positions": group_matches,  # tuples of 0-based atom indices
                        "smarts": smarts,  # echo back the SMARTS for traceability
                    }
                )

    return {
        "input_smiles": smiles,
        "mapped_smiles": mapped_smiles,
        "protecting_groups": results,
    }


@tool
def detect_functional_groups(smiles: str) -> str:
    """
    [BRIEF] Detects functional groups in a molecule represented by a SMILES string. [/BRIEF]

    [DETAILED] This function identifies functional groups present in a molecule represented by a SMILES (Simplified Molecular Input Line Entry System) string.
    It returns a summary of the detected functional groups, including their names, and positions of the atoms within the molecule.
    Functional groups are specific groups of atoms within molecules that have characteristic properties and reactivities. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to identify functional groups in a molecule for synthesis planning or analysis.
    - When you want to understand the reactive sites and properties of a molecule based on its functional groups. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Obtain the SMILES string of the molecule you want to analyze. [/PREREQUISITE]
    2. [CURRENT] Use `detect_functional_groups` to identify functional groups in the molecule. [/CURRENT]
    3. [FOLLOW_UP] Use the information about functional groups for synthesis planning, modification of the molecule, or further analysis. You can also use the tool `suggest_protecting_groups` to find suitable protecting groups for the identified functional groups. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The function takes a SMILES string as input and converts it to a molecular structure using RDKit.
    - It uses a predefined set of SMARTS patterns to identify common functional groups within the molecule.
    - For each detected functional group, it records the name and the positions of the atoms within the molecule.
    - The results are compiled into a summary string that lists the detected functional groups and the positions of the atoms. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `detect_functional_groups("CC(C)(OC(NCCOc1ccccc1)=O)C")`,
        `detect_functional_groups("CCO")`,
        `detect_functional_groups("c1ccccc1O")`,
        `detect_functional_groups("C1=CC=CC=C1")`,
        `detect_functional_groups("C1=CC=CC=C1C(=O)C")`,
    ]
    [/SYNTACTICAL]

    Args:
        smiles (str):
            [BRIEF] SMILES string of the molecule to analyze. [/BRIEF]
            [DETAILED] The SMILES string of the molecule that you want to detect the functional groups from. [/DETAILED]
            [SYNTACTICAL] Valid SMILES string [/SYNTACTICAL]
            [EXAMPLES] "CC(C)(OC(NCCOc1ccccc1)=O)C", "CCO", "c1ccccc1O" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] Summary of detected functional groups in the molecule. [/BRIEF]
            [DETAILED] The function returns a string summarizing the functional groups detected in the molecule. The summary includes the names of the functional groups and the positions of the atoms in the original molecule, providing insight into the reactive sites and properties of the molecule. [/DETAILED]
            [SYNTACTICAL] String summarizing functional groups [/SYNTACTICAL]
            [EXAMPLES] "Full mapped SMILES: C([NH:6][C:5]([O:4][C:2]([CH3:1])([CH3:3])[CH3:17])=[O:16])[CH2:8][O:9][c:10]1[cH:11][cH:12][cH:13][cH:14][cH:15]1

                        Groups found:
                            - phenyl group     pos=(10, 11, 12, 13, 14, 15)  frag=[c:10]1[cH:11][cH:12][cH:13][cH:14][cH:15]1
                            - carbamate groups pos=(2, 4, 5, 16, 6)  frag=[C:2][O:4][C:5]([NH:6])=[O:16]
                            ..." [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] Raised when the provided SMILES string is invalid. [/ERROR_WHEN]
            [ERROR_DETAILS] This occurs if the SMILES string cannot be parsed into a valid molecular structure. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the SMILES string is correctly formatted. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - The function relies on a predefined set of SMARTS patterns, which may not cover all possible functional groups.
    - The accuracy of the detection process depends on the quality of the input SMILES string.
    - The function may not handle complex molecules with overlapping or ambiguous functional groups effectively.
    - The function does not provide information on the specific locations of the functional groups within the molecule.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    res = detect_functional_groups_in_molecule(smiles)
    return get_molecule_summary(smiles, res)


def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    return {
        "search_template_catalog_by_criteria": search_template_catalog_by_criteria,
        "get_template": get_template,
        "get_available_functional_groups": get_available_functional_groups,
        "apply_template": apply_template,
        "verify_step": verify_step,
        "verify_route": verify_route,
        "search_catalog_by_cas": search_catalog_by_cas,
        "is_buyable": is_buyable,
        "suggest_protecting_groups": suggest_protecting_groups,
        "deprotect_molecule": deprotect_molecule,
        "detect_protection_groups": detect_protection_groups,
        "detect_functional_groups": detect_functional_groups,
        "smiles_to_cas": smiles_to_cas,
        "cas_to_smiles": cas_to_smiles,
    }
