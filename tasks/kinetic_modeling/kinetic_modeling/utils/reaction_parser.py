import re
from collections import defaultdict


def parse_reactions(reactions: list | str) -> tuple[list[dict], list]:
    """
    Parse a list of chemical reaction strings into structured dictionaries.

    This function processes reaction strings with the format:
    "[A] + 2 [B] > [C] + [D], k1 ; hv1, sigma1"

    Where:
    - Chemical species are enclosed in square brackets
    - Stoichiometric coefficients can be integers or decimals (e.g., 0.5 [A])
    - Reactants and products are separated by ">"
    - The rate constant identifier follows after a comma
    - Optional additional parameters follow a semicolon and are comma-separated

    Examples:
        >>> reactions = ['[A] + 2 [B] > [C], k1', '[C] > [A] + [B], k2 ; hv1, sigma1']
        >>> parsed, species = parse_reactions(reactions)
        >>> parsed
            [{'reactants': {'[A]': 1.0, '[B]': 2.0}, 'products': {'[C]': 1.0},
            'rate_constant': 'k1', 'other_multipliers': []},
            {'reactants': {'[C]': 1.0}, 'products': {'[A]': 1.0, '[B]': 1.0},
            'rate_constant': 'k2', 'other_multipliers': ['hv1', 'sigma1']}]
        >>> species
            ['[A]', '[B]', '[C]']

    Args:
        reactions (list | str): List of reaction strings to parse.

    Returns:
        tuple:
            A tuple containing:
            - parsed_reactions: list of dict
                Each dictionary contains:
                    * 'reactants': dict mapping species to stoichiometric coefficients
                    * 'products': dict mapping species to stoichiometric coefficients
                    * 'rate_constant': str, identifier of the rate constant
                    * 'other_multipliers': list of str, optional parameters for the reaction
            - sorted_species: list
                Alphabetically sorted list of all unique chemical species in the reaction network
    """

    parsed_reactions = []
    species_set = set()

    for reaction in reactions:
        reaction_dict = {
            "reactants": {},
            "products": {},
            "rate_constant": "",
            "other_multipliers": [],
        }

        # Split the reaction into main components
        reaction_part, rate_part = reaction.split(",", 1)
        rate_details = [x.strip() for x in rate_part.split(";")]

        reaction_dict["rate_constant"] = rate_details[
            0
        ]  # First element is the rate constant

        if len(rate_details) > 1:
            reaction_dict["other_multipliers"] = [
                item.strip() for item in rate_details[1].split(",")
            ]

        # Split reactants and products
        reactants_str, products_str = reaction_part.split(">")

        def parse_species(side):
            species_count = defaultdict(float)
            species_matches = re.findall(r"(?:([\d\.]+)\s*)?(\[[^\]]+\])", side)

            for count_str, species in species_matches:
                count = float(count_str) if count_str else 1.0
                species_count[species] += count
                species_set.add(species)

            return dict(species_count)

        reaction_dict["reactants"] = parse_species(reactants_str)
        reaction_dict["products"] = parse_species(products_str)

        parsed_reactions.append(reaction_dict)

    return parsed_reactions, sorted(species_set)


def reaction_string_to_matrix(reaction_string: str) -> dict:
    """
    Convert reaction strings to matrix representation (stoichiometry matrix equivalent).

    Args:
        reaction_string (str): The reaction string to convert.

    Returns:
        dict: Matrix representation of the reaction.
    """
    # This is a placeholder implementation - would need to be enhanced
    # based on the specific matrix format required
    parsed_reactions, species = parse_reactions([reaction_string])

    if not parsed_reactions:
        return {}

    reaction = parsed_reactions[0]
    matrix_data = {
        "species": species,
        "reactants": reaction["reactants"],
        "products": reaction["products"],
        "net_stoichiometry": {},
    }

    # Calculate net stoichiometry for each species
    for spec in species:
        reactant_coeff = reaction["reactants"].get(spec, 0)
        product_coeff = reaction["products"].get(spec, 0)
        matrix_data["net_stoichiometry"][spec] = product_coeff - reactant_coeff

    return matrix_data
