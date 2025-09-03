def generate_rate_law(
    reaction: str, mechanism: str = "elementary", rxn_id: int = 1
) -> dict:
    """
    Generate rate law expression for a reaction.
    Supports:
      - elementary: mass-action kinetics
      - michaelis_menten: enzyme kinetics
      - langmuir_hinshelwood: catalytic surface reactions

    Args:
        reaction: reaction string (e.g. "2A + B -> C")
        mechanism: kinetic mechanism type
        rxn_id: integer index of the reaction (for unique parameter names)
    """
    from .reaction_parser import parse_reactions

    if mechanism == "elementary":
        species = list(
            {
                c
                for c in reaction.replace("->", "+").replace("+", " ").split()
                if c.isalpha()
            }
        )
        parsed = parse_reactions([reaction], species)[0]

        reactants = parsed["reactants"]
        expr_parts = [
            f"[{sp}]**{coeff}" if coeff > 1 else f"[{sp}]"
            for sp, coeff in reactants.items()
        ]
        k_name = f"k{rxn_id}"
        expression = f"{k_name} * " + " * ".join(expr_parts) if expr_parts else k_name

        return {
            "expression": expression,
            "parameters": {k_name: 1.0},
            "units": "concentration/time",
        }

    elif mechanism == "michaelis_menten":
        species = [
            c
            for c in reaction.replace("->", "+").replace("+", " ").split()
            if c.isalpha()
        ]
        if len(species) < 2:
            raise ValueError(
                f"Michaelis-Menten requires at least enzyme and substrate: {reaction}"
            )

        substrate = species[1] if species[0].upper() == "E" else species[0]
        vmax_name = f"Vmax{rxn_id}"
        km_name = f"Km{rxn_id}"
        expression = f"{vmax_name} * [{substrate}] / ({km_name} + [{substrate}])"

        return {
            "expression": expression,
            "parameters": {vmax_name: 1.0, km_name: 1.0},
            "units": "concentration/time",
        }

    elif mechanism == "langmuir_hinshelwood":
        species = [
            c
            for c in reaction.replace("->", "+").replace("+", " ").split()
            if c.isalpha()
        ]
        if len(species) < 2:
            raise ValueError(
                f"Langmuir-Hinshelwood requires at least two reactants: {reaction}"
            )

        A, B = species[0], species[1]
        k_name = f"k{rxn_id}"
        KA_name = f"K_{A}{rxn_id}"
        KB_name = f"K_{B}{rxn_id}"
        expression = (
            f"({k_name} * [{A}] * [{B}]) / (1 + {KA_name}[{A}] + {KB_name}[{B}])"
        )

        return {
            "expression": expression,
            "parameters": {k_name: 1.0, KA_name: 1.0, KB_name: 1.0},
            "units": "concentration/time",
        }

    else:
        raise ValueError(f"Unsupported mechanism: {mechanism}")
