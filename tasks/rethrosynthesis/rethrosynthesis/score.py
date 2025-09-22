from loguru import logger

MAX_PRICE = 100.0  # Maximum price


def score_final(prediction: dict) -> float:
    """
    Function to score the retrosynthesis route based on the provided conditions.
    Returns 1.0 if all conditions are met, else returns 0.0.

    Args:
        prediction (dict): The retrosynthesis route in JSON format.

    Returns:
        float: 1.0 if all conditions are met, 0.0 if any condition is violated.
    """
    # Initialize total cost
    total_cost = 0.0

    #  1. Check if all final leaves are in the catalog
    for step in prediction["steps"]:
        # Process each step in the route
        for species in step["reactants"] + step["products"]:
            smiles = species["smiles"]
            sku = species["sku"]
            price = species["price_usd"]

            # 1.1 Check if species is in the catalog (in stock)
            if not is_in_catalog(sku):
                logger.error(f"Species {smiles} is not in the catalog.")
                return 0.0  # Invalid route

            # 1.2 Check if the price exceeds the max allowed price
            total_cost += price

            if total_cost > MAX_PRICE:
                logger.error(
                    f"Total price {total_cost} exceeds the max allowed price {MAX_PRICE}."
                )
                return 0.0  # Invalid route

            # 1.3 Check if the SMILES is correct by applying the forward reaction
            if not is_smiles_correct(
                smiles, step["template_id"], step["reactants"], step["products"]
            ):
                logger.error(
                    f"SMILES {smiles} does not match expected result for template {step['template_id']}."
                )
                return 0.0  # Invalid route

    # 2. Check if all the final leaves generate the correct molecule
    for final_species in prediction["buyable_leaves"]:
        final_smiles = final_species["smiles"]

        if not is_in_catalog(final_species["sku"]):
            logger.error(f"Final leaf species {final_smiles} is not in the catalog.")
            return 0.0  # Invalid route

        if not is_smiles_correct(
            final_smiles,
            prediction["steps"][-1]["template_id"],
            prediction["steps"][-1]["reactants"],
            prediction["steps"][-1]["products"],
        ):
            logger.error(
                f"Final leaf SMILES {final_smiles} does not match expected result."
            )
            return 0.0  # Invalid route

    # 3. If all checks pass, return 1.0
    return 1.0


def is_in_catalog(smiles):
    """
    Function to check if a species is in the catalog.
    This should be replaced with a function that queries the catalog database.

    smiles: str
        The SMILES of the species to check.

    Returns:
        bool: True if the species is in stock, False otherwise.
    """
    return smiles


def is_smiles_correct(
    smiles: str, template_id: str, reactants: list[str], products: list[str]
) -> bool:
    """
    Function to check if a given SMILES matches the expected product after applying the forward reaction.

    Args:
        smiles (str): The SMILES to check.
        template_id (str): The reaction template ID.
        reactants (list[str]): The reactants involved in the reaction.
        products (list[str]): The expected products of the reaction.

    Returns:
        bool: True if the SMILES matches the expected result, False otherwise.
    """
    return smiles, template_id, reactants, products
