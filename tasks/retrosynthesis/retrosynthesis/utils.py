import re
from time import sleep
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Tag
from rdkit import Chem
from rxnutils.chem.reaction import ChemicalReaction

from corral.utils.modal import remote_call
from corral.utils.tool_helpers import make_api_request

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en,es-ES;q=0.9,es;q=0.8",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def search_by_template(template_id: str) -> list[dict[str, Any]] | str:
    return search_catalog(template_id)


def apply_template_retro(product_smiles: str, template_id: str) -> list[str]:
    """
    Verify a retrosynthesis step by checking if the given reaction template can produce the expected product.
    This function should be replaced with a call to a retrosynthesis prediction model or API.

    product_smiles: str
        The SMILES of the expected product molecule.
    template_id: str
        The identifier of the reaction template to use.

    Returns:
        list[str]: A list of SMILES strings representing the predicted reactants.
    """
    reaction_data = search_by_template(template_id)
    rxn = ChemicalReaction(reaction_data["mapped_rxn"])
    rxn.generate_reaction_template()
    return rxn.retro_template.apply(product_smiles)


def extract_chemical_info(element: Tag) -> dict[str, Any]:
    """
    Extract chemical information from HTML element
    Returns a dictionary with chemical name, CAS number, purity, amount, and price
    """
    # Initialize result dictionary
    chemical_info = {
        "chemical_name": None,
        "cas_number": None,
        "amount": None,
        "price_per_100g": None,
    }

    # Extract chemical name
    chemical_name_cell = element.find("td", class_="chemical-name")
    if chemical_name_cell:
        # Get the text and clean it up
        name_text = chemical_name_cell.get_text().strip()
        # Remove the icon and extra whitespace, split by newlines and take first part
        chemical_info["chemical_name"] = name_text.split("\n")[0].strip()

    # Extract CAS number
    cas_cell = element.find("td", class_="cas-cell")
    if cas_cell:
        cas_span = cas_cell.find("span", class_="cas-tooltip")
        if cas_span and cas_span.get("data-cas"):
            chemical_info["cas_number"] = cas_span.get("data-cas")
        else:
            # Fallback: extract from text
            cas_text = cas_cell.get_text().strip()
            cas_match = re.search(r"\d+-\d+-\d+", cas_text)
            if cas_match:
                chemical_info["cas_number"] = cas_match.group()

    # Extract purity
    purity_cell = element.find("td", class_="purity-column")
    if purity_cell:
        purity_div = purity_cell.find("div", class_="purity-value")
        if purity_div:
            chemical_info["purity"] = purity_div.get_text().strip()

    # Extract amount/packaging
    packaging_cells = element.find_all("td")
    for cell in packaging_cells:
        price_per_100_div = cell.find("div", class_="price-per-100")
        packaging_info_div = cell.find("div", class_="packaging-info")

        if price_per_100_div and packaging_info_div:
            amount_text = price_per_100_div.get_text().strip()
            if not amount_text.startswith("$"):  # Make sure it's not a price
                chemical_info["amount"] = amount_text
            break

    # Extract prices
    price_cells = element.find_all("div", class_="price-per-100")
    for price_cell in price_cells:
        price_text = price_cell.get_text().strip()
        if "per 100 g" in price_text:
            chemical_info["price_per_100g"] = price_text

    # Extract supplier info
    supplier_cell = element.find("td", class_="supplier-cell")
    if supplier_cell:
        supplier_name_link = supplier_cell.find("a", class_="supplier-name")
        if supplier_name_link:
            chemical_info["supplier"] = supplier_name_link.get_text().strip()

    return chemical_info


def extract_chemicals(text: str) -> list[dict[str, Any]]:
    """Extract chemical information dictionaries from text using BeautifulSoup."""
    soup = BeautifulSoup(text, "html.parser")
    rows = soup.select("tr.expandable-row")
    return [extract_chemical_info(row) for row in rows]


def search_catalog(cas: str) -> list[dict[str, Any]] | str:
    """Searches a catalog for available precursors. Returns a list of chemical info dicts or a not-found message."""

    sleep(5)
    response = make_api_request(
        url=f"https://www.chemicalsuppliers.com/buy-?cas_numbers%5B%5D={cas}&physical_state_filter=all",
        method="GET",
        headers=HEADERS,
        params={},
        verbose=False,
        json=False,
    )
    chemicals = extract_chemicals(response)
    return chemicals if chemicals else []


def _is_buyable(smiles: str) -> bool:
    chemicals = search_catalog(smiles)
    return bool(chemicals)


def check_price(smiles: str) -> float:
    """
    Function to check the price of a molecule given its SMILES.
    This should be replaced with a function that queries a pricing database or API.

    smiles: str
        The SMILES of the molecule to check.

    Returns:
        float: The price of the molecule in USD.
    """
    cas_number = remote_call(function_name="return_cas_number", env_name="chemenv")(
        compound=smiles
    )
    chemicals = search_catalog(cas_number)
    price = chemicals[0].get("price_per_100g", "$0")
    try:
        return float(price.replace("$", "").replace(",", ""))
    except Exception as e:
        raise ValueError(f"Could not parse price: {price}") from e


def valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    return Chem.MolFromSmiles(smiles) is not None


def species_match(predicted: list[str], ground_truth: list[str]) -> bool:
    """Check if two lists of species (SMILES) match, ignoring order."""
    # First check if both lists have the same length
    if len(ground_truth) != len(predicted):
        return False

    # Validate all SMILES strings
    for smiles in ground_truth + predicted:
        if not valid_smiles(smiles):
            return False

    # Convert all SMILES to canonical form for comparison
    actual_canonical = []
    predicted_canonical = []

    for smiles in ground_truth:
        mol = Chem.MolFromSmiles(smiles)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        if mol is None:
            return False
        actual_canonical.append(Chem.MolToSmiles(mol))

    for smiles in predicted:
        mol = Chem.MolFromSmiles(smiles)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        if mol is None:
            return False
        predicted_canonical.append(Chem.MolToSmiles(mol))

    # Check if both lists contain the same canonical SMILES (ignoring order)
    return sorted(actual_canonical) == sorted(predicted_canonical)


def apply_template_forward(reactants: str, template_id: str) -> list[str]:
    """
    Apply a reaction template in the forward direction to predict products.

    reactants: str
        The SMILES of the reactants, separated by dots if multiple.
    template_id: str
        The identifier of the reaction template to use.

    Returns:
        list[str]: The predicted product SMILES strings.
    """
    reaction_data = search_by_template(template_id)
    rxn = ChemicalReaction(reaction_data["mapped_rxn"])
    rxn.generate_reaction_template()
    return rxn.canonical_template.apply(reactants)
