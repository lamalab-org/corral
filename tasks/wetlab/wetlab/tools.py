from corral.base import Tool
from corral.utils import tool

@tool
def possible_ions() -> str:
    """
    Returns the list of all possible unknown ions that can be present in the sample.
    """
    pass

@tool(hidden_args=['compositions'])
def measure_ph(compositions, solution_name: str) -> str:
    """
    Measures the pH of the solution using universal indicator pH paper.
    Costs: 0 mL
    Returns: the closest integer to the actual pH of the solution
    """
    pass

@tool(hidden_args=['compositions'])
def perform_flame_test(compositions, solution_name: str) -> str:
    """
    Perfoms the flame test on the given solution.
    Costs: 1 mL
    Returns: <color> if only one flame color is present
             "mixed" if more than one flame color is present
             "none" if no distinctive flame color is observed
    """
    pass

@tool
def lookup_flame_colors() -> str:
    """
    Returns the full list of cations and their flame colors.
    Costs: 0 mL
    """
    pass

@tool
def get_available_reagents() -> str:
    """
    Returns the list of available reagents and their compotions.
    -Example:
        reagent_name: NH4Cl     composition: 0.1 M NH4Cl
        reagent_name: H2S_HCl   composition: 0.1 M H2S, 0.01 M HCl
        etc.
    """
    pass

@tool(hidden_args=['compositions'])
def perform_reagent_test(compositions, solution_name: str, reagent_name: str, solution_vol: int, reagent_vol: int, label: str) -> str:
    """
    Mix the solution and the reagent with the given volumes, call it <label> and add it to the inventory
    Costs: <solution_vol> (to prevent the agent from using very small volumes, there is a minimum cap on this)
    Returns: a list of observations ("nothing happens" if no observation).
    -Examples:
        "a black precipitate forms"
        "the solution turns blood-red"
        etc.
    """
    pass

@tool(hidden_args=['compositions'])
def check_inventory() -> str:
    """
    Shows the record of previous reagent tests and their remaining volumes.
    Costs: 0 mL
    -Example:
        sample                  -                   remaining: 46 mL
        test_1      (1 mL NH4Cl + 4 mL sample)      remaining: 1 mL
        test_2      (1 mL KI + 4 mL test_1)         remaining: 5 mL
        etc.
    """
    pass

@tool
def lookup_solid_color(color: str) -> str:
    """
    Returns the list of all possible precipitates with color <color>
    """
    pass

@tool
def lookup_solution_color(color: str) -> str:
    """
    Returns the list of all possible aqueous species with color <color>
    """
    pass


def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    return {
        "possible_ions": possible_ions,
        "measure_ph": measure_ph,
        "perform_flame_test": perform_flame_test,
        "lookup_flame_colors": lookup_flame_colors,
        "get_available_reagents": get_available_reagents,
        "perform_reagent_test": perform_reagent_test,
        "check_inventory": check_inventory,
        "lookup_solid_color": lookup_solid_color,
        "lookup_solution_color": lookup_solution_color,
    }
