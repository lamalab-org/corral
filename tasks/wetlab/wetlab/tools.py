from corral.backend.tool import Tool, tool

from wetlab.engine import StockSolution, Solution, Precipitate, VolumeError
from wetlab.colors import PRECIPITATE_COLORS, PALETTE, mix_colors, closest_color_names

CATIONS = [
    'Ag+',
    'Al+3',
    'Ba+2',
    'Ca+2',
    'Cd+2',
    'Co+2',
    'Cs+',
    'Cu+2',
    'Fe+2',
    'Fe+3',
    'Hg+2',
    'Hg2+2',
    'K+',
    'Li+',
    'Mg+2',
    'Mn+2',
    'Na+',
    'NH4+',
    'Ni+2',
    'Pb+2',
    'Rb+',
    'Sr+2',
    'Zn+2',
]

ANIONS = [
    'Br-',
    'Cl-',
    'CO3-2',
    'HCO3-',
    'CrO4-2',
    'HCrO4-',
    'Cr2O7-2',
    'F-',
    'I-',
    'NO3-',
    'OH-',
    'PO4-3',
    'HPO4-2',
    'H2PO4-',
    'S-2',
    'HS-',
    'SCN-',
    'SO4-2',
    'HSO4-',
]

FLAME_COLORS = {
    "Ba+2": "green",
    "Ca+2": "orange-red",
    "Cs+": "blue-violet",
    "Cu+2": "turquoise",
    "K+": "pale violet",
    "Li+": "red",
    "Na+": "yellow",
    "Rb+": "red-violet",
    "Sr+2": "red",
}

@tool
def possible_cations() -> str:
    """[BRIEF] Returns the list of possible cations. [/BRIEF]
    
    [DETAILED] This functions returns a space-separated string of all the possible cations that can be present in an unknown sample. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Used to define the space of possible cations in unknown samples.
    - Usually used early on in the analysis.
    - Only suitable when the task involves identifying unknown cations in the sample(s). [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Make sure the task involves identifying one or more unknown cations in the samples. [/PREREQUISITE]
    2. [CURRENT] Use this tool to know what possible cations can be present in the unknown sample(s). [/CURRENT]
    3. [FOLLOW_UP] Can be followed up by tests to check for specific cations or groups of cations. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns a pre-defined list of cations as a space-separated string.
    - Each cation is formatted as 'X+n' where 'X' is the elemental symbol and '+n' is the charge.
    - The ammonium ion is represented as 'NH4+' and mercury(I) dimer is represented as 'Hg2+2'.
    - The tool does not perform any tests; it simply returns the pre-defined list of all possible cations. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `possible_cations()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] a string containing all the possible cations [/RETURNS_BRIEF]
            [RETURNS_DETAILED] a space-separated string, containing all possible cations that can be present in the unknown sample(s) [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "Ag+ Al+3 Ba+2 ..." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions under normal usage. [/ERROR_WHEN]
            [ERROR_DETAILS] N/A [/ERROR_DETAILS]
            [ERROR_RECOVERY] N/A [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - This tool returns the pre-defined list of all possible cations and does not perform any calculations or tests.
        - The list contains only the bare aqueous cations; cationic metal-complexes are not listed.
        - This tool is only useful when the task involves identifying unknown cations. If the task is about identifying solutions from a possible list of solutions with known compositions, there is no need for this tool.
    [/LIMITATIONS]
    """
    return ' '.join(CATIONS)

@tool
def possible_anions() -> str:
    """[BRIEF] Returns the list of possible anions. [/BRIEF]
    
    [DETAILED] This functions returns a space-separated string of all the possible anions that can be present in an unknown sample. One or more of these anions could be present in the unknown sample(s). [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Used to define the space of possible anions in unknown samples.
    - Usually used early on in the analysis.
    - Only suitable when the task involves identifying unknown anions in the sample(s). [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Make sure the task involves identifying one or more unknown anions in the sample(s). [/PREREQUISITE]
    2. [CURRENT] Use this tool to know what possible anions can be present in the unknown sample(s). [/CURRENT]
    3. [FOLLOW_UP] Can be followed up by tests to check for specific anions or groups of anions. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns a pre-defined list of anions as a space-separated string.
    - Each anion is formatted as 'XYZ-n' where 'XYZ' is the elemental composition and '-n' is the charge.
    - For example, hydrogen carbonate is represented as 'HCO3-' and dichromate is represented as 'Cr2O7-2'.
    - The tool does not perform any tests; it simply returns the pre-defined list of all possible anions. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `possible_anions()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] a string containing all the possible anions [/RETURNS_BRIEF]
            [RETURNS_DETAILED] a space-separated string, containing all possible anions that can be present in the unknown sample(s) [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "Br- Cl- CO3-2 HPO4-2 ..." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions under normal usage. [/ERROR_WHEN]
            [ERROR_DETAILS] N/A [/ERROR_DETAILS]
            [ERROR_RECOVERY] N/A [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - This tool returns the pre-defined list of all possible anions and does not perform any calculations or tests.
        - The list contains only the bare aqueous anions; anionic metal-complexes are not listed.
        - This tool is only useful when the task involves identifying unknown anions. If the task is about identifying solutions from a possible list of solutions with known compositions, there is no need for this tool.
    [/LIMITATIONS]
    """
    return ' '.join(ANIONS)

    
@tool(hidden_args=['compositions'])
def measure_pH(compositions, label: str) -> str:
    """[BRIEF] Measures the pH of the solution using a pH paper.[/BRIEF]
    
    [DETAILED] This tool measure the pH of a solution using a universal pH-indicator paper and reports the pH as the closest integer. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Used to measure the pH of a solution.
    - Sometimes knowing the initial pH of an unknown sample can provide information about its contents.
    - Some tests only work in certain pH ranges, therefore knowing the pH is important.
    - It can also be informative to measure the pH before and after performing a test to observe how it changes. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Make sure that `label` points to a solution and that knowing the pH of that solution gives valuable information. [/PREREQUISITE]
    2. [CURRENT] Use this tool to measure the pH of the solution. [/CURRENT]
    3. [FOLLOW_UP] After measuring the pH, you can perform another test and then measure the pH again to observe how that test changes the pH of the solution. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It uses universal indicator paper to measure the pH of a solution.
    - Returns the measure pH as an integer in the 0-14 range.
    - Values closer to 0 represent acidic, values around 7 represent neutral, and values closer to 14 represent alkaline solutions. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `measure_pH("sample")`,
        `measure_pH("sample_2")`,
        `measure_pH("sample_B")`,
        `measure_pH("test_01")`,
        `measure_pH("sample_1_test_2")`,
    ]
    [/SYNTACTICAL]

    Args:
        label (str):
            [ARGS_BRIEF] label of the target solution [/ARGS_BRIEF]
            [ARGS_DETAILED] a string representing the label of the solution in the Inventory, for which the pH will be measured [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] the label of supernatant solutions after filtration are appended with "_filtrate" [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "sample", "test_1", "test2_filtrate" [/ARGS_EXAMPLES]

    Returns:
        int:
            [RETURNS_BRIEF] the closest integer value to the actual pH of the solution [/RETURNS_BRIEF]
            [RETURNS_DETAILED] an integer value between 0-14, representing the closest integer to the actual pH of the solution [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] `4`, `6`, `11` [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        KeyError: [ERROR_WHEN] When the given `label` is invalid [/ERROR_WHEN]
                  [ERROR_DETAILS] The given `label` is not found in the Inventory [/ERROR_DETAILS]
                  [ERROR_RECOVERY] Check the Inventory and ensure you are using a correct `label` [/ERROR_RECOVERY]
        
        ValueError: [ERROR_WHEN] When the given `label` is not a solution [/ERROR_WHEN]
                    [ERROR_DETAILS] The given `label` exists in the Inventory but the object it points to is not a solution (for example it could be a precipitate) [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Check the Inventory and make sure the `label` you use points to a solution [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The pH can only be defined for solutions, not precipitates.
        - This tool does not return a precise pH value, only an integer estimate of the actual pH of the solution.
    [/LIMITATIONS]
    """
    try:
        target = compositions[label]
    except KeyError as e:
        raise KeyError(f"Invalid label: {e}")
    
    if isinstance(target, StockSolution):
        if type(target) == StockSolution:
            target = 1 * target  # converting StockSolution --> Solution
            target.equilibrate()
        pH = target.pH
    
    else:
        raise ValueError(f"{label} is not a solution!")

    pH = min(max(pH, 0), 14)
    return round(pH)


@tool(hidden_args=['compositions'])
def perform_flame_test(compositions, label: str) -> str:
    """[BRIEF] Performs a flame test on the solution. Cost = 1 mL [/BRIEF]
    
    [DETAILED] This test consumes 1 mL of the solution to perform a flame-color test. It returns the observed color of the flame (if any). [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Flame color can identify certain cations.
    - Useful to identify copper, alkali, and alkaline earth metals because of their characteristic flame colors.
    - It is especially useful for identifying alkali metals because they do not form precipitates under normal conditions. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Make sure that `label` points to a solution and that it does not contain a precipitate. If the solution has a precipitate, filter it before running this test. [/PREREQUISITE]
    2. [CURRENT] Use this tool to perform the flame test. Make sure that you have not previously introduced any interfering cations (i.e. those that result in a colored flame) to the solution as this will affect the results. [/CURRENT]
    3. [FOLLOW_UP] If a multi-colored flame is observed, you can try to remove some of the cations by precipitation and run the test again. If a certain color can be interpreted as more than one ion, you can perform additional tests to confirm which is present. You can also use the `lookup_flame_color` tool to retrieve the characteristic color of cations. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It uses 1 mL of the solution to perform a flame test.
    - If no species are present that give a characteristic flame color, the resulting observations is "No characteristic color".
    - If the flame has a single characteristic color, the resulting observation will indicate the exact name for that color.
    - If multiple different colors are observed, the resulting observation will be "A multi-colored flame". [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `perform_flame_test("sample")`,
        `perform_flame_test("sample_2")`,
        `perform_flame_test("sample_B")`,
        `perform_flame_test("test_01")`,
        `perform_flame_test("sample_1_test_2")`,
    ]
    [/SYNTACTICAL]

    Args:
        label (str):
            [ARGS_BRIEF] label of the target solution [/ARGS_BRIEF]
            [ARGS_DETAILED] a string representing the label of the solution in the Inventory, for which the flame test will be performed [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] the label of supernatant solutions after filtration are appended with "_filtrate" [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "sample", "test_1", "test2_filtrate" [/ARGS_EXAMPLES]

    Returns:
        str:
            [RETURNS_BRIEF] the resulting observation from the flame test [/RETURNS_BRIEF]
            [RETURNS_DETAILED] the resulting observation depends on the color of the flame. If no species present has a positive flame test, "No characteristic color" is returned. If exactly one color is observed, the observation will indicate that color. If more than one color is present in the flame, the observation will be "multi-colored flame"  [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "A red flame is observed.", "No characteristic flame color is observed.", "A multi-colored flame is observed." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        KeyError: [ERROR_WHEN] When the given `label` is invalid [/ERROR_WHEN]
                  [ERROR_DETAILS] The given `label` is not found in the Inventory [/ERROR_DETAILS]
                  [ERROR_RECOVERY] Check the Inventory and ensure you are using a correct `label` [/ERROR_RECOVERY]
        
        ValueError: [ERROR_WHEN] When the given `label` is not a solution [/ERROR_WHEN]
                    [ERROR_DETAILS] The given `label` exists in the Inventory but the object it points to is not a solution (for example it could be a precipitate) [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Check the Inventory and make sure the `label` you use points to a solution [/ERROR_RECOVERY]

        VolumeError: [ERROR_WHEN] When the solution contains a precipitate or less than 1 mL of it is remaining [/ERROR_WHEN]
                     [ERROR_DETAILS] The given `label` exists in the Inventory as a solution, but it either contains a precipitate or the remaining volume is less than 1 mL [/ERROR_DETAILS]
                     [ERROR_RECOVERY] If the solution contains a precipitate, filter it first and then perform the flame test again. If there is not enough solution to perform the test, you may be able to make more of it by repeating the steps that lead to it [/ERROR_RECOVERY]   
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The flame test's result depends on the concentration of cations; only species with a concentration higher than 5e-4 molar will give a positive result.
        - If you have introduced known interfering cations to the solution in previous steps, they will affect the results.
        - Some flame colors can be interpreted as more than one cation. Additional tests may be required to indicate the exact identity of the species.
        - A "multi-colored" flame means that there are at least two cations present with different flame colors. 
    [/LIMITATIONS]
    """
    try:
        target = compositions[label]
    except KeyError as e:
        raise KeyError(f"Invalid label: {e}")
    
    if not isinstance(target, StockSolution):
        raise ValueError(f"{label} is not a solution")
    
    test = 1 * target # drawing 1 mL from the target solution

    colors = []
    total_copper = 0
    for sp, conc in test.composition.items():
        if 'Cu' in sp:  # copper is the only flame-active element that exists as multiple species and not just "Cu+2", therefore, all copper-containing species are summed
            total_copper += conc
        elif (sp in FLAME_COLORS) and (conc > 5e-4):
            colors.append(FLAME_COLORS[sp])
    
    if total_copper > 5e-4:
        colors.append(FLAME_COLORS['Cu+2'])

    colors = list(set(colors))

    if len(colors) > 1:
        return "A multi-colored flame is observed."
    if len(colors) == 1:
        return f"A {colors[0]} flame is observed."
    if len(colors) == 0:
        return "No characteristic flame color is observed."


@tool
def lookup_flame_colors() -> str:
    """[BRIEF] Returns the characteristic flame colors of cations. [/BRIEF]
    
    [DETAILED] This tool returns a pre-defined list of ideal flame colors for each cation with a characteristic flame color. The mentioned colors are the color of the flame when no other cation with a positive flame test is present.  [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use this to retrieve the ideal characteristic flame colors of cations. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Perform a flame test using the `perform_flame_test`tool before looking up the colors. [/PREREQUISITE]
    2. [CURRENT] Use this tool to retrieve the expected characteristic flame colors. [/CURRENT]
    3. [FOLLOW_UP] If the result is a specific color, it usually means the presence of a specific cation. In some cases, more than one cation gives the same flame color in which case further tests can indicate which one. If the result is "No characteristic color" it means that none of the cations with a characteristic flame color are present or that their concentrations are lower than 5e-4 M. If "A multi-colored flame" is observed, it means that at least two of such cations are present but it is not clear which two (or more) are present. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - This tool does not perform any tests. It just return a pre-defined list of flame colors. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `lookup_flame_colors()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] the list of characteristic flame colors [/RETURNS_BRIEF]
            [RETURNS_DETAILED] a string in which each cation is on a separate line, along with its characteristic color [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "Ba+2     green\nCa+2     orange-red\n..." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions under normal usage. [/ERROR_WHEN]
            [ERROR_DETAILS] N/A [/ERROR_DETAILS]
            [ERROR_RECOVERY] N/A [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - This tool returns the pre-defined list of characteristic flame colors and does not perform any calculations or tests.
        - The flame test only works for cations that have a characteristic flame color.
        - If the concentration of a cation is below 5e-4, it will not give a colored flame.
    [/LIMITATIONS]
    """
    flame_colors = [f"{ion}     {color}" for ion,color in FLAME_COLORS.items()]
    return '\n'.join(flame_colors)


@tool(hidden_args=['compositions'])
def checkout_color(compositions, label: str) -> str:
    """[BRIEF] Observe the color of a solution or precipitate. [/BRIEF]
    
    [DETAILED] Observe the color of a solution or precipitate from the Inventory. This can be the color of a sample solution, a reagent or a precipitate from previous tests. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - You can use this tool to observe the color of sample solutions.
    - The changes in the color of solutions and/or precipitates are always reported as part of performing the tests, but you can also use this tool to check those colors again. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Make sure that `label` points to an object in the Inventory. [/PREREQUISITE]
    2. [CURRENT] Use this tool to check the color of the object. This is useful to observe the color of the initial sample solutions. [/CURRENT]
    3. [FOLLOW_UP] Take note of the observed colors. You can perform further tests on the object and notice how the observed colors change (or don't change). You can also use the `lookup_precipitate_colors` tool to retrieve the color of pure precipitates or the `simulate_color_mixture` tool to predict the color of a precipitate mixture, and compare that with the observed color. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns the type (reagent, solution, or precipitate) and the color of the target object.
    - For solutions that also contain a precipitate, two colors will be reported: one for the precipitate and one for the supernatant solution.
    - If a solution does not contain a precipitate, it will be called a "clear solution" and its color will be reported.
    - Color names are subjective and qualitative, so treat the reported color names as only approximate.
    - Since color names are subjective and approximate, sometimes it is possible that a color can be described with more than one color name. In those cases the possible color names are separated by a slash '/'. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `checkout_color("sample")`,
        `checkout_color("sample_A")`,
        `checkout_color("test_1_filtrate")`,
        `checkout_color("test_2_precipitate")`,
        `checkout_color("test3_precipitate_HNO3")`,
    ]
    [/SYNTACTICAL]

    Args:
        label (str):
            [ARGS_BRIEF] label of the target object [/ARGS_BRIEF]
            [ARGS_DETAILED] a string representing the label of the object in the Inventory which can be a solution (with or without a precipitate), a reagent, or a filtered precipitate [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] after filtration, the supernatant's label is appended with "_filtrate" and the precipitate is appended with "_precipitate" [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "sample", "test2_filtrate", "test3_precipitate" [/ARGS_EXAMPLES]

    Returns:
        str:
            [RETURNS_BRIEF] the type (reagent, solution, or precipitate) and the color of the object [/RETURNS_BRIEF]
            [RETURNS_DETAILED] a string containing a statement about the type and the color of the object. The type can be: a reagent solution, a precipitate, a clear solution (meaning it has no precipitate), or a solution containing a precipitate. In the latter case, the color of both the supernatant solution and the existing precipitate will be reported. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "sample_B is a clear solution with the following color: pale yellow", "test_03 is a solution that also contains a precipitate.\n Color of the precipitate: black\n Color of the supernatant solution: colorless", "test_1_precipitate is a precipitate with the following color: rosy brown / reddish gray" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        KeyError: [ERROR_WHEN] When the given `label` is invalid [/ERROR_WHEN]
                  [ERROR_DETAILS] The given `label` is not found in the Inventory [/ERROR_DETAILS]
                  [ERROR_RECOVERY] Check the Inventory and ensure you are using a correct `label` [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - If the target object is the result of a previously performed test, all color observations were already reported as part of that test, so using this tool to check those colors again will be redundant.
        - All reported colors are qualitative and approximate.
        - The perceived color of solutions will depend on the concentration of species in that solution. Both the hue and the lightness of the perceived color can change as the concentration of species in the solution change.
    [/LIMITATIONS]
    """
    try:
        target = compositions[label]
    except KeyError as e:
        raise KeyError(f"Invalid label: {e}")
    
    if type(target) == StockSolution:
        sample = 1 * target # converting StockSolution --> Solution
        return f"{label} is a reagent solution with the following color: {sample.color_name}"
    
    elif type(target) == Precipitate:
        return f"{label} is a precipitate with the following color: {target.color_name}"
    
    elif type(target) == Solution:
        if target.has_precipitate:
            prec_color = target.precipitate.color_name
            sol_color = target.color_name
            return f"{label} is a solution that also contains a precipitate.\n Color of the precipitate: {prec_color}\n Color of the supernatant solution: {sol_color}"
        else:
            return f"{label} is a clear solution with the following color: {target.color_name}"
        

@tool
def lookup_precipitate_colors() -> str:
    """[BRIEF] Returns the list of colored precipitates and their color. [/BRIEF]
    
    [DETAILED] This tool returns a pre-defined list of non-white precipitates and their corresponding color in their pure, freshly precipitated form. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use this tool if you want to know the color of specific precipitates in their pure form.
    - To know the exact color name used to describe the color of a pure precipitate, as reported by other tools. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Make sure you have observed the formation or change of appearance of precipitates before calling this tool. Useful tools include `checkout_color`, `mix_two_solutions`, and `add_solution`, and you might also observe a color change for an existing precipitate when using tools such as `add_solution` or `add_precipitate_to_solution`. [/PREREQUISITE]
    2. [CURRENT] Use this tool to get the list of color names associated with pure precipitates. [/CURRENT]
    3. [FOLLOW_UP] Remember that the listed color names will only exactly match with the reported colors from other tools if the precipitate is pure, meaning nothing else has co-precipitated with it. You can use the `simulate_color_mixture` tool to estimate the color of mixtures. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - The tool does not perform any tests; it simply returns the full pre-defined list of non-white precipitates and their colors in their pure form.
    - Since there is a large number of white precipitates, only colored (non-white) precipitates are listed; precipitates that are not listed are white. 
    - The observed color of precipitates reported by other tools will only match these listed colors if there is only one compound in the precipitate. If multiple compounds co-precipitate at the same time, it can affect the perceived color of the precipitate. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `lookup_precipitate_colors()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] the list of precipitate colors [/RETURNS_BRIEF]
            [RETURNS_DETAILED] a string where each separate line contains a single pure precipitate followed by its perceived color [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "AgBr     pale yellow\nAg2CO3     pale yellow\nAg2CrO4     brick red\n..." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions under normal usage. [/ERROR_WHEN]
            [ERROR_DETAILS] N/A [/ERROR_DETAILS]
            [ERROR_RECOVERY] N/A [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - This tool just returns the pre-defined list of precipitate colors and does not perform any calculations or tests.
        - The listed colors are for the pure precipitates only, not for the mixture of precipitates.
        - All reported colors are qualitative and approximate.
    [/LIMITATIONS]
    """
    note = "NOTE: This list only describes colored (non-white) precipitates. White precipitates are omitted; if a precipitate is not listed below, it means that it's white.\n"
    precipitate_colors = [f"{prec} :    {color}" for prec,color in PRECIPITATE_COLORS.items()]

    return note + '\n'.join(precipitate_colors)

@tool
def simulate_color_mixture(mixture: list[tuple[str, float]]) -> str:
    """[BRIEF] Given a mixture of precipitate colors, it mixes them with the given fractions and returns the name of the resulting precipitate color. [/BRIEF]
    
    [DETAILED] This tool simulates the mixing of precipitate colors with the given fractions and returns the name of the closest matching color of the resulting precipitate mixture. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use this tool to estimate the resulting color of a mixture of precipitates.
    - If the reported color of a precipitate is not listed by the `lookup_precipitate_colors` tool or if you suspect that an observed precipitate may have more than one component, you can use this tool to predict the color of a certain mixture of precipitates and compare it with the reported color.
    - If you have made a guess about the components of an observed precipitate, you can use this tool to predict its perceived color and compare that to the actual observed color. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] You need to have a hypothesis about the composition of an observed precipitate. Use the `lookup_precipitate_colors` tool to get the color names of the pure constituents. [/PREREQUISITE]
    2. [CURRENT] Assign fractions to the retrieved color names and use this tool to estimate the color of the resulting mixture. [/CURRENT]
    3. [FOLLOW_UP] Depending on the predicted color, you may want to keep or change your hypothesis, test other hypothetical compositions, or perform follow-up tests. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It receives a mixture of up to 3 color names and their fractions as a list of tuples.
    - Each tuple must be formatted as (color_name, fraction).
    - The fractions must be all positive and sum to 1.0. They will be rounded to two decimal places before performing the prediction, so there is no point in having more precision.
    - The colors will be mixed and the name of the closest matching color will be reported. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `simulate_color_mixture([("black", 0.5), ("pale yellow", 0.5)])`,
        `simulate_color_mixture([("white", 0.7), ("turquoise", 0.3)])`,
        `simulate_color_mixture([("cyan", 0.2), ("white", 0.2), ("pink", 0.6)])`,
        `simulate_color_mixture([("white", 0.1), ("reddish brown", 0.1), ("crimson", 0.8)])`,
        `simulate_color_mixture([("yellow", 0.6), ("brick red", 0.4)])`,
        
    ]
    [/SYNTACTICAL]

    Args:
        mixture (list[tuple[str, float]]):
            [ARGS_BRIEF] mixture components and their fractions as a list of tuples [/ARGS_BRIEF]
            [ARGS_DETAILED] a list of tuples, where the first element of the tuple is the color name and the second element its fraction in the mixture. The maximum allowed number of tuples in the list is three. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] list of tuples, each formatted as (<color>, <fraction>) [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] [("yellow", 0.5), ("red", 0.5)] , [("turquoise", 0.4), ("black", 0.3), ("dark blue", 0.3)] [/ARGS_EXAMPLES]

    Returns:
        str:
            [RETURNS_BRIEF] name of the resulting color [/RETURNS_BRIEF]
            [RETURNS_DETAILED] the closest matching name to the resulting color [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "dark olive", "lavender", "pale gray" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When a given color name is invalid [/ERROR_WHEN]
                    [ERROR_DETAILS] The first element of one of the tuples is not a valid color name [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Make sure you only use color names that correspond to pure precipitates as listed by the `lookup_precipitate_colors` tool [/ERROR_RECOVERY]
        
        ValueError: [ERROR_WHEN] When the mixture has more than 3 components [/ERROR_WHEN]
                    [ERROR_DETAILS] The list of color mixtures has 4 or more tuples [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Ensure that you are mixing at most 3 colors [/ERROR_RECOVERY]

        AssertionError: [ERROR_WHEN] When there is a numerical problem with the fractions [/ERROR_WHEN]
                        [ERROR_DETAILS] There is a non-positive fraction or the fractions do not sum to 1.0 [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Ensure that all fractions are positive values between 0 and 1 and they all sum to 1.0 [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - This simulation only works for the color of precipitates. It is not intended to be used to predict solution color.
        - Only a maximum of three (3) colors can be mixed.
        - The fractions will be rounded to two decimal places before the simulation.
        - The returned color name is an approximate close match to the actual color of the mixture and might not be an exact match.
    [/LIMITATIONS]
    """
    if len(mixture) > 3:
        raise ValueError(f"Attempted to mix {len(mixture)} colors. Up to 3 colors can be mixed.")
    
    try:
        hex_mixture = []
        for name, frac in mixture:
            hex = PALETTE[name]
            hex_mixture.append((hex, round(frac, 2)))
    except KeyError:
        raise ValueError(f"Undefined color name: {name}")
    
    result_hex = mix_colors(hex_mixture)
    return closest_color_names(result_hex, mode='precipitate', max_names=1)

@tool(hidden_args=['compositions'])
def get_available_reagents(compositions) -> str:
    """[BRIEF] Returns the list of available reagent solutions. [/BRIEF]
    
    [DETAILED] Returns a string where each reagent appears on a separate line. Each line contains the reagent's label as well as its composition. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Usually called in the beginning of a task to know all of the possible reagents available (if any) to solve the task.
    - If you want to perform a certain test and need a specific known solution, you can use this tool to check if that is available as a reagent.
    - You can use this tool to get the exact concentration of the components in the reagent solutions. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] None. [/PREREQUISITE]
    2. [CURRENT] Use this tool to get the full list of available reagents to use during the task. [/CURRENT]
    3. [FOLLOW_UP] You can perform any tests you wish by mixing the samples with the reagents using tools such as `mix_two_solutions` and `add_solution`. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - This tool does not perform any tests; it simply returns the full list of available reagents.
    - Each reagent appears on a separate line.
    - Each reagent is represented by its "reagent label" and its "composition".
    - The "reagent label" is the label to use when calling tools such as `mix_two_solutions`, `add_solution`, or `add_precipitate_to_solution`.
    - There is no limit on the amount of available reagent solutions, as opposed to sample solutions. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_available_reagents()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] a string containing all available reagents  [/RETURNS_BRIEF]
            [RETURNS_DETAILED] a string where each available reagent appears on a separate line and is represented by "reagent label" and "composition". If no additional reagents are available in the tasks, it will be mentioned by this tool. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "reagent label: HCl(0.02M)     composition: HCl 0.02 M, in water\nreagent label: KOH(6M)     composition: KOH 6.0 M, in water\n..." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions under normal usage. [/ERROR_WHEN]
            [ERROR_DETAILS] N/A [/ERROR_DETAILS]
            [ERROR_RECOVERY] N/A [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
    - This tool does not perform any tests; it simply returns the full list of available reagents.
    - All reagent solutions are at room temperature; there are no ways to heat up or cool down the reagents.
    [/LIMITATIONS]
    """
    reagent_descriptions = [f"reagent label: {k}     composition: {v.description}" for k,v in compositions.items() if type(v)==StockSolution]
    if len(reagent_descriptions)==0:
        return "There are no external reagents available."
    else:
        note = "NOTE: All reagent solutions are made with distilled water.\n"
        return note + '\n'.join(reagent_descriptions)

@tool(hidden_args=['compositions'])
def mix_two_solutions(compositions, test_label: str, sol1_label: str, sol1_vol: int, sol2_label: str,  sol2_vol: int) -> str:
    """[BRIEF] Mixes two solutions with the given volumes an returns observations about precipitation and color of the resulting solution [/BRIEF]
    
    [DETAILED] Mixes the two solutions (which must not contain any precipitates) with the given volumes (in mL) and reports observations about color change or precipitate formation. It also adds the resulting solution to the Inventory and labels it `test_label` [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - This tool lets you perform chemical tests and observe color changes and precipitate formations.
    - Use this tool when you want to mix two solutions with specific volumes and neither of them contain any precipitates.
    - This tool can be used to add a certain volume of a reagent to a certain volume of the sample solution or solutions from previous tests. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Make sure that `sol1_label` and `sol2_label` both point to reagents or solutions in the Inventory and that neither of them contain a precipitate. If a solution has precipitation, filter it using the `filter_solution` tool.  [/PREREQUISITE]
    2. [CURRENT] Use this tool to mix the solutions and add the new resulting solution to the Inventory. Notice the reported observations. [/CURRENT]
    3. [FOLLOW_UP] You can use your chosen `test_label` to refer to this solution when performing further tests. Sometimes for a given color change or precipitation to happen, you need to add more of one of the initial solutions, in that case you can use the `add_a_solution` tool. If the test resulted in the formation of a precipitate, you can either add other solutions to the mixture using the `add_a_solution` tool or filter the precipitate using the `filter_solution` tool. You can also lookup the color of precipitates by calling the `lookup_precipitates_color` tool. If you have a hypothesis about the composition of the formed precipitate you can try to check your hypothesis using the `simulate_color_mixture` tool. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It draws `sol1_vol` mL from `sol1_label`.
    - It draws `sol2_vol` mL from `sol2_label`.
    - Mixes the drawn volumes together in a new empty container labeled `test_label` and stirs until equilibrium is reached, at room temperature.
    - The resulting mixture is then added to the Inventory with the label `test_label`.
    - It returns two observations on two separate lines: one for the formation of precipitates (if any) and one for the color of the resulting solution.
    - Sometimes it is possible that the resulting precipitate's color can be described using more than one color name. In those cases the different given names will be separated by a slash '/'.
    - Remember that the reported colors are qualitative and approximate. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mix_two_solutions(test_label="test_1", sol1_label="HCl(1M)", sol1_vol=5, sol2_label="sample", sol2_vol=5)`, # mixing 5 mL of the reagent 'HCl(1M)' and 5 mL of 'sample', labeling the result as 'test_1'
        `mix_two_solutions(test_label="test_2_iodide", sol1_label="test_1", sol1_vol=3, sol2_label="NH4I", sol2_vol=2)`, # mixing 3 mL of 'test_1' and 2 mL of the reagent 'NH4I', labeling the result as 'test_2_iodide'
        `mix_two_solutions(test_label="A_B", sol1_label="sample_A", sol1_vol=5, sol2_label="sample_B", sol2_vol=5)`, # mixing 10 mL of 'sample_A' and 10 mL of 'sample_B', labeling the result as 'A_B'
        `mix_two_solutions(test_label="A_B_HCl", sol1_label="A_B", sol1_vol=2, sol2_label="HCl(6M)", sol2_vol=1)`, # mixing 10 mL of 'A_B' and 2 mL of the reagent 'HCl(6M)', labeling the result as 'A_B_HCl'
        `mix_two_solutions(test_label="sample_buffer", sol1_label="sample", sol1_vol=3, sol2_label="BUFFER_9", sol2_vol=5)`, # mixing 3 mL of 'sample' and 5 mL of the reagent 'BUFFER_9', labeling the result as 'sample_buffer'
    ]
    [/SYNTACTICAL]

    Args:
        test_label (str):
            [ARGS_BRIEF] label of the resulting solution [/ARGS_BRIEF]
            [ARGS_DETAILED] the label given to the resulting solution after the mixing. Use this label to refer to the resulting solution in further tests [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] use descriptive labels [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "test1_HCl", "test2_NH3_filt_iodide", "test3_excess_KOH" [/ARGS_EXAMPLES]

        sol1_label (str):
            [ARGS_BRIEF] label of the first solution [/ARGS_BRIEF]
            [ARGS_DETAILED] the label of the first solution (or reagent). This is how the solution (or reagent) is referred to in the Inventory (or reagent list). `sol1_vol` mL of this solution will be drawn [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] label string matching an entry in the Inventory or reagent list [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "sample", "test_1", "test2_filtrate", "NH4I", "HCl(1M)" [/ARGS_EXAMPLES]

        sol1_vol (int):
            [ARGS_BRIEF] volume of the first solution to draw [/ARGS_BRIEF]
            [ARGS_DETAILED] the volume (in mL) of the first solution, labeled `sol1_label`, to draw and mix with the second solution. The minimum allowed volume is 1 mL. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] integer value [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] `1`, `2`, `4` [/ARGS_EXAMPLES]

        sol2_label (str):
            [ARGS_BRIEF] label of the second solution [/ARGS_BRIEF]
            [ARGS_DETAILED] the label of the second solution (or reagent). This is how the solution (or reagent) is referred to in the Inventory (or reagent list). `sol2_vol` mL of this solution will be drawn [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] label string matching an entry in the Inventory or reagent list [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "sample", "test_1", "test2_filtrate", "NH4I", "HCl(1M)" [/ARGS_EXAMPLES]

        sol2_vol (int):
            [ARGS_BRIEF] volume of the second solution to draw [/ARGS_BRIEF]
            [ARGS_DETAILED] the volume (in mL) of the second solution, labeled `sol2_label`, to draw and mix with the first solution. The minimum allowed volume is 1 mL. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] integer value [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] `1`, `2`, `4` [/ARGS_EXAMPLES]

    Returns:
        str:
            [RETURNS_BRIEF] a string containing the observations from the test [/RETURNS_BRIEF]
            [RETURNS_DETAILED] a string containing two lines where the first line is an observation about the formation (or lack thereof) of precipitates and its color and the second line is about the color of the resulting solution. If the color of the precipitate can be described with more than one color name, up to 3 different color names will be given separated by slashes. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "No precipitate forms.\nThe resulting solution is colorless.", "A precipitate forms. Color: black\nThe supernatant solution is pale yellow." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        KeyError: [ERROR_WHEN] When `sol1_label` or `sol2_label` is invalid [/ERROR_WHEN]
                  [ERROR_DETAILS] The given `sol1_label` or `sol2_label` was not found in the Inventory [/ERROR_DETAILS]
                  [ERROR_RECOVERY] Make sure you are passing the correct labels. You can use the `get_available_reagents` and `check_inventory` tools. [/ERROR_RECOVERY]
        
        RuntimeError: [ERROR_WHEN] When `sol1_label` or `sol2_label` is not a solution [/ERROR_WHEN]
                      [ERROR_DETAILS] The given `sol1_label` or `sol2_label` was found in the Inventory but the object it points to is not a solution. [/ERROR_DETAILS]
                      [ERROR_RECOVERY] Make sure you are passing the correct labels. You can use the `get_available_reagents` and `check_inventory` tools. [/ERROR_RECOVERY]
        
        ValueError: [ERROR_WHEN] When `sol1_vol` or `sol2_vol` is not a positive integer greater than or equal to 1 [/ERROR_WHEN]
                    [ERROR_DETAILS] The given `sol1_vol` or `sol2_vol` is not an integer or it is less than 1 mL [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Make sure you use at least 1 mL of each solution [/ERROR_RECOVERY]
        
        VolumeError: [ERROR_WHEN] When `sol1_label` or `sol2_label` contain precipitates or the remaining volume is less than the requested amount [/ERROR_WHEN]
                     [ERROR_DETAILS] The given `sol1_label` or `sol2_label` was found in the Inventory and it is a solution but it either contains a precipitate or there is not enough of it remaining [/ERROR_DETAILS]
                     [ERROR_RECOVERY] If the solution contains a precipitate, either filter it using the `filter_solution` tool, or use the `add_a_solution` tool if the presence of the precipitate is necessary for the test. If the remaining amount of solution is less that the requested amount, try doing the test will a smaller volume if possible, or make more of that solution by repeating the steps that lead to it. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
    - This tool only works when mixing clear solutions, meaning solutions that do not contain any precipitates.
    - All solutions are at room temperature; there are no ways to heat up or cool down the solutions/reagents.
    - The minimum allowed volume to draw is 1 mL.
    - The reported colors are qualitative and approximate.
    - The perceived color of precipitates will depend on the composition of the precipitated solids. If more than one compound co-precipitate at the same time, the color may be different from the color of pure precipitates.
    - The perceived color of solutions will depend on the concentration of species in that solution. Both the hue and the lightness of the perceived color can change as the concentration of species in the solution change.
    - Sometimes a given change in color or precipitate formation needs more of one of the solutions to happen.
    [/LIMITATIONS]
    """
    try:
        sol1 = compositions[sol1_label]
    except KeyError as e:
        raise KeyError(f"Invalid sol1_label: {e}")
    
    try:
        sol2 = compositions[sol2_label]
    except KeyError as e:
        raise KeyError(f"Invalid sol2_label: {e}")

    if not isinstance(sol1, StockSolution):
        raise RuntimeError(f"{sol1_label} is not a solution!")
    if not isinstance(sol2, StockSolution):
        raise RuntimeError(f"{sol2_label} is not a solution!")
    
    if type(sol1_vol) != int or sol1_vol<1:
        raise ValueError("sol1_vol must be a positive integer greater than or equal to 1")
    if type(sol2_vol) != int or sol2_vol<1:
        raise ValueError("sol2_vol must be a positive integer greater than or equal to 1")
    
    test = sol1_vol * sol1 + sol2_vol * sol2
    description = f"{int(sol1_vol)} mL {sol1_label} + {int(sol2_vol)} mL {sol2_label}"
    test.description = description
    test.equilibrate()
    compositions[test_label] = test

    observations = []

    #precipitate observation
    if test.has_precipitate:
        prec_colors = test.precipitate.color_name
        tiny = "tiny amount of " if (1000 * test.precipitate.total_mol / test.volume < 5e-4) else "" # precipitates with an amount lower than 0.5 mmol/L are described as 'tiny'.
        observations.append(f"A {tiny}precipitate forms. Color: {prec_colors}")
    else:
        observations.append("No precipitate forms.")
    
    #solution observation
    sol_color = test.color_name
    if test.has_precipitate:
        observations.append(f"The supernatant solution is {sol_color}.")
    else:
        observations.append(f"The resulting solution is {sol_color}.")

    return '\n'.join(observations)


@tool(hidden_args=['compositions'])
def add_a_solution(compositions, test_label: str, sol1_label: str, sol2_label: str,  sol2_vol: int) -> str:
    """[BRIEF] Adds a specific volume of `sol2_label` to all of `sol1_label` and returns observations about the changes of solution color and precipitation amount and color. [/BRIEF]
    
    [DETAILED] Add the given volumes (in mL) of `sol2_label` (which must not contain any precipitates) to the remaining volume of `sol1_label` (which can have precipitates) and reports observations about color change or precipitate formation/dissolution. It also adds the resulting solution to the Inventory and labels it `test_label`. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - This tool lets you perform chemical tests and observe color changes and precipitate formations/dissolutions.
    - Use this tool when you want to mix two solutions and one of them (sol1) contains a precipitate.
    - This tool can be used to add a certain volume of a reagent or a clear solution to the remaining amount of another solution (which can also contain precipitates).
    - You can also use this tool to keep adding more of the same solution to a host solution (sol1).
    - Use this tool when you want to test wether the color or amount of an existing precipitate in a solution would change by adding another solution to it. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] If you used the `mix_two_solutions` tool and want to keep adding more of one of the solutions, you can use this tool. You can also use this tool if a previous test resulted in the formation of precipitate and you want to perform further tests on the resulting mixture.  [/PREREQUISITE]
    2. [CURRENT] Use this tool to add the given volume of sol2 to the whole remaining amount sol1. Notice the reported observations. [/CURRENT]
    3. [FOLLOW_UP] You can use your chosen `test_label` to refer to the resulting solution when performing further tests. If the test resulted in the formation of a precipitate, you can either add other solutions to the mixture by calling this tool again or filter the precipitate using the `filter_solution` tool. You can also lookup the color of precipitates by calling the `lookup_precipitates_color` tool. If you have a hypothesis about the color change of a pre-existing precipitate you can try to check your hypothesis using the `simulate_color_mixture` tool. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It draws `sol2_vol` mL from `sol2_label`.
    - Adds it to the same container as `sol1_label`, and stirs the contents until equilibrium is reached, at room temperature.
    - The remaining volume of `sol1_label` is set to 0 mL in the Inventory. The new solution is labeled `test_label` and is added to the Inventory.
    - It returns two observations on two separate lines: one about the change in the color and the amount of precipitates (if any) and one about the change in the color of the solution.
    - The reference for observations about the change in solution color is `sol1_label`.
    - When an observation mentions partial dissolution, it roughly means that somewhere between 15% to 50% of the original precipitate has dissolved.
    - When an observation mentions that a precipitate has mostly dissolved, it means that more than 50% of it has dissolved but there is still some undissolved precipitate remaining.
    - Sometimes it is possible that the precipitate's color can be described using more than one color name. In those cases the different given names will be separated by a slash '/'.
    - Remember that the reported colors are qualitative and approximate. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `add_a_solution(test_label="test_2B", sol1_label="test_2A", sol2_label="HCl(0.02M)", sol2_vol=1)`, # adding 1 mL of the reagent 'HCl(0.02M)' to 'test_2A', labeling the result as 'test_2B'
        `add_a_solution(test_label="A_B_HCl", sol1_label="A_B", sol2_label="HCl(6M)", sol2_vol=5)`, # adding 5 mL of the reagent 'HCl(6M)' to 'A_B', labeling the result as 'A_B_HCl'
        `add_a_solution(test_label="test_3", sol1_label="test_1", sol2_label="test_2", sol2_vol=2)`, # adding 2 mL of 'test_2' to 'test_1', labeling the result as 'test_3'
        `add_a_solution(test_label="test4_excess_KOH", sol1_label="test3_KOH", sol2_label="KOH(6M)", sol2_vol=2)`, # adding 2 mL of the reagent 'KOH(6M)' to 'test3_KOH', labeling the result as 'test4_excess_KOH'
        `add_a_solution(test_label="test5_more_NH3", sol1_label="test4_NH3", sol2_label="NH3(1M)", sol2_vol=5)`, # adding 5 mL of the reagent 'NH3(1M)' to 'test4_NH3', labeling the result as 'test4_more_NH3'
    ]
    [/SYNTACTICAL]

    Args:
        test_label (str):
            [ARGS_BRIEF] label of the resulting solution [/ARGS_BRIEF]
            [ARGS_DETAILED] the label given to the resulting solution after the mixing. Use this label to refer to the resulting solution in further tests [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] use descriptive labels [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "test1_HCl", "test2_NH3_filt_iodide", "test3_excess_KOH"  [/ARGS_EXAMPLES]

        sol1_label (str):
            [ARGS_BRIEF] label of the first solution [/ARGS_BRIEF]
            [ARGS_DETAILED] the label of the first (host) solution. This is how the solution is referred to in the Inventory. This cannot be a reagent. The remaining volume of this solution will be set to 0 mL after and it is replaced with `test_label` [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] label string matching a non-reagent solution in the Inventory [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "test_1", "test2_filtrate" [/ARGS_EXAMPLES]

        sol2_label (str):
            [ARGS_BRIEF] label of the second solution [/ARGS_BRIEF]
            [ARGS_DETAILED] the label of the second solution (or reagent), the one being added to `sol1_label`. This is how the solution (or reagent) is referred to in the Inventory (or reagent list). `sol2_vol` mL of this solution will be drawn and added to the host solution [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] label string matching an entry in the Inventory or reagent list [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "sample", "test_1", "test2_filtrate", "NH4I", "HCl(1M)" [/ARGS_EXAMPLES]

        sol2_vol (int):
            [ARGS_BRIEF] volume of the second solution to draw [/ARGS_BRIEF]
            [ARGS_DETAILED] the volume (in mL) of the second solution, labeled `sol2_label`, to draw and mix with the first solution. The minimum allowed volume is 1 mL [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] integer value [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] `1`, `2`, `4` [/ARGS_EXAMPLES]
  
    Returns:
        str:
            [RETURNS_BRIEF] a string containing the observations from the test [/RETURNS_BRIEF]
            [RETURNS_DETAILED] a string containing two lines where the first line is an observation about the formation/color change (or lack thereof) of precipitates, and the second line is about any color change of the supernatant solution. If the color of the precipitate can be described with more than one color name, up to 3 different color names will be given separated by slashes. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "The amount and color of the existing precipitate does not noticeably change.\nColor of the supernatant solution changes to very pale blue." , "The existing precipitate partially dissolves and changes color. New color: maroon / reddish brown.\nColor of the supernatant solution does not noticeably change." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        KeyError: [ERROR_WHEN] When `sol1_label` or `sol2_label` is invalid [/ERROR_WHEN]
                  [ERROR_DETAILS] The given `sol1_label` or `sol2_label` was not found in the Inventory [/ERROR_DETAILS]
                  [ERROR_RECOVERY] Make sure you are passing the correct labels. You can use the `get_available_reagents` and `check_inventory` tools. [/ERROR_RECOVERY]
        
        RuntimeError: [ERROR_WHEN] When `sol1_label` or `sol2_label` is not a solution [/ERROR_WHEN]
                      [ERROR_DETAILS] The given `sol1_label` or `sol2_label` was found in the Inventory but the object it points to is not a solution. [/ERROR_DETAILS]
                      [ERROR_RECOVERY] Make sure you are passing the correct labels. You can use the `get_available_reagents` and `check_inventory` tools. [/ERROR_RECOVERY]
        
        ValueError: [ERROR_WHEN] When `sol2_vol` is not a positive integer greater than or equal to 1 [/ERROR_WHEN]
                    [ERROR_DETAILS] The given `sol2_vol` is not an integer or it is less than 1 mL [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Make sure you are adding at least 1 mL of the second solution [/ERROR_RECOVERY]
        
        VolumeError: [ERROR_WHEN] When `sol2_label` contains a precipitate or its remaining volume is less than the requested amount [/ERROR_WHEN]
                     [ERROR_DETAILS] The given `sol2_label` was found in the Inventory and it is a solution but it either contains a precipitate or there is not enough of it remaining [/ERROR_DETAILS]
                     [ERROR_RECOVERY] If the second solution contains a precipitate, filter it using the `filter_solution` tool. If the remaining amount of it is less that the requested amount, try doing the test will a smaller `sol2_vol` if possible, or make more of that solution by repeating the steps that lead to it. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
    - The host solution (sol1) may contain precipitates but the second solution (sol2, the one being added) cannot contain any precipitates.
    - All solutions are at room temperature; there are no ways to heat up or cool down the solutions/reagents.
    - The minimum allowed volume for the second solution is 1 mL.
    - The reported colors are qualitative and approximate.
    - The perceived color of precipitates will depend on the composition of the precipitated solids. If more than one compound co-precipitate at the same time, the color may be different from the color of pure precipitates.
    - The perceived color of solutions will depend on the concentration of species in that solution. Both the hue and the lightness of the perceived color can change as the concentration of species in the solution change.
    - Any observations mentioning that an existing precipitate "partially" or "mostly" dissolves are qualitative and approximate statements.
    [/LIMITATIONS]
    """
    try:
        sol1 = compositions[sol1_label]
    except KeyError as e:
        raise KeyError(f"Invalid sol1_label: {e}")

    if type(sol1) != Solution:
        raise RuntimeError(f"{sol1_label} is not a valid solution!")
    
    if sol1.volume == 0:
        raise VolumeError(f"The remaining volume of {sol1_label} is zero!")
    
    try:
        sol2 = compositions[sol2_label]
    except KeyError as e:
        raise KeyError(f"Invalid sol2_label: {e}")
    
    if not isinstance(sol2, StockSolution):
        raise RuntimeError(f"{sol2_label} is not a solution!")
    
    if type(sol2_vol) != int or sol2_vol<1:
        raise ValueError("sol2_vol must be a positive integer greater than or equal to 1")
    
    description = f"{int(sol1.volume)} mL {sol1_label} + {int(sol2_vol)} mL {sol2_label}"

    old_supernatant, old_precipitate = sol1.filter()
    old_sol_color = sol1.color_name

    test = sol1 + sol2_vol * sol2
    test.description = description
    test.equilibrate()
    compositions[test_label] = test

    observations = []

    #precipitate observation
    if old_precipitate is None:
        if test.has_precipitate:
            prec_colors = test.precipitate.color_name
            tiny = "tiny amount of "  if (1000 * test.precipitate.total_mol / test.volume < 5e-4) else "" # precipitates with an amount lower than 0.5 mmol/L are described as 'tiny'.
            observations.append(f"A {tiny}precipitate forms. Color: {prec_colors}")
        else:
            observations.append("No precipitate forms.")
    
    else:
        old_amount = old_precipitate.total_mol
        old_color = old_precipitate.color_name

        if test.has_precipitate:
            new_amount = test.precipitate.total_mol
            new_color = test.precipitate.color_name
            # if there is at least one shared color name between the old and new color, we add a "slightly" modifier 
            old_color_set = set(old_color.split(' / '))
            new_color_set = set(new_color.split(' / '))
            shared_colors = old_color_set.intersection(new_color_set)
            slightly = "slightly " if len(shared_colors)>0 else ""
            
            precipitate_ratio = new_amount / old_amount 

            # checking if a new precipitate was formed
            test_no_prec = old_supernatant + sol2_vol * sol2.clone() # .clone() is used to prevent the volume of sol2 from decreasing twice
            test_no_prec.equilibrate()

            if test_no_prec.has_precipitate:
                additional_color = test_no_prec.precipitate.color_name
                tiny = "tiny amount of "  if (1000 * test_no_prec.precipitate.total_mol / test.volume < 5e-4) else "" # precipitates with an amount lower than 0.5 mmol/L are described as 'tiny'.
                if additional_color == old_color:
                    observations.append(f"A {tiny}precipitate with the same color as the existing precipitate forms.")
                elif new_color == old_color: 
                    observations.append(f"A {tiny}new precipitate (color: {additional_color}) forms, but does not cause the color of the existing precipitate to noticeably change.")
                else:
                    observations.append(f"A {tiny}new precipitate (color: {additional_color}) forms, mixing with the existing precipitate causing it to {slightly}change color. New color: {new_color}.")
            
            elif 0.85 < precipitate_ratio : # we assume that a change of less than 15% will not be noticeable
                if new_color == old_color:
                    observations.append("The amount and color of the existing precipitate does not noticeably change.")
                else:
                    observations.append(f"The amount of the existing precipitate does not noticeably change, but its color {slightly}changes. New color: {new_color}.")

            elif 0.5 <= precipitate_ratio <= 0.85 : # we will call a change of 15 to 50% 'partial dissolution'
                if new_color == old_color:
                    observations.append("The existing precipitate partially dissolves. Its color does not noticeably change.")
                else:
                    observations.append(f"The existing precipitate partially dissolves and {slightly}changes color. New color: {new_color}.")
            
            else: # meaning precipitate_ratio < 0.5
                if new_color == old_color:
                    observations.append("The existing precipitate mostly (but not fully) dissolves. Its color does not noticeably change.")
                else:
                    observations.append(f"The existing precipitate mostly (but not fully) dissolves and {slightly}changes color. New color: {new_color}.")

        else:
            observations.append(f"The existing precipitate fully dissolves.")


    #solution observation
    new_sol_color = test.color_name
    if new_sol_color == old_sol_color:
        if test.has_precipitate:
            observations.append("Color of the supernatant solution does not noticeably change.")
        else:
            observations.append("Color of the solution does not noticeably change.")
    
    else:
        if test.has_precipitate:
            observations.append(f"Color of the supernatant solution changes to {new_sol_color}.")
        else:
            observations.append(f"Color of the solution changes to {new_sol_color}.")

    return '\n'.join(observations)




@tool(hidden_args=['compositions'])
def filter_solution(compositions, label: str) -> str:
    """[BRIEF] Separates the precipitate from the supernatant solution. [/BRIEF]
    
    [DETAILED] Filters a solution, separating the precipitate from the supernatant solution, and adds the resulting filtrate and the precipitate to the Inventory. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use this tool when you want to remove the existing precipitate from a solution to perform further tests on only the supernatant 
    - Use this tool to collect the newly formed precipitate in a test if you want to perform further tests on the precipitate only [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] If performing a test using tools such as `mix_two_solutions` or `add_a_solution` result in the formation of a precipitate and you want to perform other tests only on the supernatant solution or the newly formed precipitate, you can use this tool two separate the two phases. [/PREREQUISITE]
    2. [CURRENT] Use this tool to separate the supernatant and the precipitate. [/CURRENT]
    3. [FOLLOW_UP] You can perform further tests on the supernatant by calling tools like `mix_two_solutions` or `add_a_solution`. You can perform further tests on the precipitate by adding it to a solution using the `add_precipitate_to_solution` tool. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It separates the aqueous phase and the solid phase from the `label`, and removes it from the inventory.
    - The supernatant is added to the Inventory, with the original label appended by '_filtrate'.
    - The precipitate is added to the Inventory, with the original label appended by '_precipitate'. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `filter_solution(label="test_2B")`, # 'test_2B' is removed from the Inventory and is replaced by 'test_2B_filtrate' and 'test_2B_precipitate'
        `filter_solution(label="test3_HCl")`, # 'test_HCl' is removed from the Inventory and is replaced by 'test3_HCl_filtrate' and 'test3_HCl_precipitate'
        `filter_solution(label="test_2A_H2S")`, # 'test_2A_H2S' is removed from the Inventory and is replaced by 'test_2A_H2S_filtrate' and 'test_2A_H2S_precipitate'
        `filter_solution(label="test2_filtrate_NH4I")`, # 'test2_filtrate_NH4I' is removed from the Inventory and is replaced by 'test2_filtrate_NH4I_filtrate' and 'test2_filtrate_NH4I_precipitate'
        `filter_solution(label="test1_precipitate_NH3")`, # 'test1_precipitate_NH3' is removed from the Inventory and is replaced by 'test1_precipitate_NH3_filtrate' and 'test1_precipitate_NH3_precipitate'
    ]
    [/SYNTACTICAL]

    Args:
        label (str):
            [ARGS_BRIEF] label of the target solution [/ARGS_BRIEF]
            [ARGS_DETAILED] the label given to the solution being filtered. This label is appended by '_filtrate' or '_precipitate' to refer to the separated phases. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] label of unfiltered solutions does not end with "_filtrate" [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "test1_HCl", "test_NH3_filt_iodide" [/ARGS_EXAMPLES]
  
    Returns:
        str:
            [RETURNS_BRIEF] a string with a message about the success/failure of the filtration [/RETURNS_BRIEF]
            [RETURNS_DETAILED] if the solution does not contain a precipitate [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "The solution was successfully filtered! The filtrate and precipitate are added to the Inventory." , "The target solution has no precipitate to filter! No change was made to the Inventory." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        KeyError: [ERROR_WHEN] When `label` is invalid [/ERROR_WHEN]
                  [ERROR_DETAILS] The given `label` was not found in the Inventory [/ERROR_DETAILS]
                  [ERROR_RECOVERY] Make sure you are passing the correct label. You can use the `check_inventory` tool. [/ERROR_RECOVERY]
        
        RuntimeError: [ERROR_WHEN] When `label` is a reagent solution or a precipitate [/ERROR_WHEN]
                      [ERROR_DETAILS] The given `label` was found in the Inventory but the object it points to is either a precipitate or a reagent solution [/ERROR_DETAILS]
                      [ERROR_RECOVERY] Make sure you are passing the correct label. You can use the `check_inventory` tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
    - Reagent solutions cannot be filtered.
    [/LIMITATIONS]
    """
    try:
        target = compositions[label]
    except KeyError as e:
        raise KeyError(f"Invalid label: {e}")
    
    if type(target) == StockSolution:
        raise RuntimeError("The target solution is a reagent and cannot be filtered.")
    elif type(target) == Precipitate:
        raise RuntimeError(f"The object with label {label} is not a solution, it's a precipitate!")
    else:
        assert type(target) == Solution
        if target.has_precipitate:
            filtrate, precipitate = target.filter()
            filtrate.description = target.description + " --> filtered"
            precipitate.description = target.description + " --> precipitate collected"
            
            filt_label = label + "_filtrate"
            prec_label = label + "_precipitate"
            compositions[filt_label] = filtrate
            compositions[prec_label] = precipitate
            compositions.pop(label)
            return "The solution was successfully filtered! The filtrate and precipitate are added to the Inventory."
        
        else:
            return "The target solution has no visible precipitate to filter! No change was made to the Inventory."


@tool(hidden_args=['compositions'])
def add_precipitate_to_solution(compositions, test_label: str, prec_label: str, sol_label: str,  sol_vol: int) -> str:
    """[BRIEF] Adds a precipitate to a specific volume of a solution and returns observations about the changes in the amount/color of the added precipitate or the solution color. [/BRIEF]
    
    [DETAILED] Draws `sol_vol` mL of `sol_label` (which must not contain any precipitates), adds to it all of the precipitate `prec_label` and reports observations about any changes in the color or the amount of the added precipitate and any color changes in the solution. It also adds the resulting solution to the Inventory and labels it `test_label`. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - This tool lets you perform chemical tests on the filtered precipitates.
    - Use this tool when you want to add the filtered precipitate from a previous test to another solution.
    - Use this tool to test if a precipitate dissolves in another solution or causes any other observable change in that solution. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] You need to have a separated precipitate in the Inventory before calling this tool. Separate the precipitates by using the `filter_solution` tool. [/PREREQUISITE]
    2. [CURRENT] Use this tool to add the precipitate to a given volume of the solution. It's better to choose the known reagents as the solution. [/CURRENT]
    3. [FOLLOW_UP] You can use your chosen `test_label` to refer to the resulting solution when performing further tests. If the added precipitate does not fully dissolve, you can add other solutions to the mixture by calling the `add_a_solution` tool or filter the precipitate again using the `filter_solution` tool. If the added precipitate fully dissolves, you can perform further tests on the resulting solution by calling tools such as `measure_pH`, `mix_two_solutions`, `add_a_solution` or `perform_flame_test`. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It draws `sol_vol` mL from `sol_label`.
    - Adds it to a new empty container labeled `test_label`.
    - Adds all of the precipitate labeled `prec_label` to the container and stirs the mixture until equilibrium is reached, at room temperature. 
    - The resulting mixture is added to the Inventory with the label `test_label`.
    - It returns two observations on two separate lines: one about the change in the color and the amount of the added precipitate, and one about the change in the color of the supernatant solution.
    - The reference for observations about the change in solution color is `sol_label`.
    - When an observation mentions partial dissolution, it roughly means that about 20-50% of the added precipitate has dissolved.
    - When an observation mentions that a precipitate has mostly dissolved, it means that more than 50% of it has dissolved but there is still some undissolved precipitate remaining.
    - Sometimes it is possible that the precipitate's color can be described using more than one color name. In those cases the different given names will be separated by a slash '/'.
    - Remember that the reported colors are qualitative and approximate. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `add_precipitate_to_solution(test_label="test_3", prec_label="test_2_precipitate", sol_label="test_1", sol_vol=4)`, # adding all of the precipitate 'test_2_precipitate' to 4 mL of the solution 'test_1', labeling the result as 'test_3'
        `add_precipitate_to_solution(test_label="prec3_HCl", prec_label="test3_precipitate", sol_label="HCl(1M)", sol_vol=5)`, # adding all of the precipitate 'test3_precipitate' to 5 mL of the reagent solution 'HCl(1M)', labeling the result as 'prec2_HCl'
        `add_precipitate_to_solution(test_label="filt4_ppt1", prec_label="test1_precipitate", sol_label="test4_K2CrO4_filtrate", sol_vol=10)`, # adding all of the precipitate 'test1_precipitate' to 10 mL of the solution 'test4_K2CrO4_filtrate', labeling the result as 'filt4_ppt1'
        `add_precipitate_to_solution(test_label="test2_ppt_NH3", prec_label="test2_precipitate", sol_label="NH3(5M)", sol_vol=4)`, # adding all of the precipitate 'test2_precipitate' to 4 mL of the reagent solution 'NH3(5M)', labeling the result as 'test2_ppt_NH3'
        `add_precipitate_to_solution(test_label="Ba_precipitate_HNO3", prec_label="test1_Ba_precipitate", sol_label="HNO3(1M)", sol_vol=10)`, # adding all of the precipitate 'test1_Ba_precipitate' to 10 mL of the reagent solution 'HNO3(1M)', labeling the result as 'Ba_precipitate_HNO3'
    ]
    [/SYNTACTICAL]

    Args:
        test_label (str):
            [ARGS_BRIEF] label of the resulting mixture [/ARGS_BRIEF]
            [ARGS_DETAILED] the label given to the resulting mixture after adding the precipitate. Use this label to refer to the resulting mixture in further tests [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] use descriptive labels [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "test1_HCl_precipitate_NH3", "test4_carbonate_precipitate_HNO3", "test3_excess_KOH_precipitate_HNO3"  [/ARGS_EXAMPLES]

        prec_label (str):
            [ARGS_BRIEF] label of the precipitate [/ARGS_BRIEF]
            [ARGS_DETAILED] the label of the precipitate to add. This is how the precipitate is referred to in the Inventory. All of the precipitate will be added to the `sol_label` solution and `prec_label` will be removed from the Inventory [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] label of filtered precipitates always end with "_precipitate" [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "test1_HCl_precipitate", "test4_carbonate_precipitate", "test3_excess_KOH_precipitate" [/ARGS_EXAMPLES]

        sol_label (str):
            [ARGS_BRIEF] label of the solution [/ARGS_BRIEF]
            [ARGS_DETAILED] the label of the solution (or reagent) that will receive the precipitate. This is how the solution (or reagent) is referred to in the Inventory (or reagent list). It must not contain any pre-existing precipitates [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] label string matching a clear solution in the Inventory or reagent list [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "test_1", "test2_filtrate", "NH4I", "HCl(1M)" [/ARGS_EXAMPLES]
        
        sol_vol (int):
            [ARGS_BRIEF] volume of the solution [/ARGS_BRIEF]
            [ARGS_DETAILED] the volume (in mL) of the host solution, labeled `sol_label`. This volume will be drawn from the solution and the precipitate is then added to the drawn volume. The minimum allowed volume is 4 mL [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] integer value [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] `1`, `2`, `4` [/ARGS_EXAMPLES]
  
    Returns:
        str:
            [RETURNS_BRIEF] a string containing the observations from the test [/RETURNS_BRIEF]
            [RETURNS_DETAILED] a string containing two lines where the first line is an observation about any changes in the color or the amount of the added precipitate, and the second line is about any color changes of the supernatant solution. If the color of the precipitate can be described with more than one color name, up to 3 different color names will be given separated by slashes. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "The added precipitate partially dissolves. Its color does not noticeably change.\nColor of the supernatant solution changes to very pale blue." , "The added precipitate fully dissolves.\nColor of the supernatant solution does not noticeably change." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        KeyError: [ERROR_WHEN] When `sol_label` or `prec_label` is invalid [/ERROR_WHEN]
                  [ERROR_DETAILS] The given `sol_label` or `prec_label` was not found in the Inventory [/ERROR_DETAILS]
                  [ERROR_RECOVERY] Make sure you are passing the correct labels. You can use the `get_available_reagents` and `check_inventory` tools. [/ERROR_RECOVERY]
        
        RuntimeError: [ERROR_WHEN] When `sol_label` is not a solution or `prec_label` is not a precipitate [/ERROR_WHEN]
                      [ERROR_DETAILS] The given `sol_label` and `prec_label` were found in the Inventory but the do not point to the correct type of object [/ERROR_DETAILS]
                      [ERROR_RECOVERY] Make sure you are passing the correct labels. You can use the `get_available_reagents` and `check_inventory` tools. [/ERROR_RECOVERY]
        
        ValueError: [ERROR_WHEN] When `sol_vol` is invalid [/ERROR_WHEN]
                    [ERROR_DETAILS] The given `sol_vol` is not an integer greater than or equal to 4 mL [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Make sure you are using at least 4 mL of the solution and passing it as an integer value [/ERROR_RECOVERY]
        
        VolumeError: [ERROR_WHEN] When `sol_label` contains a precipitate or its remaining volume is less than the requested amount [/ERROR_WHEN]
                     [ERROR_DETAILS] The given `sol_label` was found in the Inventory and it is a solution but it either contains a precipitate or there is not enough of it remaining [/ERROR_DETAILS]
                     [ERROR_RECOVERY] If the solution contains a precipitate, filter it using the `filter_solution` tool. If the remaining amount of it is less that the requested amount, try doing the test will a smaller `sol_vol` if possible, or make more of that solution by repeating the steps that lead to it. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
    - The solution may not contain any precipitates.
    - All solutions are at room temperature; there are no ways to heat up or cool down the solutions/reagents.
    - The minimum allowed volume for the solution is 4 mL.
    - The reported colors are qualitative and approximate.
    - The perceived color of precipitates will depend on the composition of the precipitated solids. If more than one compound co-precipitate at the same time, the color may be different from the color of pure precipitates.
    - The perceived color of solutions will depend on the concentration of species in that solution. Both the hue and the lightness of the perceived color can change as the concentration of species in the solution change.
    - Any observations mentioning that the added precipitate "partially" or "mostly" dissolves are qualitative and approximate statements.
    - In some cases it is possible that the chemical identity of the added precipitate change without any noticeable effects on its color or the color of the solution.
    [/LIMITATIONS]
    """
    try:
        prec = compositions[prec_label]
    except KeyError as e:
        raise KeyError(f"Invalid prec_label: {e}")
    
    try:
        sol = compositions[sol_label]
    except KeyError as e:
        raise KeyError(f"Invalid sol_label: {e}")
    
    if type(prec) != Precipitate:
        raise RuntimeError(f"{prec_label} is not a precipitate!")
    if type(sol) not in [Solution, StockSolution]:
        raise RuntimeError(f"{sol_label} is not a solution!")
    if type(sol) == Solution:
        if sol.has_precipitate: 
            raise VolumeError(f"{sol_label} already has a precipitate. If you want to add a different precipitate, you must filter it first!")

    if type(sol_vol) != int or sol_vol < 4:
        raise ValueError(f"`sol_vol` must be an integer greater than or equal to 4")
    
    old_amount = prec.total_mol
    old_color = prec.color_name

    test = sol_vol * sol
    test.equilibrate()
    old_sol_color = test.color_name

    test.add_solid(prec)
    test.equilibrate()
    test.description = f"{int(sol_vol)} mL {sol_label} + {prec_label}"

    compositions[test_label] = test
    compositions.pop(prec_label)

    observations = []

    #precipitate observation
    if test.has_precipitate:
        new_amount = test.precipitate.total_mol
        new_color = test.precipitate.color_name
        # if there is at least one shared color name between the old and new color, we add a "slightly" modifier 
        old_color_set = set(old_color.split(' / '))
        new_color_set = set(new_color.split(' / '))
        shared_colors = old_color_set.intersection(new_color_set)
        slightly = "slightly " if len(shared_colors)>0 else ""
        
        precipitate_ratio = new_amount / old_amount 

        if 0.85 <=  precipitate_ratio : # we assume that a change of less than 15% will not be noticeable
            if new_color == old_color:
                observations.append("The amount and color of the added precipitate does not noticeably change.")
            else:
                observations.append(f"The amount of the added precipitate does not noticeably change, but its color {slightly}changes. New color: {new_color}")

        elif 0.50 <= precipitate_ratio < 0.85 : # we will call a change of 15 to 50% 'partial dissolution'
            if new_color == old_color:
                observations.append("The added precipitate partially dissolves. Its color does not noticeably change.")
            else:
                observations.append(f"The added precipitate partially dissolves and its color {slightly}changes. New color: {new_color}")
        
        else:
            if new_color == old_color:
                observations.append("The added precipitate mostly (but not fully) dissolves. Its color does not noticeably change.")
            else:
                observations.append(f"The added precipitate mostly (but not fully) dissolves and its color {slightly}changes. New color: {new_color}")

    else:
        observations.append(f"The added precipitate fully dissolves.")
    
    #solution observation
    new_sol_color = test.color_name
    if new_sol_color == old_sol_color:
        if test.has_precipitate:
            observations.append("Color of the supernatant solution does not noticeably change.")
        else:
            observations.append("Color of the solution does not noticeably change.")
    
    else:
        if test.has_precipitate:
            observations.append(f"Color of the supernatant solution changes to {new_sol_color}.")
        else:
            observations.append(f"Color of the solution changes to {new_sol_color}.")
    
    return '\n'.join(observations)


@tool(hidden_args=['compositions'])
def check_inventory(compositions) -> str:
    """[BRIEF] Returns the current contents of the Inventory [/BRIEF]
    
    [DETAILED] Returns a string containing the details of the solutions and precipitates in the Inventory, including their remaining volumes in mL [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Typically used in the beginning of a task to check the unknown samples.
    - You can use this tool to check the Inventory, which is effectively a summary of the experiments performed so far. 
    - Use this tool to check the remaining volumes of the samples and test solutions.
    - Use this tool to check the correct label of solutions and precipitates to use in other tools. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] None. [/PREREQUISITE]
    2. [CURRENT] Use this tool to retrieve the full Inventory of solutions and filtered precipitates, including their remaining volumes in mL. [/CURRENT]
    3. [FOLLOW_UP] Use the labels returned by this tool to call other tools. You can also use the returned Inventory as a summary of the experiments you have performed so far and decide what experiment you want to perform next. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It returns the Inventory as a string, with each item printed on a separate line as a dictionary.
    - Each line contains the details about a separate solution or precipitate in the Inventory.
    - Solutions with a remaining volume of 0 mL are kept in the Inventory for record-keeping purposes.
    - When a solution is filtered using the `filter_solution` tool, it is removed from the Inventory and is replaced by the filtrate and the collected precipitate.
    - When a precipitate is added to a solution using the `add_precipitate_to_solution` tool, it is removed from the Inventory.
    - Reagent solutions are not part of the Inventory, use the `get_available_reagents` tool to check the reagents.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `check_inventory()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [RETURNS_BRIEF] a string containing the solutions and precipitates in the Inventory [/RETURNS_BRIEF]
            [RETURNS_DETAILED] a string where each line represents a different item of the Inventory as a dictionary. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "{'label': 'sample', 'type': 'clear solution', 'description': 'unknown', 'remaining_volume': '11 mL'}\n{'label': 'test_1', 'type': 'clear solution', 'description': '9 mL sample + 1 mL HCl(1M)', 'remaining_volume': '0 mL'}\n{'label': 'test_2_filtrate', 'type': 'clear solution', 'description': '5 mL test_1 + 1 mL NH4I --> filtered', 'remaining_volume': '6 mL'}\n{'label': 'test_2_precipitate', 'type': 'precipitate', 'description': '5 mL test_1 + 1 mL NH4I --> precipitate collected'}" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        None:
            [ERROR_WHEN] This tool does not raise exceptions under normal usage. [/ERROR_WHEN]
            [ERROR_DETAILS] N/A [/ERROR_DETAILS]
            [ERROR_RECOVERY] N/A [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - Reagents are not shown by tool. Use the `get_available_reagents` tool to check the available reagents.
        - This tool does not report the color if items, use the `checkout_color` tool to observe the color of a specific item.
        - The observations made during an experiment involving an item are not stored in the Inventory.
    [/LIMITATIONS]
    """

    solution_descriptions = [
        str({
            "label": k, 
            "type": "solution with precipitate" if v.has_precipitate else "clear solution",
            "description": v.description,
            "remaining_volume": f"{int(v.volume)} mL",
        })
        for k,v in compositions.items() if isinstance(v, Solution)
    ]
    
    precipitate_descriptions = [
        str({
            "label": k,
            "type": "precipitate",
            "description": v.description,
        })
        for k,v in compositions.items() if isinstance(v, Precipitate)
    ]

    return '\n'.join(solution_descriptions + precipitate_descriptions)

def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    return {
        "possible_cations": possible_cations,
        "possible_anions": possible_anions,
        "measure_pH": measure_pH,
        "perform_flame_test": perform_flame_test,
        "lookup_flame_colors": lookup_flame_colors,
        "checkout_color": checkout_color,
        "lookup_precipitate_colors": lookup_precipitate_colors,
        "simulate_color_mixture": simulate_color_mixture,
        "get_available_reagents": get_available_reagents,
        "mix_two_solutions": mix_two_solutions,
        "add_a_solution": add_a_solution,
        "filter_solution": filter_solution,
        "add_precipitate_to_solution": add_precipitate_to_solution,
        "check_inventory": check_inventory,
    }
