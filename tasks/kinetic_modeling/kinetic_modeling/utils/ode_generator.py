def build_ode_system(network: dict, rate_laws: list[dict]) -> str:
    """
    Return Python code string that defines an ODE system function.
    """
    species = network["species"]
    S = network["stoichiometry_matrix"]

    code = "import numpy as np\n"
    code += "def ode_system(t, y, params):\n"
    # rate laws
    for j, rl in enumerate(rate_laws):
        code += f"    r{j} = params['k{j+1}']"
        expr = (
            rl["rate_expression"].replace("[", "y[species_index['").replace("]", "']]")
        )
        code += f" * {expr}\n"
    # ODEs
    code += "    dydt = np.zeros(len(y))\n"
    for i, _sp in enumerate(species):
        terms = []
        for j in range(len(rate_laws)):
            coeff = int(S[i, j])
            if coeff != 0:
                terms.append(f"{coeff} * r{j}")
        code += f"    dydt[{i}] = " + (" + ".join(terms) if terms else "0") + "\n"
    code += "    return dydt\n"
    return code
