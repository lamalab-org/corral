# How to control the level of tool verbosity in docstrings

**Goal**: Create tools that provide different levels of documentation detail for ablation studies.

**When to use this**: You want to study how tool documentation affects agent performance.


Corral's **verbosity system** allows fine-grained control over tool description detail, enabling ablation studies on the effect of tool documentation quality.

## 1. Verbosity Levels

For this feature to work the user should write the tool with docsting for the tool in a particular syntax.
Corral lets you tag different *types* of information in tool docstrings. For example `[BRIEF] Generate a chemical formula from a SMILES string using RDKit. [/BRIEF]` is a brief description of the tool while `[DETAILED] This function takes a SMILES representation of a molecule and generates its chemical formula in Hill notation (C, H, then alphabetical order). It uses RDKit to parse the SMILES string and calculate the molecular formula. If the SMILES string is invalid or cannot be parsed, it returns an error message. [/DETAILED]` is a detailed description of the tool. Similarly user can add information related to the following tag and later choose what information should be available to the agent.


```bash
`[BRIEF]` - What the tool does (1-2 sentences)
`[DETAILED]` - How the tool works (technical details)
`[PROCEDURAL]` - When to use the tool (decision guidance)
`[WORKFLOW_INTEGRATION]` - How it fits with other tools
`[SYNTACTICAL]` - Syntax details and formats
`[EXAMPLES]` - Usage examples
```



## 2. Example of tool with different verbostiy

This is an example of docstring for a tool, where all the information are tagged with different tags. Writing  docstring for a tool in such detail would help us to later control the level of information we will show to the agent

```python
from corral.backend.tool import tool


@tool
def get_formula_from_smiles(smiles: str) -> str:
    """[BRIEF] Generate a chemical formula from a SMILES string using RDKit. [/BRIEF]

    [DETAILED] This function takes a SMILES representation of a molecule and generates its chemical formula in Hill notation (C, H, then alphabetical order). It uses RDKit to parse the SMILES string and calculate the molecular formula. If the SMILES string is invalid or cannot be parsed, it returns an error message. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you have a SMILES string and need to obtain the chemical formula of the corresponding molecule.
    - When you want to validate the chemical structure of a proposed molecule.
    - When you need to convert a SMILES representation into a chemical formula for further analysis or reporting.
    - Recommended for tasks that require chemical formula generation from SMILES strings. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have the SMILES representation of a molecule that you think can produce the analysis results described in the task. Use the tools `carbon_nmr_spectra`, `proton_nmr_spectra`, `ir_spectra`, `hsqc_nmr_spectra`, and `mass_spectrometry_spectra` for obtaining the different spectra. Use the tools `retrieve_protons_shifts`, `retrieve_aromatic_protons_shifts` and `retrieve_carbon_shifts` for having more info about the chemical shifts. [/PREREQUISITE]
    2. [CURRENT] Apply this tool with a valid SMILES string to generate the chemical formula of the molecule. [/CURRENT]
    3. [FOLLOW_UP] Use the generated chemical formula to compare your proposed molecule with the molecular analysis in the task (MS results). [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It uses RDKit to parse the provided SMILES string and create a molecular object.
    - If the SMILES string is valid, it calculates the molecular formula using `rdMolDescriptors.CalcMolFormula`.
    - The formula is returned in Hill notation, which lists carbon (C) atoms first, followed by hydrogen (H) atoms, and then other elements in alphabetical order.
    - If the SMILES string is invalid or cannot be parsed, it returns an error message indicating the issue. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_formula_from_smiles("CCO")`,
        `get_formula_from_smiles("C1=CC=CC=C1")`,
        `get_formula_from_smiles("C(C(=O)O)N")`,
        `get_formula_from_smiles("C1=CC=C(C=C1)C(=O)O")`,
        `get_formula_from_smiles("C1=CC=CC=C1C(=O)O")`,
    ]
    [/SYNTACTICAL]

    Args:
        smiles (str):
            [ARGS_BRIEF] SMILES representation of a molecule [/ARGS_BRIEF]
            [ARGS_DETAILED] The SMILES string representing the chemical structure of the molecule. It should be a valid SMILES notation that RDKit can parse. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Valid SMILES string [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "CCO", "C1=CC=CC=C1", "C(C(=O)O)N", "C1=CC=C(C=C1)C(=O)O" [/ARGS_EXAMPLES]

    Returns:
        str:
            [ARGS_BRIEF] The chemical formula in Hill notation (C, H, then alphabetical) [/ARGS_BRIEF]
            [ARGS_DETAILED] The chemical formula of the molecule represented by the SMILES string, formatted in Hill notation. If the SMILES string is invalid or cannot be parsed, it returns an error message. [/ARGS_DETAILED]
            [ARGS_EXAMPLES] '"C2H6O" for ethanol, "C6H6" for benzene, "C2H5NO" for acetic acid amide, "C7H6O3" for salicylic acid' [/ARGS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If the SMILES string is invalid or cannot be parsed by RDKit. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when the SMILES string cannot be parsed by RDKit, indicating that it is not a valid SMILES representation of a molecule. It can occur due to syntax errors or unsupported structures in the SMILES string. [/ERROR_DETAILS]
            [ERROR_RECOVERY] If the SMILES string is invalid, try to provide a valid SMILES string. You can validate the SMILES using the `validate_smiles` tool. Otherwise do not try to solve the error. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - It may return "Invalid SMILES string" if the provided SMILES cannot be parsed.
        - RDKit may remove certain chemical features or properties during sanitization, which can sometimes lead to unexpected results.
    [/LIMITATIONS]
    """
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return "Invalid SMILES string"
        return rdMolDescriptors.CalcMolFormula(mol)

    except Exception as e:
        return f"Error: {e!s}"
```


///info
Also note that for Argument description also one should write multiple level of description but contained within [ARGS_<tag>] [/ARGS_<tag>]
///

## 3.  Different verbosity levels available

Following are the recognized tags. The verbosity increases as we go down (meaning, `full` verbosity would show all information. `workflow` on the other hand would show all information tagged with all other tags up unitl `workflow` tag, i.e. `brief`, `detailed`, `procedural`, `contextual`)

```python
from corral.router.verbosity import ToolVerbosity

# Available levels (increasing detail):
ToolVerbosity.MINIMAL  # Tool name + basic description
ToolVerbosity.BRIEF  # + [BRIEF] sections
ToolVerbosity.DETAILED  # + [DETAILED] sections
ToolVerbosity.PROCEDURAL  # + [PROCEDURAL] sections (when to use)
ToolVerbosity.CONTEXTUAL  # + [CONTEXTUAL] sections
ToolVerbosity.WORKFLOW  # + [WORKFLOW_INTEGRATION] sections
ToolVerbosity.SYNTACTICAL  # + [SYNTACTICAL] sections
ToolVerbosity.COMPREHENSIVE  # + [RAISES], [LIMITATIONS], [EXAMPLES]
ToolVerbosity.FULL  # Complete original docstring
```



## 4. Test different verbosity levels

`CorralRunner.bench()` method can set the verbosity level with which benchmarking would happen.

```python
from corral.run import CorralRunner
from corral.router import CorralRouter
from corral.agents import ReActAgent

interface = CorralRouter(base_url="http://localhost:8000")

# Test with brief verbosity
agent = ReActAgent(model="openai/gpt-4o")
runner = CorralRunner(interface, agent)

result_minimal = runner.bench(
    task_ids=["search_task"], tool_verbosity="brief", trials_per_task=3
)

# Test with full verbosity
result_full = runner.bench(
    task_ids=["search_task"], tool_verbosity="comprehensive", trials_per_task=3
)

print(f"Minimal: {result_minimal.average_score():.2f}")
print(f"Full: {result_full.average_score():.2f}")
```

**Done**: You can now compare performance across verbosity levels.
