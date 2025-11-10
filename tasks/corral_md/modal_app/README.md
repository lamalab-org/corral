# Modal App

The Modal app allows you to run computationally intensive simulation functions on [Modal's](https://modal.com) cloud servers.

## Quick Setup

1. **Create a Modal account** at [modal.com](https://modal.com)
2. **Authenticate**: Run `modal setup` (Modal is already in project dependencies)
3. **Optional**: Set app name via `SIMAGENT_NAME` environment variable

For detailed Modal setup instructions, see the [Modal documentation](https://modal.com/docs/guide).

## Deployment

Deploy your Modal app:

```bash
modal deploy app.py
# or for LAMMPS
modal deploy lammps_app.py
```

## Using the `@app.function` Decorator in Corral

The `@app.function` decorator defines functions that run on Modal's cloud infrastructure.

### Basic Example

```python
from modal import App, Image

app = App("my-app-name")


@app.function(image=Image.debian_slim().pip_install("numpy"))
def compute_something(data: str) -> float:
    """A function that runs on Modal's servers."""
    import numpy as np

    return np.sum([1, 2, 3, 4, 5])
```

### Corral-Specific: Using `modal_tool`

For Corral environments, use the `modal_tool` decorator to create tools that execute in the cloud:

```python
from corral.utils.modal import modal_tool, MODAL_TOOL_REGISTRY
from modal import App, Image

app = App("my-tools")


@modal_tool(app=app, image=Image.debian_slim().pip_install("rdkit"), memory=1024)
def calculate_mol_weight(smiles: str) -> float:
    """Calculate molecular weight in the cloud."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)


# Tool is automatically registered
tool = MODAL_TOOL_REGISTRY["calculate_mol_weight"]
```

See Modal's [function configuration docs](https://modal.com/docs/guide/function-config) for all available parameters (image, memory, cpu, gpu, timeout, secrets, volumes, etc.).

## Usage from Client Code

Call deployed Modal functions from your code:

```python
import modal

# Look up the deployed function
# Note: Both from_name() and lookup() work, this codebase uses from_name()
calculate_lattice_energy = modal.Function.from_name(
    "simagent", "calculate_lattice_energy"
)

# Call the function remotely
energy = calculate_lattice_energy.remote("structure.cif")
```

For more on calling Modal functions, see [Modal's documentation](https://modal.com/docs/guide/call-functions).
