# Modal App

The Modal app allows you to run computationally intensive simulation functions on Modal's cloud servers. This is particularly useful for tasks that require significant computational resources or specific dependencies.

## Prerequisites

Before you can use Modal, you need to:

1. **Create a Modal account**: Sign up at [modal.com](https://modal.com)
2. **Install Modal**: The Modal package is already included in the project dependencies
3. **Authenticate**: Set up your Modal credentials

## Setup

### 1. Install Modal CLI

Modal is already included in the project's dependencies. Verify the installation:

```bash
modal --version
```

If you need to install it separately:

```bash
pip install modal
```

### 2. Authenticate with Modal

Before deploying or using Modal functions, you must authenticate:

```bash
modal setup
```

This will open a browser window where you can log in to your Modal account. After authentication, a token will be stored locally on your machine.

Alternatively, you can set a Modal token environment variable:

```bash
export MODAL_TOKEN_ID="your-token-id"
export MODAL_TOKEN_SECRET="your-token-secret"
```

You can find your tokens in the Modal dashboard under Settings > Tokens.

### 3. Configure App Name (Optional)

By default, the app is named `simagent`. You can customize the app name using the `SIMAGENT_NAME` environment variable:

```bash
export SIMAGENT_NAME="my-custom-name"
```

This will create an app named `simagent-my-custom-name`.

## Deployment

Deploy your Modal app with a single command:

```bash
modal deploy app.py
```

For the LAMMPS app:

```bash
modal deploy lammps_app.py
```

After successful deployment, you'll see output confirming that your functions are available online:

```
✓ Created objects.
├── 🔨 Created mount /home/runner/work/corral/corral/modal_app
├── 🔨 Created function calculate_lattice_energy.
└── 🔨 Created App simagent.
```

## Using the `@app.function` Decorator

The `@app.function` decorator is how you define functions that run on Modal's cloud infrastructure. Here's how to use it:

### Basic Usage

```python
from modal import App, Image

# Create a Modal app
app = App("my-app-name")

# Define a function that runs in the cloud
@app.function(image=Image.debian_slim().pip_install("numpy"))
def compute_something(data: str) -> float:
    """A function that runs on Modal's servers.
    
    Args:
        data: Input data as a string
        
    Returns:
        Computed result as a float
    """
    import numpy as np
    # Your computation here
    return np.sum([1, 2, 3, 4, 5])
```

### Decorator Parameters

The `@app.function` decorator accepts several parameters to configure the cloud environment:

- **`image`**: Docker image configuration (required). Defines the runtime environment and dependencies.
  ```python
  image=Image.debian_slim().pip_install("rdkit", "numpy")
  ```

- **`memory`**: Memory allocation in MB (optional)
  ```python
  memory=2048  # 2GB
  ```

- **`cpu`**: Number of CPUs (optional)
  ```python
  cpu=2.0
  ```

- **`gpu`**: GPU type (optional)
  ```python
  gpu="T4"  # or "A10G", "A100"
  ```

- **`timeout`**: Maximum execution time in seconds (optional)
  ```python
  timeout=300  # 5 minutes
  ```

- **`secrets`**: Modal secrets for API keys and credentials (optional)
  ```python
  secrets=[modal.Secret.from_name("my-api-key")]
  ```

- **`volumes`**: Persistent storage volumes (optional)
  ```python
  volumes={"/data": modal.Volume.from_name("my-volume")}
  ```

### Complete Example

```python
from modal import App, Image, Secret, Volume

app = App("materials-simulation")

# Create a volume for persistent storage
potentials_volume = Volume.from_name("potentials", create_if_missing=True)

# Define a custom Docker image with your dependencies
custom_image = (
    Image.debian_slim()
    .pip_install("pymatgen", "numpy", "scipy")
    .apt_install("libgomp1")
)

@app.function(
    image=custom_image,
    memory=4096,  # 4GB memory
    cpu=2.0,      # 2 CPUs
    timeout=600,  # 10 minute timeout
    volumes={"/potentials": potentials_volume},
)
def calculate_properties(structure_file: str, potential: str) -> dict:
    """Calculate material properties using LAMMPS.
    
    Args:
        structure_file: Path to the structure file
        potential: Name of the potential to use
        
    Returns:
        Dictionary containing computed properties
    """
    # Your computation logic here
    return {"energy": -123.45, "stress": [1.0, 2.0, 3.0]}
```

## Usage from Client Code

Once deployed, you can call your Modal functions from anywhere:

### Method 1: Using Modal Function Lookup

```python
import modal

# Look up the deployed function
calculate_lattice_energy = modal.Function.lookup("simagent", "calculate_lattice_energy")

# Call the function remotely
structure_file = "path/to/your/structure.cif"
energy = calculate_lattice_energy.remote(structure_file)

print(f"Calculated energy: {energy} eV")
```

### Method 2: Using the modal_tool Decorator in Corral

In the Corral framework, you can also define Modal tools using the `modal_tool` decorator:

```python
from corral.utils.modal import modal_tool, MODAL_TOOL_REGISTRY
from modal import App, Image

app = App("my-tools")

@modal_tool(
    app=app,
    image=Image.debian_slim().pip_install("rdkit"),
    memory=1024
)
def complex_calculation(formula: str) -> float:
    """Calculate molecular properties in the cloud.
    
    Args:
        formula: Chemical formula (e.g., 'H2O')
        
    Returns:
        Calculated molecular weight
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(formula)
    return Chem.Descriptors.MolWt(mol)

# Access the tool from the registry
tool = MODAL_TOOL_REGISTRY["complex_calculation"]
```

## Managing Modal Resources

### Viewing Deployed Apps

```bash
modal app list
```

### Viewing Logs

```bash
modal app logs simagent
```

### Stopping an App

```bash
modal app stop simagent
```

## Working with Volumes

Modal volumes allow you to persist data between function calls:

```python
from modal import Volume

# Create or reference a volume
volume = Volume.from_name("my-data", create_if_missing=True)

# Use the volume in a function
@app.function(volumes={"/data": volume})
def process_data(filename: str):
    # Read from volume
    with open(f"/data/{filename}", "r") as f:
        data = f.read()
    
    # Write to volume
    with open(f"/data/output.txt", "w") as f:
        f.write(processed_data)
```

You can upload files to a volume:

```python
with volume.batch_upload() as batch:
    batch.put_directory("./local_dir/", "/")
```

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:

```bash
# Re-authenticate
modal token new
modal setup
```

### Import Errors

Make sure all dependencies are installed in your Modal image:

```python
image = Image.debian_slim().pip_install("package1", "package2")
```

### Timeout Errors

Increase the timeout parameter in your function decorator:

```python
@app.function(timeout=1800)  # 30 minutes
```

## Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Python SDK Reference](https://modal.com/docs/reference)
- [Modal Pricing](https://modal.com/pricing)
