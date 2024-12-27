# Modal app

The modal app is a simple app that runs the simulation functions in the modal servers.

## Deployment

Deploy the modal app only needs of a simple bash command.

```bash
modal deploy app.py
```

If the process is successful, the app will be available online.

## Usage

The app is a simple interface that allows the user to run the simulation functions in the modal servers.
To run the different functions, first it is needed to define them:

```python
import modal

calculate_lattice_energy = modal.Function.lookup("simagent", "calculate_lattice_energy")
```

Then, the function can be called with the desired parameters:

```python
structure_file = "path/to/your/structure.cif"

# Call the function
energy = calculate_lattice_energy.remote(structure_file)
```
