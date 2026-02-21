## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- `uv` (recommended) or `pip` for package management

///info

   Installing `uv`
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
///


# Installation from GitHub

**1. Clone the repository**

   ```bash
   git clone https://github.com/lamalab-org/corral
   cd corral
   ```

**2. Install the framework**

Create a virtual environment

   ```bash
   uv venv
   ```

install the package

   ```bash
   uv sync
   ```

**3. Install specific environment dependencies**

Each of the folder in task is a standalone repository

   ```bash
   cd tasks/samplemath && uv venv && uv sync
   ```

**4. (Optional) Install Hydra support**

For composable configuration and multi-run sweeps:

   ```bash
   uv pip install -e ".[hydra]"
   ```

   This enables the `corral-hydra` CLI. See the [Hydra Launching guide](documentation/how_tos/hydra_launching.md) for details.
