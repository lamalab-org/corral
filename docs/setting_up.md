## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- `uv` (recommended) or `pip` for package management

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/lamalab-org/mat-agent-bench.git
   cd mat-agent-bench
   ```

2. **Install the framework**

   ```bash
   uv pip install -e .
   ```

3. **Install specific environment dependencies**

   ```bash
   # create task environments
   cd tasks/samplemath && uv venv && uv pip install -e .  # create an env for running sample math
   # ... repeat for other tasks as needed
   ```
