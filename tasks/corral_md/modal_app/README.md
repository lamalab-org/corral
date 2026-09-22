# Modal App

The Modal app allows you to run computationally intensive simulation functions on [Modal's](https://modal.com) cloud servers.

## Quick Setup

1. **Create a Modal account** at [modal.com](https://modal.com)
2. **Authenticate**: Run `modal setup` (Modal is already in project dependencies)

For detailed Modal setup instructions, see the [Modal documentation](https://modal.com/docs/guide).

## MD worker setup and deployment

From `tasks/corral_md`, using the project environment:

```bash
uv run python modal_app/release.py
```

This one command verifies the bundled ZIP archives and the exact MACE
checkpoints in [assets.json](assets.json), checks SHA-256, uploads missing
assets, publishes a versioned workspace seed in the `corral-md-bases` Volume,
mounts that Volume read-only in workers, deploys the worker under a
single stable `simagent` app, and runs LAMMPS plus GPU Python smoke tests that
check asset permissions and workspace isolation. The release command prints the
internal release ID after all steps pass; set it as `CORRAL_MD_RELEASE_ID` when
running Corral. Repeating the command
skips identical assets and refuses to overwrite conflicting ones. Assets are
stored in `corral-md-<asset-kind>-<content-hash>` Volumes, mounted read-only at
the same paths in every worker. Changing a checkpoint or potential creates a
new asset Volume; unchanged asset sets are shared across releases. The base
manifest and local release record include the exact asset Volume names. Keep
legacy `models`, `potentials`, and `structures` Volumes for older deployments.
Use
`uv run python modal_app/release.py --print-id` to inspect the release ID
without deployment. A release ID covers the asset manifest, dependency lock,
worker source, client protocol, and workspace seed. Redeploying `simagent`
replaces the development worker, so discard active development runs before a
redeploy and freeze the app while real runs are in progress.
Deployment also calculates and publishes the fixed Task 3 dimer and Task 5
strained-silicon references under the release ID. Live verification results are
cached by their complete scientific inputs and release, so repeated scoring does
not rerun identical calculator, descriptor, or checkpoint-prediction jobs.
If using `CORRAL_MD_MODAL_VOLUME`, set it for deployment. The app reports the
resulting Volume name and each execution records it for restart.

Each execution has a persistent directory under `/corral/runs/<run_id>/` in the
`simulations` Volume. The initializer creates it from the pinned release base
on the first remote tool call after `ExecutionStarted`. Local and remote seeds
contain `input/`, `scripts/`, and `output/`. Before each LAMMPS or Python call, the bridge
uploads changed files and removes locally deleted files. The worker copies the
current remote workspace into a fresh directory under
`attempts/<action_id>/<attempt_id>/workspace`. It stages those files in a
separate ephemeral Volume and runs agent code in a Modal Sandbox at `/workspace`.
Only that task Volume and the shared read-only assets are mounted in the Sandbox;
`/results`, `/bases`, evaluation files and controller credentials are not exposed.
The process drops to an unprivileged user before running the command. Once the
Sandbox stops and commits its Volume, the controller validates and collects the
writable files and commits a result manifest only on success. The bridge downloads changed
outputs and atomically publishes them locally. Failed attempts retain their
diagnostics under the remote run and do not replace the last successful result.

All MD file-tool paths must be canonical absolute paths beginning with
`/workspace/`. For example, `/workspace/output/result.json` is accepted, while
`output/result.json`, `./output/result.json`, parent traversal, symbolic links
and host paths are rejected. Shared assets live at `/workspace/potentials`,
`/workspace/models` and `/workspace/structures`. They are accessed separately
and are excluded from task synchronization and workspace snapshots.

Python and the MD terminal now use Modal for CPU execution too; they never run
agent code as a subprocess of the local controller. `use_gpu=True` reserves an
A100 inside the Sandbox. The default working directory is `/workspace` and
LAMMPS logs and captured stdout/stderr are saved under `/workspace/output`.
`timeout` and an absolute `working_dir` apply to both CPU and GPU Python calls.

Absolute-path validation applies to structured tool arguments. Inside Python or
shell code, relative system calls still work relative to the sandbox's working
directory. The sandbox also exposes its own runtime libraries, devices and OS
files; it does not expose the controller or other tasks. This is not a claim that
all paths outside `/workspace` are invisible to arbitrary code.

The action ID and Modal FunctionCall ID are saved in a local journal beside the
workspace; the worker also records its call ID in the Volume. Restarting a
pending action reattaches to that call, or uses its durable result manifest if
the worker already finished. The local workspace is also captured by Corral's
normal workspace snapshots at tool completion. A failed output sync or local
snapshot raises `ToolRecoveryPending` and leaves Corral's action running;
retrying the execution resumes that action before the agent can issue new
work. This also applies when an SDK adapter catches the interruption. A
definitively cancelled, timed-out, expired, or failed infrastructure call is
marked for a new attempt on the next recovery, using the same action ID and
the saved inputs. Transport failures retain the existing call ID until its
outcome is known. Simulation/input errors with a durable failure manifest are
reported as tool failures so the agent can correct the input. Every attempt's
diagnostics remain available. A process crash in the tiny
interval after dispatch and before either call ID has been recorded leaves an
ambiguous dispatch; recovery reports this explicitly and waits for the worker
record rather than launching a second untracked call. Modal completed-call
results expire, so the Volume manifest is the durable recovery source.

Remote directories are retained through completion or cancellation. After the
execution is confirmed closed, mark it with
`uv run python modal_app/manage_runs.py close <run_id> completed` (or
`cancelled`). Set `CORRAL_MD_MODAL_VOLUME` to the run's pinned Volume name if
using a custom Volume. Closed runs remain recoverable for 30 days. List eligible runs
with `uv run python modal_app/manage_runs.py prune`; add `--execute` to delete
them. Active runs and release bases are never pruned by that command. A run ID
is the local Corral workspace directory name. Keep a backup of Corral's local
workspace snapshots and `.corral-modal` journals for recovery after local host
loss. Long simulation progress within a failed attempt requires simulation
restart files and a continuation input; the bridge recovers completed tools.

To prepare and inspect inputs locally without accessing Modal:

```bash
uv run python modal_app/setup_assets.py --output-dir /tmp/corral-md-assets
```

CPU and GPU Sandboxes mount models read-only at:

- `/workspace/models/teacher.model`: original medium MACE-MP-0,
  also used to generate proxy labels in the fine-tuning task.
- `/workspace/models/student.model`: medium MACE-MP-0b, the fine-tuning starting
  checkpoint. Use the trained checkpoint for that task's subsequent MD.

Use `mace_mp(model=<absolute path>, device="cuda", default_dtype="float64",
dispersion=False)`. Do not rely on the changing default of `mace_mp()`.
The [upstream checkpoint mapping](https://github.com/ACEsuit/mace/blob/v0.3.13/mace/calculators/foundations_models.py)
identifies these two releases. Model binaries are downloaded during setup and
are not committed to this repository.

The worker pins Python 3.12.11, LAMMPS `stable_22Jul2025` by full commit hash,
and all resolved Python dependencies in [requirements.txt](requirements.txt).
To deliberately regenerate the dependency lock, from `modal_app` run:

```bash
uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --exclude-newer 2025-09-01 -o requirements.txt
```

The manifest and actual installed package versions are retained in the image
at `/opt/corral-md/assets.json` and `/opt/corral-md/installed-packages.txt`.
This records the runtime baseline used for benchmark jobs and release-owned
fixed-reference generation. Fixed references are stored under the release ID;
record the generation script, random seeds, manifest and installed-package list
when changing that process. OS packages, GPU drivers and stochastic
trajectories are not made bit-for-bit reproducible by these pins.

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

## Verification workers and controlled dynamics

The same release now exposes `verify_calculations`, `verify_md_provenance`, and
`run_verified_md`. Calculations reuse the pinned scientific image, assets and
A100 Sandboxes. `torch-dftd==0.5.1` adds the dispersion calculation required for
silver teacher labels; it is locked separately against the base runtime in
`verification_requirements.txt`. Regenerate that lock with:

```bash
uv pip compile verification_requirements.in --constraint requirements.txt \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --exclude-newer 2025-09-01 -o verification_requirements.txt
```

The image uses Python 3.12.11 on Bookworm and pins Modal's image builder to
`2025.06` for this app. This avoids legacy account defaults that inject
dependencies incompatible with Python 3.12; no account-wide setting is changed.
LAMMPS is explicitly installed under `/usr/local`. Agent sandboxes mount shared
assets outside the writable Volume and expose the usual `/workspace` asset paths
through root-owned links protected by a sticky workspace directory. This avoids
nested Volume mounts and prevents the agent from replacing the public links.

Verification Sandboxes block external network access and expose only their
selected inputs, read-only models/potentials, and a private output Volume.
SW-only batches run separately with an IPv4 loopback allowlist so Open MPI's
local listener can start. Checkpoints, MACE, SOAP and controlled dynamics retain
fully disabled networking.
Submitted checkpoints are evaluated separately from trusted-model calculations.
The controlled MD worker accepts JSON parameters and runs installed code, so its
record can attest actual model usage, steps, and continuous state transitions.
The controller stores that record with the immutable successful action, outside
any agent mount. Ordinary agent scripts retain their existing execution path.

Enable grading with `CORRAL_MD_VERIFICATION=modal`. See the
[scoring documentation](../src/corral_md/workflow_scoring/README.md#independent-modal-verification)
for the API, CLI, evidence fields, and limits. `run_verified_md` is exposed as an
MD tool and shares normal action replay, output synchronization and recovery.
