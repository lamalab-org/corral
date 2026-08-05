"""Test config for the backend suite.

Put this directory on `sys.path` so `worker_fixtures` (the picklable env
builders for the process-worker tests) is importable by a plain module name.
The insertion happens in the parent test process; `multiprocessing`'s
`spawn` start method copies the parent's `sys.path` into each worker it
spawns, so the workers can re-import the module to rebuild their environments.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
