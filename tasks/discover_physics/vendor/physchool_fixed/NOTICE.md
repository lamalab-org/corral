# Vendored copy of `physchool`

Source: https://github.com/SampsonML/DiscoverPhysics, `PhysicsSchool/physchool/`,
commit `33b7fa9df96de9c35744efd181ca7e5a8dd60ad5`. MIT licensed (see upstream
`LICENSE`).

## Why this is vendored instead of a plain git dependency

Upstream's `PhysicsSchool/setup.py` uses `find_packages()`, and
`physchool/worlds/` has no `__init__.py`. `find_packages()` silently drops any
subdirectory without one, so a normal (non-editable) install — which is what
`uv`/`pip` do for a `git = ...` dependency — ships only the empty top-level
`physchool/__init__.py`; `physchool.worlds` (the actual simulator code:
`field_sampler.py`, `nbody_sampler.py`, `force_laws.py`, `analytic_orbits.py`,
`utils.py`) is missing entirely, and `import physchool.worlds.field_sampler`
fails (this also breaks `scienceagent`, which imports from
`physchool.worlds.field_sampler` itself).

`pip install -e PhysicsSchool/` (upstream's own documented install command)
works around this by accident: editable/develop-mode installs never call
`find_packages()` to select what to copy, they just add the source tree to
`sys.path`, so `physchool.worlds` resolves as an implicit PEP 420 namespace
package straight from source. `uv` has no equivalent for a *git* dependency,
though (`editable = true` is rejected outright alongside `git = ...` in
`[tool.uv.sources]` — editable only applies to `path` sources).

**The only change from upstream is the addition of
`physchool/worlds/__init__.py` (empty, matching every other `__init__.py` in
this tree)** so a normal wheel build includes the subpackage. Everything else
is byte-for-byte upstream.

If upstream fixes this (adds the missing `__init__.py`), this vendored copy
can be dropped and `physchool` switched back to a plain
`{ git = "https://github.com/SampsonML/DiscoverPhysics", subdirectory = "PhysicsSchool" }`
dependency, matching `scienceagent` (which is unaffected by this bug and is
not vendored).
