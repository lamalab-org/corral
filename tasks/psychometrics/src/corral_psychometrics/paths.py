"""Resolve task identities and data-root-relative paths."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

#: ``gen_l<level>_t<number>_<slug>.py``
GENERATOR_NAME = re.compile(r"gen_l(?P<level>\d+)_t(?P<number>\d+)_(?P<slug>.+)\.py")

#: Optional data-root override. this is useful if we install without the generated data.
# we can also think about moving all data to HF
ROOT_VARIABLE = "CORRAL_PSYCHOMETRICS_ROOT"

_GENERATORS = "generators"


@cache
def _checkout_root() -> Path | None:
    """Return the source checkout when available."""
    # src/corral_psychometrics/paths.py -> src/corral_psychometrics -> src -> root
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else None


def task_root(explicit: str | Path | None = None) -> Path:
    """Select an explicit, configured, or checkout data root."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if configured := os.environ.get(ROOT_VARIABLE):
        return Path(configured).expanduser().resolve()

    if (checkout := _checkout_root()) is not None:
        return checkout

    raise RuntimeError(
        "cannot locate the psychometrics data. It is generated, not installed "
        f"with the package, so set ${ROOT_VARIABLE} or pass an explicit "
        "data root."
    )


@dataclass(frozen=True)
class Task:
    """A task and its files under one data root."""

    level: int
    number: int
    slug: str
    root: Path

    def at(self, root: str | Path | None) -> Task:
        """The same task, under a different data root."""
        return replace(self, root=task_root(root))

    @property
    def label(self) -> str:
        """Short name for build output and failure reports."""
        return f"l{self.level}t{self.number:02d} {self.slug}"

    @property
    def generator(self) -> str:
        """Importable module that builds this task."""
        return f"{__package__}.{_GENERATORS}.level_{self.level}.{self.stem}"

    @property
    def stem(self) -> str:
        return f"gen_l{self.level}_t{self.number:02d}_{self.slug}"

    @property
    def artifacts(self) -> Path:
        """The public files, copied into the agent's workspace."""
        return self.root / "artifacts" / f"level_{self.level}" / f"task_{self.number:02d}"

    @property
    def truth(self) -> Path:
        """The answer key, read only by the scorer."""
        return self.root / "truth" / f"level_{self.level}" / f"task_{self.number:02d}.json"

    @property
    def definition(self) -> Path:
        """The generated task definition."""
        return tasks_json_dir(self.level, root=self.root) / f"task_{self.number:02d}.json"

    @property
    def artifacts_relpath(self) -> str:
        """Recorded in a definition as ``scoring_params.data_dir``."""
        return self.relpath(self.artifacts)

    @property
    def truth_relpath(self) -> str:
        """Recorded in a definition as ``scoring_params.truth_path``."""
        return self.relpath(self.truth)

    def relpath(self, path: Path) -> str:
        """The data-root-relative form that task definitions record."""
        return path.relative_to(self.root).as_posix()


def tasks_json_dir(level: int, *, root: str | Path | None = None) -> Path:
    """Where one level's task definitions are written."""
    return task_root(root) / "environments" / f"level_{int(level)}" / "tasks_json"


def resolve(recorded: str | Path, *, root: str | Path | None = None) -> Path:
    """Turn a path recorded in a task definition back into a real one."""
    recorded = Path(recorded)
    if recorded.is_absolute() or ".." in recorded.parts:
        raise ValueError(f"task definitions record data-root-relative paths: {recorded}")
    base = task_root(root)
    resolved = (base / recorded).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"task path escapes the data root: {recorded}")
    return resolved


@cache
def _identities() -> tuple[tuple[int, int, str], ...]:
    """Every task, read off the generator filenames, in level and number order."""
    package = Path(__file__).resolve().parent / _GENERATORS
    found = [
        (int(m["level"]), int(m["number"]), m["slug"])
        for path in package.glob("level_*/gen_l*_t*.py")
        if (m := GENERATOR_NAME.match(path.name))
    ]
    return tuple(sorted(found))


def tasks(
    *,
    level: int | None = None,
    numbers: set[int] | None = None,
    root: str | Path | None = None,
) -> list[Task]:
    """Every task, optionally narrowed to one level or a set of numbers."""
    resolved = task_root(root)
    return [
        Task(level=item_level, number=number, slug=slug, root=resolved)
        for item_level, number, slug in _identities()
        if (level is None or item_level == level) and (numbers is None or number in numbers)
    ]


def find(level: int, number: int, *, root: str | Path | None = None) -> Task:
    """The one task with this numbering."""
    for item in tasks(level=level, numbers={number}, root=root):
        return item
    raise LookupError(f"no psychometrics task for level {level} number {number}")


def task(generator_file: str | Path, *, root: str | Path | None = None) -> Task:
    """The task a generator builds. Called as ``paths.task(__file__)``."""
    name = Path(generator_file).name
    match = GENERATOR_NAME.match(name)
    if match is None:
        raise ValueError(f"not a psychometrics generator filename: {name}")
    return find(int(match["level"]), int(match["number"]), root=root)


__all__ = [
    "GENERATOR_NAME",
    "ROOT_VARIABLE",
    "Task",
    "find",
    "resolve",
    "task",
    "task_root",
    "tasks",
    "tasks_json_dir",
]
