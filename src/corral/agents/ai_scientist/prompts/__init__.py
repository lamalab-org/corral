"""Packaged prompt loading with deliberately small templating semantics."""

from functools import cache, lru_cache
from importlib.resources import files


@cache
def load_prompt(name: str) -> str:
    """Load a template before entering a restricted worker identity."""
    return files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")


def render_prompt(name: str, **values: object) -> str:
    prompt = load_prompt(name)
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", str(value))
    return prompt


__all__ = ["render_prompt"]
