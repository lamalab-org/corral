"""Packaged prompt loading with deliberately small templating semantics."""

from importlib.resources import files


def render_prompt(name: str, **values: object) -> str:
    prompt = files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", str(value))
    return prompt


__all__ = ["render_prompt"]
