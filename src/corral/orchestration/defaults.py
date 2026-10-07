"""Agent-dependent defaults resolved before task execution."""


def resolve_max_iterations(value: int | None, agent: object) -> int:
    """Preserve explicit budgets; otherwise allow 100 AI Scientist calls or 50.

    Accept CLI names or live instances without importing optional agent SDKs.
    Class ancestry also gives AI Scientist subclasses the same default.
    """
    if value is not None:
        return value
    names = (
        (agent,)
        if isinstance(agent, str)
        else (base.__name__ for base in type(agent).__mro__)
    )
    for name in names:
        normalized = "".join(char for char in name.casefold() if char.isalnum())
        if normalized in {"aiscientist", "aiscientistagent"}:
            return 100
    return 50
