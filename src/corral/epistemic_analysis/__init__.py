"""Epistemic graph annotation and trace checks, independent of trace providers."""

__all__ = [
    "AnnotatedGraph",
    "AnnotationBackend",
    "AnnotationConfig",
    "DecisionsAnnotator",
    "EpistemicReport",
    "GroupingMetadata",
    "MotifAssessment",
    "NormalizedTrace",
    "SourceReference",
    "annotate_trace",
    "read_graph",
]


def __getattr__(name):
    # Checks-only use must not initialize graph or live-agent integrations.
    from importlib import import_module

    modules = {
        "AnnotationBackend": "annotators.base",
        "annotate_trace": "pipeline",
        "DecisionsAnnotator": "decisions",
        "EpistemicReport": "decisions",
        "MotifAssessment": "decisions",
    }
    if name in __all__:
        value = getattr(
            import_module(f"{__name__}.{modules.get(name, 'schema')}"), name
        )
        globals()[name] = value
        return value
    raise AttributeError(name)
