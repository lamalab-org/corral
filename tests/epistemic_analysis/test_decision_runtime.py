"""Reject checkpoint configurations that cannot retain the saved architecture."""

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_native_validation_preserves_probability_tolerance(monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    from decision_hf_backend import HFScorer

    class Model:
        device = "cpu"

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[[0.0, 0.0]]]))

    scorer = HFScorer(Model(), chunk_tokens=2)
    scorer.score = lambda *args, **kwargs: ([0.0, 0.1], {"prefix_cache_hit": True})
    with pytest.raises(RuntimeError, match="Prefill parity check failed"):
        scorer.validate(
            None,
            input_ids=[1] * 6,
            selected_ids=[0, 1],
            prefix_length=4,
            probability_transform=lambda values: (
                torch.tensor(values).softmax(-1).tolist()
            ),
        )

    with pytest.raises(ValueError, match="multiple prefill chunks"):
        scorer.validate(None, input_ids=[1] * 6, selected_ids=[0, 1], prefix_length=0)
