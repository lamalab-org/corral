from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoTokenizer

from corral.utils.tool_helpers import (
    DEFAULT_CHEMICAL_EMBEDDING_MODEL,
    DEFAULT_CHEMICAL_EMBEDDING_MODEL_REVISION,
    embed_text,
)


def test_default_chemical_embedding_model_uses_immutable_revision(monkeypatch):
    calls = {}

    class FakeTokenizer:
        def __call__(self, chunks, **kwargs):
            calls["tokenize"] = (chunks, kwargs)
            return {"input_ids": torch.tensor([[1]])}

    class FakeModel:
        def __call__(self, **inputs):
            calls["inference"] = inputs
            return SimpleNamespace(pooler_output=torch.tensor([[1.0, 2.0]]))

    def load_tokenizer(model, **kwargs):
        calls["tokenizer"] = (model, kwargs)
        return FakeTokenizer()

    def load_model(model, **kwargs):
        calls["model"] = (model, kwargs)
        return FakeModel()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(AutoModel, "from_pretrained", load_model)

    result = embed_text(["CCO"], model=DEFAULT_CHEMICAL_EMBEDDING_MODEL, chemical=True)

    assert result == [[1.0, 2.0]]
    assert calls["tokenizer"] == (
        DEFAULT_CHEMICAL_EMBEDDING_MODEL,
        {
            "trust_remote_code": True,
            "revision": DEFAULT_CHEMICAL_EMBEDDING_MODEL_REVISION,
        },
    )
    assert calls["model"] == (
        DEFAULT_CHEMICAL_EMBEDDING_MODEL,
        {
            "deterministic_eval": True,
            "trust_remote_code": True,
            "revision": DEFAULT_CHEMICAL_EMBEDDING_MODEL_REVISION,
        },
    )
    assert calls["tokenize"] == (
        ["CCO"],
        {"padding": True, "truncation": True, "return_tensors": "pt"},
    )
    assert calls["inference"]["input_ids"].tolist() == [[1]]
