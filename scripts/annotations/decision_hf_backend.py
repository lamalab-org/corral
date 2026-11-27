"""Bounded prefill and exact-prefix reuse for the local HF benchmark servers."""

from __future__ import annotations

import copy
import time


class HFScorer:
    """Keep one immutable prefix cache; copy it before each question's suffix.

    The cache is reused only for identical token IDs, including the template.
    Chunked causal prefill bounds temporary activations without removing tokens.
    Both costs and reused tokens are reported, so warm latency is distinguishable
    from the first request's full prefill.
    """

    def __init__(self, model, *, max_tokens=131072, chunk_tokens=4096):
        import torch

        self.torch = torch
        self.model = model
        self.max_tokens = max_tokens
        self.chunk_tokens = chunk_tokens
        self.prefix_ids = []
        self.prefix_cache = None
        self.prefix_bytes = 0

    def new_cache(self):
        """Construct Gemma4's native sliding/full cache explicitly.

        Transformers 5.12 slices layer_types[:-num_kv_shared_layers]. With zero
        shared layers, [:-0] is empty and silently creates full-history caches
        for every layer. Explicit native layer types retain the same attention
        semantics without retaining keys outside each layer's trained window.
        Other architectures keep their model-provided cache implementation.
        """
        config = self.model.config.get_text_config()
        if config.model_type != "gemma4_text":
            return None
        from transformers.cache_utils import (
            Cache,
            DynamicLayer,
            DynamicSlidingWindowLayer,
        )

        if config.num_kv_shared_layers:
            raise ValueError(
                "This Gemma4 reference backend requires unshared KV layers"
            )
        return Cache(
            layers=[
                DynamicSlidingWindowLayer(config)
                if kind == "sliding_attention"
                else DynamicLayer(config)
                for kind in config.layer_types
            ]
        )

    def clear(self):
        self.prefix_cache = None
        self.prefix_ids = []
        self.prefix_bytes = 0

    def validate(
        self,
        tokenizer,
        *,
        input_ids=None,
        selected_ids=None,
        prefix_length=4500,
        probability_transform=None,
    ):
        """Compare full, chunked, and cache-hit reads on a multi-chunk prompt."""
        torch = self.torch
        ids = (
            list(input_ids)
            if input_ids is not None
            else tokenizer.encode(
                "A measurement was repeated and independently checked.\n" * 700,
                add_special_tokens=False,
            )[:5000]
        )
        selected = (
            list(selected_ids)
            if selected_ids is not None
            else [
                tokenizer.encode(label, add_special_tokens=False)[-1]
                for label in ("A", "B", "C")
            ]
        )
        if not (
            self.chunk_tokens < len(ids) <= self.max_tokens
            and 0 < prefix_length < len(ids)
        ):
            raise ValueError(
                "Validation must exercise multiple prefill chunks and a suffix"
            )

        def probabilities(values):
            values = torch.as_tensor(values, dtype=torch.float32).cpu()
            if probability_transform is not None:
                return torch.tensor(probability_transform(values.tolist()))
            return values.softmax(-1)

        with torch.inference_mode():
            batch = torch.tensor([ids], device=self.model.device)
            full_logits = (
                self.model(input_ids=batch, use_cache=False, logits_to_keep=1)
                .logits[0, -1, selected]
                .float()
                .cpu()
            )
        full = probabilities(full_logits)
        first, _ = self.score(ids, selected, prefix_length=prefix_length)
        second, metrics = self.score(ids, selected, prefix_length=prefix_length)
        fresh = probabilities(first)
        cached = probabilities(second)
        delta = float((full - fresh).abs().max())
        cache_delta = float((fresh - cached).abs().max())
        self.clear()
        torch.cuda.empty_cache()
        result = {
            "tokens": len(ids),
            "full_probabilities": full.tolist(),
            "chunked_probabilities": fresh.tolist(),
            "cached_probabilities": cached.tolist(),
            "max_probability_delta": delta,
            "cache_hit_delta": cache_delta,
            "cache_hit_verified": metrics["prefix_cache_hit"],
        }
        if delta > 0.02 or cache_delta > 1e-5 or not metrics["prefix_cache_hit"]:
            raise RuntimeError(f"Prefill parity check failed: {result}")
        return result

    def _forward(self, ids, cache):
        torch = self.torch
        if cache is None:
            cache = self.new_cache()
        output = None
        for start in range(0, len(ids), self.chunk_tokens):
            batch = torch.tensor(
                [ids[start : start + self.chunk_tokens]], device=self.model.device
            )
            output = self.model(
                input_ids=batch, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
            cache = output.past_key_values
        return output, cache

    def score(self, ids, selected_ids, *, prefix_length=0):
        torch = self.torch
        if not ids or len(ids) > self.max_tokens:
            raise ValueError(
                f"Context overflow: {len(ids)} > {self.max_tokens}; no truncation"
            )
        prefix_length = min(max(0, prefix_length), len(ids) - 1)
        prefix = ids[:prefix_length]
        hit = (
            bool(prefix) and prefix == self.prefix_ids and self.prefix_cache is not None
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                if not hit:
                    self.clear()
                    if prefix:
                        baseline = torch.cuda.memory_allocated()
                        output, self.prefix_cache = self._forward(prefix, None)
                        del output
                        self.prefix_ids = prefix
                        self.prefix_bytes = max(
                            0, torch.cuda.memory_allocated() - baseline
                        )
                # A fixed token cutoff needlessly penalizes small models and
                # repeatedly prefills long evidence. Share one memory policy:
                # retain an immutable copy only if it leaves 8 GiB for forward
                # activations and allocator variation. Otherwise consume this
                # prefix once, still preserving every input token.
                free_bytes, _ = torch.cuda.mem_get_info()
                available = (
                    free_bytes
                    + torch.cuda.memory_reserved()
                    - torch.cuda.memory_allocated()
                )
                retain_prefix = (
                    bool(prefix) and self.prefix_bytes + 8 * 2**30 <= available
                )
                cache_bytes = self.prefix_bytes
                if retain_prefix:
                    cache = copy.deepcopy(self.prefix_cache)
                else:
                    cache = self.prefix_cache
                    self.clear()
                output, cache = self._forward(ids[prefix_length:], cache)
                logits = output.logits[0, -1].float()
                values = [float(logits[i]) for i in selected_ids]
                del output, cache, logits
            torch.cuda.synchronize()
            metrics = {
                "native_prompt_tokens": len(ids),
                "forward_s": time.perf_counter() - started,
                "prefix_cache_hit": hit,
                "prefix_cache_retained": retain_prefix,
                "prefix_cache_bytes": cache_bytes,
                "cache_memory_headroom_bytes": 8 * 2**30,
                "reused_prompt_tokens": prefix_length if hit else 0,
                "processed_prompt_tokens": len(ids) - (prefix_length if hit else 0),
                "prefill_chunk_tokens": self.chunk_tokens,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            return values, metrics
        except Exception:
            self.clear()
            torch.cuda.empty_cache()
            raise


def shared_prefix_length(tokenizer, prefix_text, full_ids):
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    count = 0
    for left, right in zip(prefix, full_ids, strict=False):
        if left != right:
            break
        count += 1
    # Keep the template boundary in the suffix; comparison still uses actual IDs.
    return max(0, count - 8)
