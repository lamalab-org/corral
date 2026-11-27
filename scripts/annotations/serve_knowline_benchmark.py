"""Run the pinned official KnowLine front end with its HF backend on one GPU.

Only the last-position logits are materialized (the official backend materializes
every position then selects the last). This preserves the decision readout and
avoids allocating sequence_length * vocabulary_size logits for long traces.
Native token counts, CUDA time, and peak memory are included in each response.
"""

from __future__ import annotations

# CLI progress is intentionally written to stdout.
# ruff: noqa: T201
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from decision_hf_backend import HFScorer, shared_prefix_length


def install_annotation_endpoints(
    vendor, path, context_tokens, revision, *, backend_metadata=None, encode_text=None
):
    """Expose native rendering/token counts without changing the scoring protocol."""
    original_get, original_post = vendor.Handler.do_GET, vendor.Handler.do_POST

    def metadata(handler):
        engine = handler.server.engine
        return {
            **({"inference_backend": backend_metadata} if backend_metadata else {}),
            "revision": revision,
            "context_tokens": context_tokens,
            "model": str(path),
            "scoring": {
                "temperature": engine.T,
                "temperatures": engine.T_by_type,
                "layout": vars(engine.layout) if engine.layout else None,
                "readout": "native-choice-softmax",
            },
            "source_hashes": {
                name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                for name in (
                    "knowline_server.py",
                    "knowline_layout.py",
                    "tokenizer.json",
                    "temperatures.json",
                )
                if (path / name).exists()
            },
        }

    def get(handler):
        if handler.path == "/annotation-metadata":
            return handler._send(200, metadata(handler))
        return original_get(handler)

    def post(handler):
        if handler.path != "/tokenize":
            return original_post(handler)
        try:
            size = int(handler.headers.get("Content-Length", 0))
            if not 0 < size <= vendor.REQUEST_MAX_BYTES:
                handler.close_connection = True
                return handler._send(413, {"error": "Invalid token request size"})
            body = json.loads(handler.rfile.read(size))
            if body.get("assets"):
                raise ValueError("Annotation token preflight is text-only")
            engine = handler.server.engine
            _, prompts, media = engine._render(body["state"], body["questions"], False)
            if media:
                raise ValueError("Annotation token preflight is text-only")
            tokenizer = getattr(engine.processor, "tokenizer", engine.processor)
            counts = {
                name: len(
                    encode_text(prompt[0])
                    if encode_text
                    else tokenizer.encode(prompt[0], add_special_tokens=False)
                )
                for name, prompt in prompts.items()
            }
            return handler._send(200, {"count": max(counts.values()), "counts": counts})
        except (ValueError, KeyError, TypeError) as exc:
            return handler._send(422, {"error": str(exc)})

    vendor.Handler.do_GET, vendor.Handler.do_POST = get, post


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=4096,
        help="Tokens processed per forward pass; smaller chunks reduce peak memory.",
    )
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument(
        "--revision", help="Pinned model revision; defaults to snapshot directory name."
    )
    args = parser.parse_args()
    if args.prefill_chunk_tokens < 1:
        parser.error("--prefill-chunk-tokens must be positive")
    path = args.model_path.resolve()
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "knowline_official", path / "knowline_server.py"
    )
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    install_annotation_endpoints(
        vendor, path, args.max_tokens, args.revision or path.name
    )
    original_init = vendor.HF.__init__

    def init(self, *a, **kw):
        start = time.perf_counter()
        original_init(self, *a, **kw)
        self.load_seconds = time.perf_counter() - start
        self.measurements = []
        if not self.torch.cuda.is_available() or self.model.device.type != "cuda":
            raise RuntimeError(
                "Benchmark requires a CUDA device; CPU fallback is disabled"
            )
        self.model.config.use_cache = False
        self.scorer = HFScorer(
            self.model,
            max_tokens=args.max_tokens,
            chunk_tokens=args.prefill_chunk_tokens,
        )
        self.prefix_text = ""
        validation = self.scorer.validate(self.tok)
        if args.validation_json:
            args.validation_json.write_text(json.dumps(validation, indent=2))
        print(
            f"PREFILL_VALIDATED delta={validation['max_probability_delta']}", flush=True
        )
        print(f"MODEL_LOADED seconds={self.load_seconds:.3f}", flush=True)

    def score(self, text, media, ids):
        if media:
            raise ValueError("This benchmark is text-only")
        tokens = self.tok.encode(text, add_special_tokens=False)
        if len(tokens) > args.max_tokens:
            raise ValueError(
                f"Context overflow: {len(tokens)} > {args.max_tokens}; no truncation"
            )
        with self.lock:
            values, metrics = self.scorer.score(
                tokens,
                ids,
                prefix_length=shared_prefix_length(self.tok, self.prefix_text, tokens),
            )
            self.measurements.append(metrics)
        return values, len(tokens)

    original_run = vendor.KnowLine.run

    def run(self, *a, **kw):
        from knowline_layout import render_request

        self.backend.measurements.clear()
        state, questions = a[:2]
        rendered = render_request(
            {"state": state, "questions": questions},
            self.processor,
            self.layout,
            labels=self.labels,
        )
        self.backend.prefix_text = rendered["prefix"]
        result = original_run(self, *a, **kw)
        # Official front end returns (answers, usage).
        answers, usage = result
        usage["benchmark"] = {
            "backend": "transformers-bf16-chunked-prefill-prefix-cache",
            "load_s": self.backend.load_seconds,
            "forwards": list(self.backend.measurements),
            "max_tokens": args.max_tokens,
            "truncated": False,
        }
        return answers, usage

    vendor.HF.__init__ = init
    vendor.HF.score = score
    vendor.KnowLine.run = run
    sys.argv = [
        str(path / "knowline_server.py"),
        "--model",
        str(path),
        "--backend",
        "hf",
        "--device",
        "cuda:0",
        "--workers",
        "1",
        "--port",
        str(args.port),
        "--host",
        "127.0.0.1",
    ]
    if (path / "temperatures.json").exists():
        sys.argv += ["--temperatures", str(path / "temperatures.json")]
    vendor.main()


if __name__ == "__main__":
    main()
