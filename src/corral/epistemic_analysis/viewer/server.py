"""Serve the comparison viewer over loopback without extra dependencies."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import threading
import time
from collections import Counter, OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)

METHODS = ("decisions_direct", "decisions_graph", "llm_graph")
ASSETS = Path(__file__).parent
DOWNLOADS = {
    "trace": "traces",
    "snapshot": "snapshots",
    **{method: f"results/{method}" for method in METHODS},
}


def read_json(path):
    return json.loads(path.read_text())


def stamp(path):
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        with (self.root / "sample.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.rows = {row["langfuse_trace_id"]: row for row in rows}
        if len(self.rows) != len(rows) or not self.rows:
            raise ValueError(
                "The run must contain a nonempty sample with unique trace IDs"
            )
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key) for key in self.rows):
            raise ValueError("Invalid trace identifier in sample")
        self.lock = threading.RLock()
        self.index_cache = {}
        self.trace_cache = OrderedDict()
        self.catalog_cache = None
        self.catalog_time = 0

    def require_trace(self, identity):
        if identity not in self.rows:
            raise KeyError("Trace is not part of this run's sample")

    def result_path(self, identity, method):
        self.require_trace(identity)
        if method not in METHODS:
            raise KeyError("Unknown method")
        return self.root / "results" / method / f"{identity}.json"

    def catalog(self):
        with self.lock:
            if (
                self.catalog_cache is not None
                and time.monotonic() - self.catalog_time < 3
            ):
                return self.catalog_cache
            traces, counts = [], {method: Counter() for method in METHODS}
            paired = 0
            for identity, row in self.rows.items():
                statuses = {}
                for method in METHODS:
                    path = self.root / "indices" / method / f"{identity}.json"
                    if not path.exists():
                        path = self.result_path(identity, method)
                    signature = stamp(path)
                    cached = self.index_cache.get((identity, method))
                    if cached is None or cached[0] != signature:
                        status = (
                            read_json(path).get("status", "unknown")
                            if signature
                            else "pending"
                        )
                        cached = signature, status
                        self.index_cache[(identity, method)] = cached
                    statuses[method] = cached[1]
                    counts[method][cached[1]] += 1
                paired += all(value == "complete" for value in statuses.values())
                traces.append(
                    {
                        "id": identity,
                        **{
                            key: row.get(key, "")
                            for key in (
                                "environment",
                                "agent",
                                "model",
                                "task_id",
                                "level",
                                "trial",
                                "score",
                                "success",
                            )
                        },
                        "statuses": statuses,
                    }
                )
            self.catalog_cache = {
                "run": self.root.name,
                "run_path": str(self.root),
                "total": len(traces),
                "paired_complete": paired,
                "methods": {method: dict(value) for method, value in counts.items()},
                "traces": traces,
            }
            self.catalog_time = time.monotonic()
            return self.catalog_cache

    def trace(self, identity):
        self.require_trace(identity)
        paths = [self.root / "traces" / f"{identity}.json"]
        paths.extend(self.result_path(identity, method) for method in METHODS)
        signature = tuple(stamp(path) for path in paths)
        with self.lock:
            cached = self.trace_cache.get(identity)
            if cached is not None and cached[0] == signature:
                self.trace_cache.move_to_end(identity)
                return cached[1]
        original = read_json(paths[0])
        methods = {}
        annotation_messages = None
        for method, path in zip(METHODS, paths[1:], strict=True):
            if not path.exists():
                methods[method] = {"status": "pending"}
                continue
            data = read_json(path)
            result = {
                "status": data.get("status", "unknown"),
                "error": data.get("error"),
                "labels": data.get("labels", {}),
                "model": data.get("configuration", {}).get(
                    "llm_model" if method == "llm_graph" else "decisions_model", ""
                ),
            }
            graph = data.get("graph")
            view = (
                graph.get("annotation_view", {})
                if graph
                else data.get("annotation_view", {})
            )
            if "messages" in view:
                if annotation_messages is None:
                    annotation_messages = view["messages"]
                result["view_length"] = len(view["messages"])
                result["clipped"] = {
                    str(mapping["msg_idx"]): mapping["omitted_chars"]
                    for mapping in view.get("mappings", [])
                    if mapping.get("omitted_chars")
                }
            if method == "decisions_direct":
                result["windows"] = [
                    {
                        "start": window["start"],
                        "end": window["end"],
                        "assessments": window["report"].get("assessments", []),
                    }
                    for window in data.get("windows", [])
                ]
            if graph:
                result.update(
                    nodes=graph["nodes"],
                    edges=graph["edges"],
                    trace_checks=graph.get("trace_checks", []),
                    trace_checks_status=graph.get("trace_checks_status", "unspecified"),
                    warnings=graph.get("qc", {}).get("warnings", []),
                    review_issues=graph.get("qc", {}).get("review_issues", []),
                    unlabelled_messages=graph.get("qc", {}).get(
                        "uncovered_messages", []
                    ),
                    reasoning_effort=graph.get("provenance", {}).get(
                        "reasoning_effort"
                    ),
                )
            methods[method] = result
        row = self.rows[identity]
        payload = {
            "id": identity,
            "revision": str(signature),
            "metadata": {
                key: row.get(key, "")
                for key in (
                    "environment",
                    "agent",
                    "model",
                    "task_id",
                    "level",
                    "trial",
                    "score",
                    "success",
                    "langfuse_trace_url",
                )
            },
            "messages": original["messages"],
            "message_sources": [
                {
                    key: source[key]
                    for key in ("original_role", "name", "tool_name")
                    if key in source
                }
                for source in original.get("message_sources", [])
            ],
            "annotation_messages": annotation_messages or [],
            "methods": methods,
        }
        with self.lock:
            self.trace_cache[identity] = signature, payload
            self.trace_cache.move_to_end(identity)
            while len(self.trace_cache) > 4:
                self.trace_cache.popitem(last=False)
        return payload

    def decisions(self, identity, index):
        """Expose saved classification answers for one message, including abstentions."""
        path = self.result_path(identity, "decisions_graph")
        data = read_json(path)
        graph = data.get("graph", {})
        if not 0 <= index < len(graph.get("messages", [])):
            raise KeyError("Message index outside this trace")
        prefix = f"m{index}_"
        candidates = {}
        for call in graph.get("annotation_requests", []):
            answers = call.get("response", {}).get("answers", [])
            selected = [
                answer
                for answer in answers
                if answer.get("name", "").startswith(prefix)
            ]
            if not selected:
                continue
            state = json.loads(call["request"]["input"])
            for answer in selected:
                span_key, node_type = answer["name"].rsplit("_", 1)
                candidate = state.get("candidates", {}).get(span_key, {})
                item = candidates.setdefault(
                    span_key,
                    {
                        "key": span_key,
                        "quote": candidate.get("quote", ""),
                        "kind": candidate.get("kind", ""),
                        "answers": [],
                    },
                )
                item["answers"].append(
                    {
                        "type": node_type,
                        "choice": answer.get("choice"),
                        "confidence": answer.get("confidence"),
                        "probabilities": answer.get("probabilities", []),
                    }
                )
        return {"message_index": index, "candidates": list(candidates.values())}

    def download_path(self, identity, kind):
        self.require_trace(identity)
        if kind not in DOWNLOADS:
            raise KeyError("Unknown download")
        return self.root / DOWNLOADS[kind] / f"{identity}.json"


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body, content_type, *, filename=None):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            if filename:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{filename}"'
                )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            parts = path.strip("/").split("/")
            try:
                static = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                }
                if path in static:
                    filename, kind = static[path]
                    self.send_bytes((ASSETS / filename).read_bytes(), kind)
                    return
                if path == "/api/catalog":
                    payload = store.catalog()
                elif len(parts) == 3 and parts[:2] == ["api", "trace"]:
                    payload = store.trace(parts[2])
                elif (
                    len(parts) == 5
                    and parts[:2] == ["api", "trace"]
                    and parts[3] == "decisions"
                ):
                    payload = store.decisions(parts[2], int(parts[4]))
                elif len(parts) == 4 and parts[:2] == ["api", "download"]:
                    file = store.download_path(parts[2], parts[3])
                    self.send_bytes(
                        file.read_bytes(),
                        "application/json",
                        filename=f"{parts[3]}-{parts[2]}.json",
                    )
                    return
                else:
                    raise KeyError("Unknown route")
                self.send_bytes(
                    json.dumps(payload, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            except (KeyError, FileNotFoundError, ValueError):
                self.send_error(404, "Requested annotation is not available")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                logger.exception("Viewer request failed: %s", path)
                self.send_error(500, "Could not read annotations")

        def log_message(self, fmt, *args):
            logger.info("%s %s", self.address_string(), fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler_for(RunStore(args.run))
    )
    logger.info(
        "Annotation viewer: http://127.0.0.1:%s · %s", args.port, args.run.resolve()
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
