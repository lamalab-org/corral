"""Browse saved report conversations: python3 reports/serve_traces.py.

Uses only the standard library. CSV indexes load at startup; compressed reports
are scanned only when a trial is opened, retaining at most that trial in memory.
The HTTP server serves only the browser assets and the indexed report trials.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
import re
import sys
import threading
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "site"
STRUCTURAL = re.compile(r'["\\{}\[\]]')
PRIMITIVE_END = re.compile(r"[\s,\]}]")
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


class JSONStream:
    """Read or skip a JSON value without buffering its enclosing document."""

    def __init__(self, source, chunk_size=64 * 1024):
        self.source = source
        self.chunk_size = chunk_size
        self.buffer = ""
        self.pos = 0

    def _fill(self):
        if self.pos == len(self.buffer):
            self.buffer = self.source.read(self.chunk_size)
            self.pos = 0
        return bool(self.buffer)

    def peek(self):
        while self._fill():
            char = self.buffer[self.pos]
            if not char.isspace():
                return char
            self.pos += 1
        raise ValueError("Unexpected end of report JSON")

    def expect(self, expected):
        if self.peek() != expected:
            raise ValueError(f"Expected {expected!r} in report JSON")
        self.pos += 1

    def value(self, capture=True):
        first = self.peek()
        parts = []
        # Primitives end at whitespace or an enclosing array/object delimiter.
        if first not in '{["':
            while self._fill():
                match = PRIMITIVE_END.search(self.buffer, self.pos)
                end = match.start() if match else len(self.buffer)
                if capture:
                    parts.append(self.buffer[self.pos : end])
                self.pos = end
                if match:
                    break
        else:
            string_value = first == '"'
            inside_string = string_value
            depth = 0 if string_value else 1
            escaped = False
            start = self.pos
            self.pos += 1
            while True:
                if escaped and self.pos < len(self.buffer):
                    self.pos += 1
                    escaped = False
                match = STRUCTURAL.search(self.buffer, self.pos)
                if not match:
                    if capture:
                        parts.append(self.buffer[start:])
                    self.pos = len(self.buffer)
                    if not self._fill():
                        raise ValueError("Incomplete value in report JSON")
                    start = self.pos
                    continue
                char = match.group()
                self.pos = match.end()
                if inside_string:
                    if char == "\\":
                        if self.pos == len(self.buffer):
                            escaped = True
                        else:
                            self.pos += 1
                    elif char == '"':
                        inside_string = False
                        if string_value:
                            break
                elif char == '"':
                    inside_string = True
                elif char in "{[":
                    depth += 1
                elif char in "}]":
                    depth -= 1
                    if depth == 0:
                        break
            if capture:
                parts.append(self.buffer[start : self.pos])
        return json.loads("".join(parts)) if capture else None

    def object_keys(self):
        self.expect("{")
        if self.peek() == "}":
            self.expect("}")
            return
        while True:
            key = self.value()
            self.expect(":")
            yield key
            if self.peek() == "}":
                self.expect("}")
                return
            self.expect(",")


def read_trial(path: Path, task_id: str, slot: int) -> dict:
    """Locate one trial, skipping all other values in bounded chunks."""
    with path.open("rb") as source:
        if source.read(len(LFS_POINTER)) == LFS_POINTER:
            raise FileNotFoundError(
                "This report is a Git LFS pointer. Run `git lfs pull` in the "
                "repository to download the report, then retry."
            )
    opener = (
        gzip.open
        if path.suffix == ".gz"
        else lzma.open
        if path.suffix == ".xz"
        else open
    )
    with opener(path, "rt", encoding="utf-8") as source:
        stream = JSONStream(source)
        for key in stream.object_keys():
            if key != "task_results":
                stream.value(capture=False)
                continue
            for task in stream.object_keys():
                if task != task_id:
                    stream.value(capture=False)
                    continue
                for field in stream.object_keys():
                    if field != "trials":
                        stream.value(capture=False)
                        continue
                    stream.expect("[")
                    for position in range(slot + 1):
                        if stream.peek() == "]":
                            raise LookupError(
                                "Trial position is absent from the report"
                            )
                        if position == slot:
                            trial = stream.value()
                            if not isinstance(trial, dict):
                                raise ValueError("Report trial must be an object")
                            return trial
                        stream.value(capture=False)
                        if stream.peek() == "]":
                            raise LookupError(
                                "Trial position is absent from the report"
                            )
                        stream.expect(",")
                    break
    raise LookupError("Task or trial is absent from the report")


def number(value):
    return float(value) if value not in (None, "") else None


def load_index(reports: Path):
    traces = []
    locations = {}
    availability = {}
    for index in sorted(reports.glob("*/trials.csv")):
        with index.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                report = (index.parent / row["report_file"]).resolve()
                if not report.is_relative_to(reports.resolve()):
                    raise ValueError(f"Report path escapes reports directory: {index}")
                if report not in availability:
                    if not report.is_file():
                        availability[report] = "missing"
                    else:
                        with report.open("rb") as handle:
                            availability[report] = (
                                "lfs"
                                if handle.read(len(LFS_POINTER)) == LFS_POINTER
                                else "available"
                            )
                trace_id = hashlib.sha256(
                    f"{index.parent.name}/{row['report_file']}:{row['task_id']}:{row['trial']}".encode()
                ).hexdigest()[:24]
                trace = {
                    "id": trace_id,
                    "environment": row["environment"],
                    "agent": row["agent"],
                    "model": row["model"],
                    "level": row["level"],
                    "task_id": row["task_id"],
                    "trial": int(row["trial"]),
                    "execution_id": row["execution_id"],
                    "run_id": row["run_id"],
                    "score": number(row.get("score")),
                    "success": row.get("success", "").lower() == "true"
                    if row.get("success", "").lower() in ("true", "false")
                    else None,
                    "error": row.get("error", ""),
                    "duration": number(row.get("duration_s")),
                    "input_tokens": number(row.get("input_tokens")),
                    "output_tokens": number(row.get("output_tokens")),
                    "tool_calls": number(row.get("tool_calls")),
                    "langfuse_url": row.get("langfuse_trace_url", ""),
                    "report_file": report.relative_to(reports).as_posix(),
                    "availability": availability[report],
                }
                traces.append(trace)
                locations[trace_id] = (
                    report,
                    trace["task_id"],
                    trace["trial"],
                    trace["execution_id"],
                )
    return traces, locations


def message_presentation(messages, agent):
    """Recover legacy OpenHands prompts without inventing tool/thought timing.

    The adapter used to persist MCP calls live, then append its buffered SDK
    transcript. The session discarded duplicate SDK tool messages, including
    the only anchors for interleaving thoughts with calls. Keep the saved data
    intact and return display indices; unanchored prose gets a separate group.
    """
    if agent != "openhands":
        return None

    def is_runtime(message):
        return message.get("role") == "tool" or bool(message.get("tool_calls"))

    order, untimed = [], []
    start = 0
    while start < len(messages):
        end = start + 1
        session = messages[start].get("agent_run_id")
        while end < len(messages) and messages[end].get("agent_run_id") == session:
            end += 1
        cursor = start
        while cursor < end:
            # A session already beginning with prompts is not the legacy shape.
            # Do not move a later, legitimate prompt ahead of its earlier turns.
            if messages[cursor].get("role") in {"system", "user"}:
                order.extend(range(cursor, end))
                break
            if not messages[cursor].get("tool_calls"):
                # In particular, retain late results from the preceding run.
                order.append(cursor)
                cursor += 1
                continue
            prologue = cursor
            while prologue < end and is_runtime(messages[prologue]):
                prologue += 1
            if (
                prologue + 1 >= end
                or messages[prologue].get("role") != "system"
                or messages[prologue + 1].get("role") != "user"
            ):
                order.extend(range(cursor, prologue))
                cursor = prologue
                continue
            tail = prologue + 2
            while (
                tail < end
                and messages[tail].get("role") == "assistant"
                and not is_runtime(messages[tail])
            ):
                tail += 1
            # Leave the closing response and harness errors at the end, where
            # they were saved. Earlier commentary has no recoverable timing.
            closing = tail
            while (
                closing > prologue + 2
                and messages[closing - 1].get("name") == "openhands-error"
            ):
                closing -= 1
            if closing > prologue + 2 and not messages[closing - 1].get("name"):
                closing -= 1
            order.extend(range(prologue, closing))
            untimed.extend(range(prologue + 2, closing))
            order.extend(range(cursor, prologue))
            order.extend(range(closing, tail))
            cursor = tail
        start = end
    if order == list(range(len(messages))):
        return None
    return {"order": order, "untimed": untimed}


class TraceStore:
    def __init__(self, reports):
        self.traces, self.locations = load_index(reports)
        self.agents = {trace["id"]: trace["agent"] for trace in self.traces}
        self.index = json.dumps({"traces": self.traces}, ensure_ascii=False).encode()
        self.cache = OrderedDict()
        self.cache_bytes = 0
        # Serializing report scans bounds peak memory even during rapid navigation.
        self.lock = threading.Lock()

    def trial(self, trace_id):
        if trace_id not in self.locations:
            raise LookupError("Unknown trace")
        path, task, slot, execution_id = self.locations[trace_id]
        stat = path.stat()
        key = (trace_id, stat.st_mtime_ns, stat.st_size)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            trial = read_trial(path, task, slot)
            if trial.get("trial_id") != execution_id:
                raise ValueError(
                    "The CSV index and report disagree about this trial. "
                    "Rebuild the index with reports/build_index.py and restart the browser."
                )
            presentation = message_presentation(
                trial.get("messages") or [], self.agents[trace_id]
            )
            if presentation:
                trial["_trace_browser"] = presentation
            payload = gzip.compress(
                json.dumps(trial, ensure_ascii=False).encode(), compresslevel=1
            )
            limit = 32 * 1024 * 1024
            while self.cache and self.cache_bytes + len(payload) > limit:
                _, previous = self.cache.popitem(last=False)
                self.cache_bytes -= len(previous)
            if len(payload) <= limit:
                self.cache[key] = payload
                self.cache_bytes += len(payload)
            return payload


class TraceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(head=False)

    def do_HEAD(self):
        self.respond(head=True)

    def respond(self, head=False):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/api/traces":
                self.send_payload(
                    self.server.store.index, "application/json", head=head
                )
            elif path.startswith("/api/traces/"):
                payload = self.server.store.trial(path.removeprefix("/api/traces/"))
                if "gzip" in self.headers.get("Accept-Encoding", ""):
                    self.send_payload(
                        payload, "application/json", head=head, compressed=True
                    )
                else:
                    self.send_payload(
                        gzip.decompress(payload), "application/json", head=head
                    )
            else:
                assets = {
                    "/": ("report-traces.html", "text/html"),
                    "/report-traces.html": ("report-traces.html", "text/html"),
                    "/report-traces.css": ("report-traces.css", "text/css"),
                    "/report-traces.js": ("report-traces.js", "text/javascript"),
                    "/favicon.ico": ("corral_logo.png", "image/png"),
                }
                if path not in assets:
                    raise LookupError("Not found")
                name, content_type = assets[path]
                self.send_payload((ASSETS / name).read_bytes(), content_type, head=head)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (FileNotFoundError, LookupError) as error:
            self.send_error_json(str(error), HTTPStatus.NOT_FOUND, head)
        except (ValueError, OSError, EOFError) as error:
            self.send_error_json(
                f"Unable to read report: {error}", HTTPStatus.UNPROCESSABLE_ENTITY, head
            )

    def send_error_json(self, message, status, head):
        self.send_payload(
            json.dumps({"error": message}).encode(), "application/json", status, head
        )

    def send_payload(
        self, data, content_type, status=HTTPStatus.OK, head=False, compressed=False
    ):
        self.send_response(status)
        self.send_header(
            "Content-Type",
            content_type
            + ("; charset=utf-8" if not content_type.startswith("image/") else ""),
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        if compressed:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def log_message(self, fmt, *args):
        # Avoid logging trial IDs and task content; errors still appear in the UI.
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        default=ROOT / "reports",
        help="Reports root containing */trials.csv",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open", action="store_true", help="Open the browser automatically"
    )
    args = parser.parse_args()
    store = TraceStore(args.reports.resolve())
    if not store.traces:
        parser.error(f"No trials.csv indexes found under {args.reports}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), TraceHandler)
    server.daemon_threads = True
    server.store = store
    url = f"http://127.0.0.1:{server.server_port}"
    sys.stdout.write(
        f"Corral trace browser · {len(store.traces):,} trials\n{url}\nPress Ctrl+C to stop.\n"
    )
    sys.stdout.flush()
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
