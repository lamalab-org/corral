"""Lightweight annotation server for human review of LLM trace annotations.

Run from reasoning_reports/human_annotation/app/:
    python serve.py [--port PORT] [--old]

Opens at http://localhost:8000
"""

from __future__ import annotations

import http.server
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import fire
from loguru import logger as logging

APP_DIR = Path(__file__).resolve().parent
HUMAN_ANNOTATION_DIR = APP_DIR.parent
REASONING_DIR = HUMAN_ANNOTATION_DIR.parent

# Configurable via --old flag
_config = {"files_dir_name": "files2annotate", "annotation_prefix": "annotations"}


class AnnotationHandler(http.server.SimpleHTTPRequestHandler):
    """Serves static files from app/ and exposes JSON API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/config":
            return self._json_response(
                {
                    "old": _config["files_dir_name"] == "oldfiles2annotate",
                    "prefix": _config["annotation_prefix"],
                }
            )
        if self.path == "/api/list-files":
            return self._json_response(self._list_annotated_files())
        if self.path == "/api/list-annotations":
            return self._json_response(self._list_annotations())
        if self.path.startswith("/api/annotated/"):
            name = unquote(self.path[len("/api/annotated/") :])
            return self._serve_json(
                HUMAN_ANNOTATION_DIR / _config["files_dir_name"] / name
            )
        if self.path.startswith("/api/trace/"):
            rel = unquote(self.path[len("/api/trace/") :])
            # Some annotated files store input_file with a
            # "corral/reasoning_reports/" prefix; strip it so the path
            # resolves correctly relative to REASONING_DIR.
            for prefix in ("corral/reasoning_reports/",):
                if rel.startswith(prefix):
                    rel = rel[len(prefix) :]
                    break
            return self._serve_json(REASONING_DIR / rel)
        if self.path.startswith("/api/annotation/"):
            name = unquote(self.path[len("/api/annotation/") :])
            return self._serve_json(HUMAN_ANNOTATION_DIR / name)
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            name = re.sub(r"[^a-zA-Z0-9_-]", "", body.get("name", ""))
            if not name:
                return self._json_response({"error": "Invalid name"}, 400)
            out = HUMAN_ANNOTATION_DIR / f"{_config['annotation_prefix']}_{name}.json"
            out.write_text(json.dumps(body.get("data", {}), indent=2))
            return self._json_response({"status": "ok", "filename": out.name})
        self.send_error(405)
        return None

    def _json_response(self, data, status=200):
        blob = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _serve_json(self, path: Path):
        resolved = path.resolve()
        if not str(resolved).startswith(str(REASONING_DIR)):
            self.send_error(403, "Forbidden")
            return
        if not resolved.is_file():
            self.send_error(404, "Not found")
            return
        blob = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    @staticmethod
    def _list_annotated_files():
        d = HUMAN_ANNOTATION_DIR / _config["files_dir_name"]
        return sorted(f.name for f in d.glob("*.annotated.json")) if d.exists() else []

    @staticmethod
    def _list_annotations():
        return sorted(
            f.name
            for f in HUMAN_ANNOTATION_DIR.glob(f"{_config['annotation_prefix']}_*.json")
        )

    # silence per-request logging unless error
    def log_message(self, fmt, *args):
        if int(args[1]) >= 400 if len(args) > 1 else True:
            super().log_message(fmt, *args)


def main(port: int = 8000, old: bool = False):
    if old:
        _config["files_dir_name"] = "oldfiles2annotate"
        _config["annotation_prefix"] = "old_annotations"

    max_attempts = 100
    for _attempt in range(max_attempts):
        try:
            server = http.server.HTTPServer(("", port), AnnotationHandler)
            break
        except OSError:
            logging.warning(f"Port {port} is in use, trying {port + 1}...")
            port += 1
    else:
        logging.error(f"Could not find a free port after {max_attempts} attempts.")
        sys.exit(1)
    logging.info(f"Annotation server running at http://localhost:{port}")
    logging.info(f"  App dir:    {APP_DIR}")
    logging.info(f"  Annotations:{HUMAN_ANNOTATION_DIR}")
    logging.info(f"  Files dir:  {_config['files_dir_name']}")
    logging.info(f"  Prefix:     {_config['annotation_prefix']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    fire.Fire(main)
