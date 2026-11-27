"""Serve a checks export and named graph runs through one loopback port."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assets = config.get("runtime")
    if not assets:
        # Run snapshots may place this file directly under scripts/.
        script_dir = Path(__file__).resolve().parent
        root = (
            script_dir.parents[1]
            if script_dir.name == "annotations"
            else script_dir.parent
        )
        assets = root / "src/corral/epistemic_analysis/viewer"
    sys.path.insert(0, str(assets))
    from benchmark import ReviewStore, handler_for
    from graph_run import GraphRevisionStore, GraphRunStore
    from server import handler_for as graph_handler_for

    runs = []
    for entry in config["graph_runs"]:
        store = (
            GraphRevisionStore(entry["root"], entry["previous"])
            if entry.get("previous")
            else GraphRunStore(entry["root"])
        )
        runs.append((store, entry["url"], entry["label"]))

    class NamedStore:
        def __init__(self, underlying):
            self.underlying = underlying

        def __getattr__(self, name):
            return getattr(self.underlying, name)

        def catalog(self):
            result = self.underlying.catalog()
            navigation = []
            for raw, url, label in runs:
                selected = raw is self.underlying
                catalog = result if selected else raw.catalog()
                navigation.append(
                    {
                        "url": url,
                        "label": label,
                        "current": selected,
                        "trace_ids": [row["id"] for row in catalog["traces"]],
                    }
                )
                if selected:
                    result["viewer_label"] = label
            result["viewer_navigation"] = navigation
            return result

    default = next(raw for raw, url, _ in runs if url == "/graphs/")
    handlers = {
        url.rstrip("/"): graph_handler_for(NamedStore(raw), base_path=url.rstrip("/"))
        for raw, url, _ in runs
        if url != "/graphs/"
    }
    review = ReviewStore(Path(config["review_run"]), Path(config["review_database"]))
    token = Path(config["token_file"]).read_text().strip()
    if not token:
        raise ValueError("Missing review token")

    class Handler(handler_for(review, token, graph_store=NamedStore(default))):
        def do_GET(self):
            path = urlsplit(self.path).path
            for mount, handler in handlers.items():
                if path == mount or path.startswith(mount + "/"):
                    if not self.allowed_origin():
                        return self.send(
                            {"error": "Local same-origin access required"}, status=403
                        )
                    return handler.do_GET(self)
            return super().do_GET()

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Annotation viewers listening on 127.0.0.1:{args.port}", flush=True)  # noqa: T201
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
