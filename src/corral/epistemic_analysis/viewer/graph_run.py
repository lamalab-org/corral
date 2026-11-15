"""Read completed native graph jobs without importing or modifying their run."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter, OrderedDict
from pathlib import Path

FINISHED = {"complete", "requires_review"}
METHOD = "decisions_graph"
PATTERN_VERSION = "graph-motifs-and-answer-provenance-v3"


def read(path):
    return json.loads(path.read_text())


def aggregate_pattern_reports(units):
    """OR the unit matches without constructing fictitious cross-unit edges."""
    assessed = [u for u in units if u["analysis"] is not None]
    if not assessed:
        return None
    patterns = []
    for template in assessed[0]["analysis"]["patterns"]:
        matches, statuses = [], []
        for unit in units:
            result = next(
                (
                    p
                    for p in (unit["analysis"] or {}).get("patterns", [])
                    if p["name"] == template["name"]
                ),
                {},
            )
            statuses.append(result.get("status", "unassessed"))
            if result.get("status") == "matched":
                matches.append(
                    {"id": unit["id"], "label": unit["label"], "count": result["count"]}
                )
        patterns.append(
            {
                **{k: template[k] for k in ("name", "category", "description")},
                "status": "matched"
                if matches
                else "not_found"
                if all(s == "not_found" for s in statuses)
                else "unassessed",
                "units": matches,
            }
        )
    return {
        "scope": "parent_trace",
        "aggregation": "any_unit",
        "units_total": len(units),
        "units_assessed": len(assessed),
        "patterns": patterns,
        "repeated_test_units": [
            {
                "id": u["id"],
                "label": u["label"],
                "count": len(u["analysis"]["repeated_tests"]),
            }
            for u in assessed
            if u["analysis"]["repeated_tests"]
        ],
    }


def with_parent_traces(catalog, jobs, metadata):
    """Count each source trace once while exposing every independent unit."""
    available = {row["id"]: row for row in catalog["traces"]}
    parents = {}
    for identity, job in jobs.items():
        parent = job.get("parent_id") or identity
        group = parents.setdefault(parent, {**job["origin"], "id": parent, "units": []})
        row = available.get(identity)
        group["units"].append(
            {
                "id": identity,
                **metadata(identity),
                "available": row is not None,
                "statuses": row["statuses"] if row else {METHOD: "pending"},
                "pattern_hits": row.get("pattern_hits", []) if row else [],
            }
        )
    updated, partial = 0, 0
    for group in parents.values():
        units = group["units"]
        units.sort(
            key=lambda u: (
                u["scientist_unit"].get("kind") != "main",
                u["scientist_unit"].get("node_id", ""),
                u["id"],
            )
        )
        group["unit_count"] = len(units)
        group["available_unit_count"] = sum(u["available"] for u in units)
        group["updated_unit_count"] = group["available_unit_count"]
        group["updated_complete"] = group["updated_unit_count"] == len(units)
        group["pattern_hits"] = sorted(
            {name for unit in units for name in unit["pattern_hits"]}
        )
        updated += group["updated_complete"]
        partial += 0 < group["updated_unit_count"] < len(units)
        group["statuses"] = {
            METHOD: "partial"
            if group["available_unit_count"] < len(units)
            else "requires_review"
            if any(u["statuses"][METHOD] == "requires_review" for u in units)
            else "complete"
        }
    return {
        **catalog,
        "parent_trace_count": len(parents),
        "updated_parent_count": updated,
        "partial_parent_count": partial,
        "parent_traces": [
            group for group in parents.values() if group["available_unit_count"]
        ],
        "pattern_counts": dict(
            Counter(
                name for group in parents.values() for name in group["pattern_hits"]
            )
        ),
    }


class GraphRunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        request = read(self.root / "queue-request.json")
        self.profile = request["profile_fingerprint"]
        self.rows = {job["id"]: job for job in request["selection"]}
        if len(self.rows) != len(request["selection"]) or any(
            not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key) for key in self.rows
        ):
            raise ValueError("Invalid or duplicate graph job identity")
        self.lock = threading.RLock()
        self.cache = OrderedDict()
        self.pattern_cache = {}

    def pattern_analysis(self, identity):
        record = self.completed(identity)
        path = self.root / "pattern-analysis" / f"{identity}.json"
        if record is None or not path.exists():
            return None
        stat = path.stat()
        signature = record["graph_sha256"], stat.st_mtime_ns, stat.st_size
        with self.lock:
            cached = self.pattern_cache.get(identity)
            if cached and cached[0] == signature:
                return cached[1]
        raw = path.read_bytes()
        result = json.loads(raw)
        if (
            result.get("version") != PATTERN_VERSION
            or result.get("graph_sha256") != record["graph_sha256"]
        ):
            return None
        result["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.lock:
            self.pattern_cache[identity] = signature, result
        return result

    def parent_patterns(self, identity):
        parent = self.rows[identity].get("parent_id") or identity
        return aggregate_pattern_reports(
            [
                {
                    "id": key,
                    "label": self.metadata(key)["unit_label"] or "Trace",
                    "analysis": self.pattern_analysis(key),
                }
                for key, job in self.rows.items()
                if (job.get("parent_id") or key) == parent
            ]
        )

    def completed(self, identity):
        if identity not in self.rows:
            raise KeyError("Unknown graph job")
        path = self.root / "completed" / f"{identity}.json"
        if not path.exists():
            return None
        record = read(path)
        if record.get("state") not in FINISHED:
            return None
        if (
            record.get("id") != identity
            or record.get("input_fingerprint")
            != self.rows[identity]["input_fingerprint"]
            or record.get("profile_fingerprint") != self.profile
            or record.get("graph") != f"graphs/{identity}.json"
        ):
            raise ValueError("Completed graph identity changed")
        return record

    def metadata(self, identity):
        job = self.rows[identity]
        unit = job.get("scientist_unit") or {}
        return {
            **job["origin"],
            "parent_id": job.get("parent_id"),
            "scientist_unit": unit,
            "unit_label": "Main run"
            if unit.get("kind") == "main"
            else unit.get("node_id", "").replace("node_", "Node "),
        }

    def catalog(self):
        traces, counts = [], Counter()
        for identity in self.rows:
            record = self.completed(identity)
            if record is None:
                continue
            counts[record["state"]] += 1
            patterns = self.pattern_analysis(identity)
            traces.append(
                {
                    "id": identity,
                    **self.metadata(identity),
                    "statuses": {METHOD: record["state"]},
                    "node_count": record.get("node_count", 0),
                    "edge_count": record.get("edge_count", 0),
                    "pattern_hits": [
                        p["name"]
                        for p in patterns["patterns"]
                        if p["status"] == "matched"
                    ]
                    + (["repeated_tool_testing"] if patterns["repeated_tests"] else [])
                    if patterns
                    else [],
                }
            )
        catalog = {
            "mode": "graph_run",
            "run": self.root.name,
            "run_path": str(self.root),
            "total": len(traces),
            "total_jobs": len(self.rows),
            "methods": {METHOD: dict(counts)},
            "traces": traces,
        }
        return with_parent_traces(catalog, self.rows, self.metadata)

    def graph(self, identity):
        record = self.completed(identity)
        if record is None:
            raise KeyError("This graph has not finished")
        path = self.root / record["graph"]
        stat = path.stat()
        signature = (record["graph_sha256"], stat.st_mtime_ns, stat.st_size)
        with self.lock:
            cached = self.cache.get(identity)
            if cached and cached[0] == signature:
                self.cache.move_to_end(identity)
                return record, cached[1]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["graph_sha256"]:
            raise ValueError("Saved graph does not match its completion record")
        graph = json.loads(raw)
        with self.lock:
            self.cache[identity] = (signature, graph)
            self.cache.move_to_end(identity)
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        return record, graph

    def trace(self, identity):
        record, graph = self.graph(identity)
        view = graph.get("annotation_view") or {}
        qc, provenance = graph.get("qc", {}), graph.get("provenance", {})
        annotation = view.get("messages", graph["messages"])
        scope = provenance.get("graph_scope", {})
        annotation = annotation[: scope.get("message_count", len(annotation))]
        patterns = self.pattern_analysis(identity)
        parent_patterns = self.parent_patterns(identity)
        return {
            "id": identity,
            "revision": record["graph_sha256"]
            + ":"
            + hashlib.sha256(
                json.dumps([patterns, parent_patterns], sort_keys=True).encode()
            ).hexdigest(),
            "parent_patterns": parent_patterns,
            "metadata": self.metadata(identity),
            "messages": graph["messages"],
            "message_sources": graph.get("message_sources", []),
            "annotation_messages": annotation,
            "methods": {
                METHOD: {
                    "status": record["state"],
                    "model": provenance.get("model", "KnowLine"),
                    "view_length": len(annotation),
                    "clipped": {
                        str(m["msg_idx"]): m["omitted_chars"]
                        for m in view.get("mappings", [])
                        if m.get("omitted_chars")
                    },
                    "nodes": graph["nodes"],
                    "edges": graph["edges"],
                    "warnings": qc.get("warnings", []),
                    "review_issues": qc.get("review_issues", []),
                    "coverage": qc.get("context_coverage", {}),
                    "window": provenance.get("window"),
                    "overlap": provenance.get("overlap"),
                    "threshold": provenance.get("min_confidence", 0.6),
                    "graph_semantics": provenance.get("graph_semantics"),
                    "graph_scope": scope,
                    "pattern_analysis": patterns,
                    "trace_checks": [],
                    "trace_checks_status": "not_requested",
                }
            },
        }

    def decisions(self, identity, index):
        _, graph = self.graph(identity)
        if not 0 <= index < len(graph["messages"]):
            raise KeyError("Invalid message index")
        end = (
            graph.get("provenance", {})
            .get("graph_scope", {})
            .get(
                "message_count",
                len(
                    graph.get("annotation_view", {}).get("messages", graph["messages"])
                ),
            )
        )
        if index >= end:
            return {"candidates": [], "outside_graph_scope": True}
        gpu = self.rows[identity]["gpu"]
        if type(gpu) is not int:
            raise ValueError("Invalid artifact replica")
        artifacts = self.root / "artifacts" / f"gpu-{gpu}"
        candidates = {}
        for call in graph.get("annotation_requests", []):
            names = [
                key
                for key in call.get("questions", [])
                if re.fullmatch(rf"m{index}_s\d+_[HTEJCNF]", key)
            ]
            if not names:
                continue
            for key in ("exchange", "state_ref"):
                if not re.fullmatch(r"[0-9a-f]{64}", call[key]):
                    raise ValueError("Invalid native request reference")
            exchange = read(artifacts / "exchanges" / f"{call['exchange']}.json")
            saved_state = read(artifacts / "states" / f"{call['state_ref']}.json")
            for name in names:
                span, kind = name.rsplit("_", 1)
                candidate = saved_state.get("candidates", {}).get(span)
                answer = exchange.get("response", {}).get("answers", {}).get(name)
                if candidate is None or answer is None:
                    continue
                item = candidates.setdefault(
                    span, {**candidate, "id": span, "answers": []}
                )
                probabilities = answer.get("probabilities", {})
                item["answers"].append(
                    {
                        "type": kind,
                        "choice": answer["choice"],
                        "confidence": probabilities.get(answer["choice"]),
                        "native_confidence": answer.get("confidence"),
                        "probabilities": [
                            {"value": k, "probability": v}
                            for k, v in probabilities.items()
                        ],
                    }
                )
        return {"candidates": list(candidates.values())}

    def download_path(self, identity, kind):
        record, _ = self.graph(identity)
        if kind == METHOD:
            return self.root / record["graph"]
        if kind == "patterns" and self.pattern_analysis(identity) is not None:
            return self.root / "pattern-analysis" / f"{identity}.json"
        if kind in {"trace", "snapshot"}:
            job = self.rows[identity]
            path = Path(job["input"])
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != job["input_file_sha256"]
            ):
                raise ValueError("Source snapshot changed")
            return path
        raise KeyError("Unknown download")
