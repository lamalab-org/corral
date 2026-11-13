"""Local review of frozen decision benchmarks; standard-library-only server.

Source traces and predictions are read-only. SQLite transactions retain every
review revision and prediction reveal. Exports never overwrite human-review.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import re
import secrets
import sqlite3
import statistics
import threading
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

try:
    from corral.epistemic_analysis.review_projection import apply_review
except ModuleNotFoundError:
    # Standalone viewer deployments copy this standard-library-only helper.
    from review_projection import apply_review

logger = logging.getLogger(__name__)

ASSETS = Path(__file__).parent
STATUSES = ("present", "absent", "uncertain", "not_applicable")
VERDICTS = ("unreviewed", "correct", "incorrect", "unsure")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def reduce_labels(labels):
    if "present" in labels:
        return "present"
    if "uncertain" in labels:
        return "uncertain"
    return "not_applicable" if all(s == "not_applicable" for s in labels) else "absent"


class Conflict(ValueError):
    pass


class ReviewStore:
    def __init__(self, root, database=None):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        self.questions = self.manifest["protocol"]["questions"]
        self.rows = {r["id"]: r for r in self.manifest["selection"]}
        self.models = list(self.manifest["models"])
        for identifier in [*self.rows, *self.models, *self.questions]:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
                raise ValueError("Unsafe identifier in benchmark manifest")
        self.plans = {}
        self.hashes = {"manifest.json": digest(self.root / "manifest.json")}
        for identity in self.rows:
            for directory in ("plans", "traces"):
                name = f"{directory}/{identity}.json"
                self.hashes[name] = digest(self.root / name)
            plan = read_json(self.root / "plans" / f"{identity}.json")
            self.plans[identity] = []
            for unit in plan["units"]:
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", unit["id"]):
                    raise ValueError("Unsafe unit identifier")
                self.plans[identity].append(
                    {
                        **{k: v for k, v in unit.items() if k != "slices"},
                        "slices": [s["id"] for s in unit["slices"]],
                    }
                )
        self.code_assessments = {}
        self.code_checks = set(self.manifest.get("code_checks", []))
        if self.code_checks:
            path = self.root / "code-assessments.json"
            self.hashes[path.name] = digest(path)
            code = read_json(path)
            if code["policy_version"] != self.manifest["code_policy_version"]:
                raise ValueError("Code assessment policy differs from manifest")
            for identity, assessment in code["traces"].items():
                if assessment["trace_sha256"] != self.hashes.get(
                    f"traces/{identity}.json"
                ):
                    raise ValueError("Code assessment trace fingerprint differs")
                if {u["unit_id"] for u in assessment["units"]} != {
                    u["id"] for u in self.plans[identity]
                }:
                    raise ValueError("Code assessment unit coverage differs")
                for checks in [
                    assessment["checks"],
                    *[u["checks"] for u in assessment["units"]],
                ]:
                    if {c["name"] for c in checks} != self.code_checks or any(
                        c.get("decision_source") != "code"
                        or c.get("model_called") is not False
                        or c.get("status") not in self.questions[c["name"]]["criteria"]
                        for c in checks
                    ):
                        raise ValueError("Invalid code assessment")
                self.code_assessments[identity] = assessment
            if set(self.code_assessments) != set(self.rows):
                raise ValueError("Code assessments do not cover every frozen trace")
        self.supplemental_runs = []
        assigned = set(self.code_checks)
        for index, entry in enumerate(self.manifest.get("supplemental_runs", [])):
            path = (self.root / entry["path"]).resolve()
            signature = digest(path / "manifest.json")
            if signature != entry["manifest_sha256"]:
                raise ValueError("Supplemental manifest fingerprint differs")
            extra = read_json(path / "manifest.json")
            names = set(entry["checks"])
            if (
                not names
                or names & assigned
                or names != set(extra["protocol"]["questions"])
            ):
                raise ValueError("Invalid supplemental check selection")
            if any(
                extra["protocol"]["questions"][n] != self.questions.get(n)
                for n in names
            ):
                raise ValueError("Supplemental rubric differs from the review")
            if extra["models"] != self.manifest["models"]:
                raise ValueError("Supplemental model revisions differ")
            if {r["id"] for r in extra["selection"]} != set(self.rows) or any(
                any(
                    r.get(k) != self.rows[r["id"]].get(k)
                    for k in ("trace_sha256", "plan_sha256")
                )
                for r in extra["selection"]
            ):
                raise ValueError("Supplemental frozen evidence differs")
            assigned.update(names)
            self.hashes[f"supplemental/{index}/manifest.json"] = signature
            self.supplemental_runs.append({"path": path, "checks": names})
        self.fingerprint = hashlib.sha256(
            json.dumps(self.hashes, sort_keys=True).encode()
        ).hexdigest()
        self.lock = threading.RLock()
        self.cache = OrderedDict()
        self.assistant_cache = None
        self.database = (
            Path(database) if database else self.root / "annotations/reviews.sqlite3"
        )
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS reviews (
                    trace_id TEXT, unit_id TEXT, check_name TEXT, version INTEGER, data TEXT,
                    PRIMARY KEY (trace_id, unit_id, check_name));
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY, trace_id TEXT, unit_id TEXT, check_name TEXT,
                    kind TEXT, at TEXT, data TEXT);
                CREATE TABLE IF NOT EXISTS annotation_reviews (
                    trace_id TEXT, unit_id TEXT, check_name TEXT, source_id TEXT,
                    version INTEGER, data TEXT,
                    PRIMARY KEY (trace_id, unit_id, check_name, source_id));
            """)
            prior = db.execute(
                "SELECT value FROM metadata WHERE key='fingerprint'"
            ).fetchone()
            if prior and prior[0] != self.fingerprint:
                raise ValueError(
                    "This review database belongs to different traces or rubric. Use --database with a new path."
                )
            db.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('fingerprint', ?)",
                (self.fingerprint,),
            )
        keyfile = self.root / "model-key.json"
        self.aliases = (
            read_json(keyfile)
            if keyfile.exists()
            else {m: chr(65 + i) for i, m in enumerate(self.models)}
        )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def key(self, trace_id, unit_id, check):
        if trace_id not in self.rows:
            raise KeyError("Unknown trace")
        if unit_id not in {u["id"] for u in self.plans[trace_id]}:
            raise KeyError("Unknown review unit")
        if check not in self.questions:
            raise KeyError("Unknown check")
        return trace_id, unit_id, check

    def raw_trace(self, identity):
        if identity not in self.rows:
            raise KeyError("Unknown trace")
        with self.lock:
            path = self.root / "traces" / f"{identity}.json"
            signature = (path.stat().st_mtime_ns, path.stat().st_size)
            cached = self.cache.get(identity)
            if cached is None or cached[0] != signature:
                if digest(path) != self.hashes[f"traces/{identity}.json"]:
                    raise Conflict(
                        "Source trace changed since opening the benchmark. Restart with a new review database."
                    )
                cached = (signature, read_json(path))
                self.cache[identity] = cached
            self.cache.move_to_end(identity)
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
            return cached[1]

    def records(self, identity=None):
        with self.connect() as db:
            rows = db.execute(
                "SELECT data FROM reviews" + (" WHERE trace_id=?" if identity else ""),
                (identity,) if identity else (),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get_review(self, trace_id, unit_id, check):
        key = self.key(trace_id, unit_id, check)
        with self.connect() as db:
            row = db.execute(
                "SELECT data FROM reviews WHERE trace_id=? AND unit_id=? AND check_name=?",
                key,
            ).fetchone()
        return (
            json.loads(row[0])
            if row
            else {
                "trace_id": trace_id,
                "unit_id": unit_id,
                "check": check,
                "version": 0,
                "stage": "unreviewed",
                "initial": None,
                "final_status": "",
                "rationale": "",
                "reviewer": "",
                "evidence": [],
                "suggested_status": "",
                "suggestion": "",
                "predictions_revealed": False,
                "draft_initial_status": "",
            }
        )

    def catalog(self):
        reviews = self.records()
        approved = Counter(r["trace_id"] for r in reviews if r["stage"] == "approved")
        initial = Counter(r["trace_id"] for r in reviews if r.get("initial"))
        traces = []
        for identity, row in self.rows.items():
            traces.append(
                {
                    "id": identity,
                    **{
                        k: row.get(k, "")
                        for k in (
                            "environment",
                            "agent",
                            "model",
                            "task_id",
                            "level",
                            "reference_tokens",
                            "message_count",
                            "has_saved_user_or_system",
                        )
                    },
                    "units": len(self.plans[identity]),
                    "total": len(self.plans[identity]) * len(self.questions),
                    "approved": approved[identity],
                    "initial": initial[identity],
                }
            )
        legacy = self.root / "human-review.csv"
        legacy_count = 0
        if legacy.exists():
            with legacy.open(encoding="utf-8", newline="") as stream:
                legacy_count = sum(
                    bool(r.get("human_status")) for r in csv.DictReader(stream)
                )
        return {
            "run": self.root.name,
            "fingerprint": self.fingerprint,
            "rubric_version": self.manifest["protocol"].get("trace_check_version", ""),
            "questions": self.questions,
            "traces": sorted(traces, key=lambda r: int(r["reference_tokens"] or 0)),
            "total": sum(r["total"] for r in traces),
            "approved": sum(approved.values()),
            "final_decisions": len(self.gold(reviews)),
            "legacy_labels": legacy_count,
            "database": str(self.database),
        }

    def trace(self, identity):
        trace = self.raw_trace(identity)
        sources = trace.get("message_sources", [])
        return {
            "id": identity,
            "units": self.plans[identity],
            "capture": {
                k: trace.get(k) for k in ("completion_status", "reconstruction_status")
            },
            "messages": [
                {
                    "index": i,
                    "role": m.get("role", "unknown"),
                    "chars": len(m["content"]),
                    "preview": m["content"][:280],
                    "source": sources[i] if i < len(sources) else {},
                }
                for i, m in enumerate(trace["messages"])
            ],
            "reviews": self.records(identity),
        }

    def message(self, identity, index, start=0):
        messages = self.raw_trace(identity)["messages"]
        if not 0 <= index < len(messages):
            raise KeyError("Message index outside trace")
        text = messages[index]["content"]
        if not 0 <= start <= len(text):
            raise ValueError("Invalid message offset")
        end = min(start + 16000, len(text))
        return {
            "index": index,
            "start": start,
            "end": end,
            "total": len(text),
            "text": text[start:end],
        }

    def prompt_context(self, identity):
        trace = self.raw_trace(identity)
        sections, notes = [], []
        sources = trace.get("message_sources", [])
        for i, message in enumerate(trace["messages"]):
            if message.get("role") not in {"system", "developer", "user"}:
                break
            source = sources[i] if i < len(sources) else {}
            reconstructed = source.get("origin", "").startswith("reconstructed_")
            sections.append(
                {
                    "title": "Task description used for annotation"
                    if source.get("origin") == "reconstructed_task_prompt"
                    else f"{message['role'].capitalize()} prompt",
                    "role": message["role"],
                    "content": message["content"],
                    "status": "reconstructed" if reconstructed else "recorded",
                    "message_index": i,
                    "source_evidence": source.get("source_evidence", []),
                }
            )
        path = self.root / "prompt-context" / f"{identity}.json"
        if path.exists():
            extra = read_json(path)
            if extra.get("trace_sha256") != self.hashes[f"traces/{identity}.json"]:
                raise Conflict("Prompt context belongs to a different source trace")
            sections.extend(extra.get("sections", []))
            notes.extend(extra.get("notes", []))
        if not any(s["role"] == "system" for s in sections):
            notes.insert(
                0,
                "The system prompt was not saved in this report. Its full runtime text is unavailable.",
            )
        if not any(s["role"] == "user" for s in sections):
            notes.append("No startup user prompt was saved in this report.")
        sections.sort(key=lambda s: {"system": 0, "developer": 1}.get(s["role"], 2))
        return {"trace_id": identity, "sections": sections, "notes": notes}

    def search(self, identity, query, unit_id=""):
        if not 2 <= len(query) <= 200:
            raise ValueError("Search requires 2-200 characters")
        indices = None
        if unit_id:
            unit = next((u for u in self.plans[identity] if u["id"] == unit_id), None)
            if unit is None:
                raise KeyError("Unknown unit")
            indices = set(unit["indices"])
        hits = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        for i, msg in enumerate(self.raw_trace(identity)["messages"]):
            if indices is not None and i not in indices:
                continue
            for match in pattern.finditer(msg["content"]):
                hits.append(
                    {
                        "index": i,
                        "start": match.start(),
                        "end": match.end(),
                        "snippet": msg["content"][
                            max(0, match.start() - 70) : match.end() + 100
                        ],
                    }
                )
                if len(hits) == 100:
                    return {"hits": hits, "limited": True}
        return {"hits": hits, "limited": False}

    def validate_fields(self, body, check, trace_id):
        statuses = set(self.questions[check]["criteria"])
        final = body.get("final_status", "")
        suggested = body.get("suggested_status", "")
        initial = body.get("initial_status", body.get("draft_initial_status", ""))
        if (
            final not in statuses | {""}
            or suggested not in statuses | {""}
            or initial not in statuses | {""}
        ):
            raise ValueError("Status is not allowed by the frozen rubric")
        result = {
            "final_status": final,
            "suggested_status": suggested,
            "draft_initial_status": initial,
        }
        for name, limit in (
            ("reviewer", 160),
            ("rationale", 20000),
            ("suggestion", 20000),
        ):
            value = body.get(name, "")
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"Invalid {name}")
            result[name] = value.strip()
        evidence = body.get("evidence", [])
        if not isinstance(evidence, list) or len(evidence) > 100:
            raise ValueError("Invalid evidence list")
        messages = self.raw_trace(trace_id)["messages"]
        clean = []
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError("Invalid evidence item")
            idx, start, end = (item.get(k) for k in ("msg_idx", "start", "end"))
            if any(type(v) is not int for v in (idx, start, end)) or not 0 <= idx < len(
                messages
            ):
                raise ValueError("Invalid evidence coordinates")
            quote = item.get("quote")
            if (
                not isinstance(quote, str)
                or not 0 <= start < end <= len(messages[idx]["content"])
                or messages[idx]["content"][start:end] != quote
            ):
                raise ValueError(
                    "Evidence must exactly match the original message and Unicode offsets"
                )
            clean.append({"msg_idx": idx, "start": start, "end": end, "quote": quote})
        result["evidence"] = clean
        return result

    def save_review(self, body):
        key = self.key(body["trace_id"], body["unit_id"], body["check"])
        action = body.get("action", "draft")
        if action not in {"draft", "initial", "approve"}:
            raise ValueError("Unknown review action")
        fields = self.validate_fields(body, key[2], key[0])
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT data FROM reviews WHERE trace_id=? AND unit_id=? AND check_name=?",
                key,
            ).fetchone()
            old = json.loads(row[0]) if row else self.get_review(*key)
            if (
                type(body.get("version")) is not int
                or body["version"] != old["version"]
            ):
                raise Conflict(
                    "Another review changed this decision. Reload it before saving; your edits have not been overwritten."
                )
            initial = old.get("initial")
            if action == "initial":
                if initial:
                    raise ValueError(
                        "The initial judgment is already recorded. Use the final label to revise it."
                    )
                status = body.get("initial_status")
                if (
                    status not in self.questions[key[2]]["criteria"]
                    or not fields["reviewer"]
                ):
                    raise ValueError(
                        "Choose an initial label and enter your reviewer name"
                    )
                exposed = db.execute(
                    "SELECT 1 FROM history WHERE kind IN ('dashboard_reveal', 'annotation_view') LIMIT 1"
                ).fetchone()
                initial = {
                    "status": status,
                    "reviewer": fields["reviewer"],
                    "at": now(),
                    "comparison_seen": bool(exposed),
                    "rationale": fields["rationale"],
                    "evidence": fields["evidence"],
                }
            if action == "approve":
                if (
                    not initial
                    or not fields["final_status"]
                    or not fields["reviewer"]
                    or not fields["rationale"]
                ):
                    raise ValueError(
                        "Approval needs a saved initial judgment, final label, reviewer, and rationale"
                    )
                if fields["final_status"] == "present" and not fields["evidence"]:
                    raise ValueError(
                        "A present label requires at least one exact evidence quote"
                    )
            record = {
                **old,
                **fields,
                "initial": initial,
                "version": old["version"] + 1,
                "stage": "approved"
                if action == "approve"
                else ("initial" if initial else "draft"),
                "updated_at": now(),
            }
            db.execute(
                "INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?)",
                (*key, record["version"], json.dumps(record)),
            )
            db.execute(
                "INSERT INTO history (trace_id, unit_id, check_name, kind, at, data) VALUES (?, ?, ?, ?, ?, ?)",
                (*key, action, record["updated_at"], json.dumps(record)),
            )
        return record

    def reveal(self, trace_id, unit_id, check):
        key = self.key(trace_id, unit_id, check)
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            record = self.get_review(*key)
            if not record["initial"]:
                raise ValueError(
                    "Save your initial judgment before revealing model predictions"
                )
            if not record["predictions_revealed"]:
                record.update(predictions_revealed=True, version=record["version"] + 1)
                db.execute(
                    "UPDATE reviews SET version=?, data=? WHERE trace_id=? AND unit_id=? AND check_name=?",
                    (record["version"], json.dumps(record), *key),
                )
                db.execute(
                    "INSERT INTO history (trace_id, unit_id, check_name, kind, at, data) VALUES (?, ?, ?, 'reveal', ?, ?)",
                    (*key, now(), json.dumps(record)),
                )
        return {"review": record, "predictions": self.predictions(*key)}

    def predictions(self, trace_id, unit_id, check):
        results = []
        for model in self.models:
            path = self.root / "runs" / model / trace_id / "assessment.0.json"
            if (
                not path.exists()
                and trace_id not in self.code_assessments
                and not self.supplemental_runs
                and not (self.root / "fabrication-review" / f"{trace_id}.json").exists()
            ):
                results.append({"alias": self.aliases[model], "status": "not_run"})
                continue
            assessment = self.assessment(model, trace_id)
            unit = next(
                (u for u in assessment["units"] if u["unit_id"] == unit_id), None
            )
            result = (
                next((c for c in unit["checks"] if c["name"] == check), {})
                if unit
                else {}
            )
            results.append({"alias": self.aliases[model], **result})
        return sorted(results, key=lambda r: r["alias"])

    def assessment(self, model, trace_id):
        """Apply frozen code decisions while retaining raw model outputs on disk."""
        path = self.root / "runs" / model / trace_id / "assessment.0.json"
        try:
            assessment = read_json(path)
        except FileNotFoundError:
            if (
                trace_id not in self.code_assessments
                and not self.supplemental_runs
                and not (self.root / "fabrication-review" / f"{trace_id}.json").exists()
            ):
                raise
            assessment = {"checks": [], "units": []}
        for run in self.supplemental_runs:
            path = run["path"] / "runs" / model / trace_id / "assessment.0.json"
            if not path.exists():
                continue
            extra = read_json(path)
            expected_units = {u["id"] for u in self.plans[trace_id]}
            if {u["unit_id"] for u in extra["units"]} != expected_units:
                raise ValueError("Supplemental unit coverage differs")
            if any(
                {c["name"] for c in checks} != run["checks"]
                for checks in [extra["checks"], *[u["checks"] for u in extra["units"]]]
            ):
                raise ValueError("Supplemental result contains unexpected checks")
            assessment["checks"] = [
                c for c in assessment["checks"] if c["name"] not in run["checks"]
            ] + extra["checks"]
            units = {u["unit_id"]: u for u in assessment["units"]}
            assessment["units"] = [
                {
                    **units.get(u["unit_id"], {}),
                    **{k: v for k, v in u.items() if k != "checks"},
                    "checks": [
                        c
                        for c in units.get(u["unit_id"], {}).get("checks", [])
                        if c["name"] not in run["checks"]
                    ]
                    + u["checks"],
                }
                for u in extra["units"]
            ]
        review_path = self.root / "fabrication-review" / f"{trace_id}.json"
        if review_path.exists():
            assessment = apply_review(
                assessment,
                read_json(review_path),
                self.raw_trace(trace_id)["messages"],
                self.plans[trace_id],
            )
        code = self.code_assessments.get(trace_id)
        if code is None:
            return assessment

        def merge(original, assigned):
            previous = {c["name"]: c for c in original}
            return [c for c in original if c["name"] not in self.code_checks] + [
                {
                    **c,
                    "superseded_model_status": previous.get(c["name"], {}).get(
                        "status"
                    ),
                }
                for c in assigned
            ]

        units = {u["unit_id"]: u for u in assessment.get("units", [])}
        assessment["checks"] = merge(assessment.get("checks", []), code["checks"])
        assessment["units"] = [
            {
                **units.get(u["unit_id"], {}),
                "unit_id": u["unit_id"],
                "checks": merge(
                    units.get(u["unit_id"], {}).get("checks", []), u["checks"]
                ),
            }
            for u in code["units"]
        ]
        return assessment

    def annotation_sources(self):
        return [
            {"id": model, "name": model, "kind": "model"} for model in self.models
        ] + [{"id": "assistant", "name": "Assistant reference", "kind": "assistant"}]

    def assistant_labels(self):
        """Load a complete reference without importing it into human ground truth."""
        paths = sorted(self.root.glob("assistant-review-*/assistant-labels.json"))
        if not paths:
            return (
                {
                    "available": False,
                    "reason": "No assistant reference file is available for this benchmark.",
                },
                {},
                {},
                {},
            )
        path = paths[-1]
        try:
            signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            with self.lock:
                if self.assistant_cache and self.assistant_cache[0] == signature:
                    return self.assistant_cache[1]
            data = read_json(path)
            protocol = self.manifest["protocol"]
            if data.get("rubric_version") != protocol.get("trace_check_version"):
                raise ValueError("Assistant reference uses a different rubric version")
            if data.get("rubric_sha256") != value_digest(protocol):
                raise ValueError(
                    "Assistant reference rubric fingerprint does not match"
                )
            integrity = data.get("integrity", {})
            source_hashes = integrity.get("frozen_source_file_sha256")
            if source_hashes is not None:
                normalized = {
                    "/".join(Path(name).parts[-2:]): value
                    for name, value in source_hashes.items()
                }
                for name, expected in self.hashes.items():
                    if name != "manifest.json" and normalized.get(name) != expected:
                        raise ValueError(
                            "Assistant reference source fingerprint does not match"
                        )
            if (
                integrity.get("manifest_sha256", self.hashes["manifest.json"])
                != self.hashes["manifest.json"]
            ):
                raise ValueError(
                    "Assistant reference manifest fingerprint does not match"
                )
            labels, trace_labels, coverage = {}, {}, {}
            seen_traces = set()
            for trace in data["traces"]:
                tid = trace["trace_id"]
                if tid not in self.rows or tid in seen_traces:
                    raise ValueError(
                        "Assistant reference has an unknown or duplicate trace"
                    )
                seen_traces.add(tid)
                coverage[tid] = trace.get("coverage_notes", "")
                messages = self.raw_trace(tid)["messages"]
                seen_units = set()
                for unit in trace["units"]:
                    uid = unit["unit_id"]
                    if uid in seen_units:
                        raise ValueError("Assistant reference has a duplicate unit")
                    seen_units.add(uid)
                    plan_unit = next(
                        (u for u in self.plans[tid] if u["id"] == uid), None
                    )
                    if plan_unit is None:
                        raise ValueError("Assistant reference has an unknown unit")
                    for check in unit["checks"]:
                        name = check["name"]
                        key = self.key(tid, uid, name)
                        if (
                            key in labels
                            or check["status"] not in self.questions[name]["criteria"]
                        ):
                            raise ValueError(
                                "Assistant reference has a duplicate or invalid check"
                            )
                        if not isinstance(check.get("rationale", ""), str):
                            raise ValueError("Assistant rationale must be text")
                        support = check.get("support", [])
                        if not isinstance(support, list):
                            raise ValueError("Assistant evidence must be a list")
                        if check["status"] == "present" and not support:
                            raise ValueError(
                                "Assistant positive label has no source evidence"
                            )
                        for evidence in support:
                            idx, start, end = (
                                evidence.get(k) for k in ("msg_idx", "start", "end")
                            )
                            if (
                                any(type(v) is not int for v in (idx, start, end))
                                or idx not in plan_unit["indices"]
                            ):
                                raise ValueError(
                                    "Assistant evidence is outside its review unit"
                                )
                            if not 0 <= start < end <= len(
                                messages[idx]["content"]
                            ) or messages[idx]["content"][start:end] != evidence.get(
                                "quote"
                            ):
                                raise ValueError(
                                    "Assistant evidence does not match original text"
                                )
                        labels[key] = check
            expected = {
                (tid, unit["id"], name)
                for tid, units in self.plans.items()
                for unit in units
                for name in self.questions
            }
            if set(labels) != expected:
                raise ValueError(
                    "Assistant reference does not cover every frozen unit and check"
                )
            for tid, units in self.plans.items():
                for name in self.questions:
                    trace_labels[(tid, name)] = reduce_labels(
                        [labels[(tid, unit["id"], name)]["status"] for unit in units]
                    )
            for check in data.get("trace_checks", []):
                if (
                    trace_labels.get((check["trace_id"], check["name"]))
                    != check["status"]
                ):
                    raise ValueError(
                        "Assistant trace labels do not match the frozen unit reducer"
                    )
            metadata = {
                "available": True,
                "reason": "",
                "reference_kind": data.get(
                    "reference_kind", "Assistant reference; not human-approved"
                ),
                "rubric_version": data["rubric_version"],
                "rubric_sha256": data["rubric_sha256"],
                "path": str(path.relative_to(self.root)),
                "sha256": digest(path),
            }
            result = metadata, labels, trace_labels, coverage
            with self.lock:
                self.assistant_cache = signature, result
            return result
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return (
                {
                    "available": False,
                    "reason": f"Assistant reference unavailable: {exc}",
                    "path": str(path.relative_to(self.root)),
                },
                {},
                {},
                {},
            )

    def annotation_records(self, trace_id=None, unit_id=None):
        clauses, params = [], []
        for column, value in (("trace_id", trace_id), ("unit_id", unit_id)):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        with self.connect() as db:
            rows = db.execute(
                "SELECT data FROM annotation_reviews"
                + (" WHERE " + " AND ".join(clauses) if clauses else ""),
                params,
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def annotations(self, trace_id, unit_id, record_exposure=True):
        self.key(trace_id, unit_id, next(iter(self.questions)))
        # Confirm the selected trace still matches the source used by this session.
        self.raw_trace(trace_id)
        sources = self.annotation_sources()
        assistant, reference, trace_reference, coverage = self.assistant_labels()
        model_checks = {}
        for model in self.models:
            try:
                assessment = self.assessment(model, trace_id)
                unit = next(
                    (u for u in assessment["units"] if u["unit_id"] == unit_id), {}
                )
                model_checks[model] = (
                    {c["name"]: c for c in unit.get("checks", [])},
                    {c["name"]: c for c in assessment.get("checks", [])},
                    "",
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                model_checks[model] = {}, {}, f"Saved prediction unavailable: {exc}"
        saved = {
            (r["check"], r["source_id"]): r
            for r in self.annotation_records(trace_id, unit_id)
        }
        checks = []
        counts = {s["id"]: Counter() for s in sources}
        for name in self.questions:
            annotations = []
            for source in sources:
                sid = source["id"]
                if sid == "assistant":
                    raw = reference.get((trace_id, unit_id, name), {})
                    trace_status = trace_reference.get((trace_id, name), "not_run")
                    reason = assistant.get("reason", "")
                else:
                    unit_checks, full_checks, reason = model_checks[sid]
                    raw = unit_checks.get(name, {})
                    trace_status = full_checks.get(name, {}).get("status", "not_run")
                status = raw.get("status", "not_run")
                if status not in self.questions[name]["criteria"]:
                    status = "not_run"
                    reason = (
                        reason
                        or "No valid saved annotation is available for this check."
                    )
                fingerprint = value_digest(
                    {
                        "source_id": sid,
                        "trace_id": trace_id,
                        "unit_id": unit_id,
                        "check": name,
                        "annotation": {
                            k: v
                            for k, v in raw.items()
                            if k != "superseded_model_status"
                        },
                        "trace_status": trace_status,
                        "rubric": value_digest(self.manifest["protocol"]),
                        "trace_sha256": self.hashes[f"traces/{trace_id}.json"],
                    }
                )
                verdict = saved.get(
                    (name, sid),
                    {
                        "trace_id": trace_id,
                        "unit_id": unit_id,
                        "check": name,
                        "source_id": sid,
                        "version": 0,
                        "verdict": "unreviewed",
                        "corrected_status": "",
                        "notes": "",
                        "reviewer": "",
                        "source_fingerprint": fingerprint,
                    },
                )
                verdict = {
                    **verdict,
                    "stale": verdict.get("source_fingerprint") != fingerprint,
                }
                annotation = {
                    "source_id": sid,
                    "source_name": source["name"],
                    "kind": source["kind"],
                    "decision_source": raw.get("decision_source", source["kind"]),
                    "superseded_model_status": raw.get("superseded_model_status"),
                    "second_pass_review": raw.get("second_pass_review"),
                    "original_assessment": raw.get("original_assessment"),
                    "status": status,
                    "trace_status": trace_status,
                    "available": status != "not_run",
                    "rationale": raw.get("rationale", "") or reason,
                    "confidence": raw.get("confidence", ""),
                    "evidence": raw.get("support", raw.get("evidence", [])),
                    "elapsed_s": raw.get("elapsed_s"),
                    "request_count": raw.get("request_count"),
                    "coverage": raw.get("coverage"),
                    "probabilities": raw.get("probabilities"),
                    "provenance": raw.get("provenance"),
                    "validation_facts": raw.get("validation_facts"),
                    "evidence_record": raw.get("evidence_record"),
                    "pipeline_state": raw.get("state"),
                    "trigger_slices": raw.get("trigger_slices", []),
                    "errors": raw.get("errors", 0),
                    "source_fingerprint": fingerprint,
                    "verdict": verdict,
                }
                annotations.append(annotation)
                counts[sid]["unavailable"] += not annotation["available"]
                counts[sid]["stale"] += verdict["stale"]
                counts[sid][
                    "unreviewed" if verdict["stale"] else verdict["verdict"]
                ] += 1
            checks.append({"name": name, "annotations": annotations})
        if record_exposure:
            with self.lock, self.connect() as db:
                # A single exposure entry per unit is enough to preserve provenance.
                exposed = db.execute(
                    "SELECT 1 FROM history WHERE trace_id=? AND unit_id=? AND kind='annotation_view' LIMIT 1",
                    (trace_id, unit_id),
                ).fetchone()
                if not exposed:
                    db.execute(
                        "INSERT INTO history (trace_id, unit_id, kind, at, data) VALUES (?, ?, 'annotation_view', ?, ?)",
                        (trace_id, unit_id, now(), json.dumps({"sources": sources})),
                    )
        return {
            "trace_id": trace_id,
            "unit_id": unit_id,
            "sources": sources,
            "checks": checks,
            "assistant": {**assistant, "coverage_notes": coverage.get(trace_id, "")},
            "reviews_summary": {
                "reviewed": sum(
                    c[v]
                    for c in counts.values()
                    for v in ("correct", "incorrect", "unsure")
                ),
                "total": len(sources) * len(self.questions),
                "by_source": [
                    {
                        "source_id": sid,
                        **{v: c[v] for v in (*VERDICTS, "stale", "unavailable")},
                    }
                    for sid, c in counts.items()
                ],
            },
        }

    def save_annotation_review(self, body):
        key = self.key(body["trace_id"], body["unit_id"], body["check"])
        sid = body.get("source_id")
        if sid not in {s["id"] for s in self.annotation_sources()}:
            raise ValueError("Unknown annotation source")
        verdict = body.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError("Choose correct, incorrect, unsure, or unreviewed")
        corrected = body.get("corrected_status", "")
        if corrected not in set(self.questions[key[2]]["criteria"]) | {""}:
            raise ValueError("Corrected status is not allowed by the frozen rubric")
        if corrected and verdict != "incorrect":
            raise ValueError(
                "A corrected label is only recorded for an incorrect annotation"
            )
        fields = {}
        for name, limit in (("reviewer", 160), ("notes", 20000)):
            value = body.get(name, "")
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"Invalid {name}")
            fields[name] = value.strip()
        if verdict != "unreviewed" and not fields["reviewer"]:
            raise ValueError(
                "Enter your reviewer name before saving an annotation review"
            )
        with self.lock:
            bundle = self.annotations(key[0], key[1], record_exposure=False)
            check = next(c for c in bundle["checks"] if c["name"] == key[2])
            annotation = next(a for a in check["annotations"] if a["source_id"] == sid)
            if body.get("source_fingerprint") != annotation["source_fingerprint"]:
                raise Conflict(
                    "This source annotation changed. Reload it before reviewing the new version."
                )
            if not annotation["available"] and verdict != "unreviewed":
                raise ValueError(
                    "This source annotation is unavailable and cannot be reviewed"
                )
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT data FROM annotation_reviews WHERE trace_id=? AND unit_id=? AND check_name=? AND source_id=?",
                    (*key, sid),
                ).fetchone()
                old = json.loads(row[0]) if row else annotation["verdict"]
                if (
                    type(body.get("version")) is not int
                    or body["version"] != old["version"]
                ):
                    raise Conflict(
                        "Another review changed this annotation. Reload it before saving; your edits have not been overwritten."
                    )
                record = {
                    "trace_id": key[0],
                    "unit_id": key[1],
                    "check": key[2],
                    "source_id": sid,
                    "version": old["version"] + 1,
                    "verdict": verdict,
                    "corrected_status": corrected,
                    **fields,
                    "source_fingerprint": annotation["source_fingerprint"],
                    "source_annotation": {
                        k: v for k, v in annotation.items() if k != "verdict"
                    },
                    "updated_at": now(),
                    "stale": False,
                }
                db.execute(
                    "INSERT OR REPLACE INTO annotation_reviews VALUES (?, ?, ?, ?, ?, ?)",
                    (*key, sid, record["version"], json.dumps(record)),
                )
                db.execute(
                    "INSERT INTO history (trace_id, unit_id, check_name, kind, at, data) VALUES (?, ?, ?, 'annotation_verdict', ?, ?)",
                    (*key, record["updated_at"], json.dumps(record)),
                )
        return record

    def annotation_history(self, trace_id, unit_id, check, source_id):
        key = self.key(trace_id, unit_id, check)
        if source_id not in {s["id"] for s in self.annotation_sources()}:
            raise ValueError("Unknown annotation source")
        with self.connect() as db:
            rows = db.execute(
                "SELECT at, data FROM history WHERE trace_id=? AND unit_id=? AND check_name=? AND kind='annotation_verdict' ORDER BY id DESC",
                key,
            ).fetchall()
        entries = [
            {"kind": "annotation_verdict", "at": r[0], "review": json.loads(r[1])}
            for r in rows
        ]
        return [
            entry for entry in entries if entry["review"].get("source_id") == source_id
        ]

    def export_annotation_reviews_csv(self):
        stream = io.StringIO(newline="")
        columns = [
            "trace_id",
            "unit_id",
            "check",
            "source_id",
            "annotation_status",
            "verdict",
            "corrected_status",
            "reviewer",
            "notes",
            "version",
            "source_fingerprint",
            "stale",
            "updated_at",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in self.current_annotation_records():
            writer.writerow(
                {
                    **{key: record.get(key, "") for key in columns},
                    "annotation_status": record.get("source_annotation", {}).get(
                        "status", ""
                    ),
                }
            )
        return stream.getvalue()

    def current_annotation_records(self, records=None):
        records = self.annotation_records() if records is None else records
        bundles = {}
        for record in records:
            key = record["trace_id"], record["unit_id"]
            if key not in bundles:
                bundle = self.annotations(*key, record_exposure=False)
                bundles[key] = {
                    (c["name"], a["source_id"]): a["source_fingerprint"]
                    for c in bundle["checks"]
                    for a in c["annotations"]
                }
        return [
            {
                **r,
                "stale": r.get("source_fingerprint")
                != bundles[(r["trace_id"], r["unit_id"])].get(
                    (r["check"], r["source_id"])
                ),
            }
            for r in records
        ]

    def history(self, trace_id, unit_id, check):
        key = self.key(trace_id, unit_id, check)
        with self.connect() as db:
            rows = db.execute(
                "SELECT kind, at, data FROM history WHERE trace_id=? AND unit_id=? AND check_name=? ORDER BY id DESC",
                key,
            ).fetchall()
        return [{"kind": r[0], "at": r[1], "review": json.loads(r[2])} for r in rows]

    def gold(self, records=None):
        records = self.records() if records is None else records
        by_key = {
            (r["trace_id"], r["unit_id"], r["check"]): r
            for r in records
            if r["stage"] == "approved"
        }
        gold = []
        for trace_id, units in self.plans.items():
            for check in self.questions:
                reviews = [by_key.get((trace_id, u["id"], check)) for u in units]
                if all(reviews):
                    gold.append(
                        {
                            "trace_id": trace_id,
                            "check": check,
                            "status": reduce_labels(
                                [r["final_status"] for r in reviews]
                            ),
                            "units": reviews,
                        }
                    )
        return gold

    def export(self):
        with self.connect() as db:
            # One read snapshot keeps labels and their audit trail consistent
            # even when another reviewer saves during an export.
            db.execute("BEGIN")
            records = [json.loads(r[0]) for r in db.execute("SELECT data FROM reviews")]
            annotation_reviews = [
                json.loads(r[0])
                for r in db.execute("SELECT data FROM annotation_reviews")
            ]
            history = [
                {"id": r[0], "kind": r[1], "at": r[2], "data": json.loads(r[3])}
                for r in db.execute(
                    "SELECT id, kind, at, data FROM history ORDER BY id"
                )
            ]
        return {
            "schema": "corral-benchmark-review-v1",
            "run": self.root.name,
            "fingerprint": self.fingerprint,
            "source_hashes": self.hashes,
            "rubric": self.manifest["protocol"],
            "exported_at": now(),
            "reviews": records,
            "annotation_reviews": self.current_annotation_records(annotation_reviews),
            "annotation_sources": self.annotation_sources(),
            "assistant_reference": self.assistant_labels()[0],
            "code_assessments": self.code_assessments,
            "gold": self.gold(records),
            "history": history,
        }

    def export_csv(self):
        stream = io.StringIO(newline="")
        columns = [
            "trace_id",
            "check",
            "trace",
            "human_status",
            "evidence_message_ids",
            "reviewer",
            "notes",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        gold = {(g["trace_id"], g["check"]): g for g in self.gold()}
        for trace_id in self.rows:
            for check in self.questions:
                g = gold.get((trace_id, check))
                units = g["units"] if g else []
                writer.writerow(
                    {
                        "trace_id": trace_id,
                        "check": check,
                        "trace": f"traces/{trace_id}.txt",
                        "human_status": g["status"] if g else "",
                        "evidence_message_ids": ";".join(
                            f"m{i}"
                            for i in sorted(
                                {e["msg_idx"] for r in units for e in r["evidence"]}
                            )
                        ),
                        "reviewer": "; ".join(sorted({r["reviewer"] for r in units})),
                        "notes": " | ".join(
                            f"{r['unit_id']}: {r['rationale']}" for r in units
                        ),
                    }
                )
        return stream.getvalue()

    def dashboard(self):
        with self.connect() as db:
            db.execute(
                "INSERT INTO history (kind, at, data) VALUES ('dashboard_reveal', ?, '{}')",
                (now(),),
            )
        gold = [g for g in self.gold() if g["check"] not in self.code_checks]
        times = defaultdict(lambda: defaultdict(float))
        timing_paths = [
            self.root / "timings.csv",
            *[r["path"] / "timings.csv" for r in self.supplemental_runs],
        ]
        for timing_path in timing_paths:
            if not timing_path.exists():
                continue
            with timing_path.open(newline="", encoding="utf-8") as stream:
                for r in csv.DictReader(stream):
                    if (
                        r.get("repeat", "0") == "0"
                        and r.get("outcome") == "ok"
                        and r.get("model_called", "True").lower() != "false"
                        and r["check"] not in self.code_checks
                    ):
                        times[r["model"]][(r["trace_id"], r["check"])] += float(
                            r["elapsed_s"]
                        )
        results = []
        for model in self.models:
            predicted = {}
            for trace_id in self.rows:
                try:
                    assessment = self.assessment(model, trace_id)
                except FileNotFoundError:
                    continue
                for c in assessment["checks"]:
                    predicted[(trace_id, c["name"])] = c["status"]
            per_check, matrix = (
                [],
                {s: dict.fromkeys((*STATUSES, "missing"), 0) for s in STATUSES},
            )
            correct = covered = resolved = 0
            for g in gold:
                p = predicted.get((g["trace_id"], g["check"]), "missing")
                if p not in STATUSES:
                    p = "missing"
                matrix[g["status"]][p] += 1
                correct += p == g["status"]
                if g["status"] in {"present", "absent"}:
                    resolved += 1
                    covered += p in {"present", "absent"}
            for check in self.questions:
                if check in self.code_checks:
                    continue
                group = [
                    g
                    for g in gold
                    if g["check"] == check and g["status"] in {"present", "absent"}
                ]
                tp = fp = fn = positives = 0
                for g in group:
                    p = predicted.get((g["trace_id"], check), "missing")
                    positives += g["status"] == "present"
                    tp += g["status"] == "present" and p == "present"
                    fp += g["status"] == "absent" and p == "present"
                    fn += g["status"] == "present" and p != "present"
                per_check.append(
                    {
                        "check": check,
                        "n": len(group),
                        "positives": positives,
                        "precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / (tp + fn) if tp + fn else None,
                        "f1": 2 * tp / (2 * tp + fp + fn) if positives else None,
                    }
                )
            supported = [c["f1"] for c in per_check if c["f1"] is not None]
            latency = sorted(times[model].values())
            results.append(
                {
                    "model": model,
                    "alias": self.aliases[model],
                    "n": len(gold),
                    "accuracy": correct / len(gold) if gold else None,
                    "coverage": covered / resolved if resolved else None,
                    "macro_f1": statistics.mean(supported) if supported else None,
                    "supported_checks": len(supported),
                    "checks": per_check,
                    "matrix": matrix,
                    "timed_decisions": len(latency),
                    "mean_s": statistics.mean(latency) if latency else None,
                    "median_s": statistics.median(latency) if latency else None,
                    "p95_s": latency[math.ceil(0.95 * len(latency)) - 1]
                    if latency
                    else None,
                }
            )
        return {
            "gold_count": len(gold),
            "excluded_code_checks": sorted(self.code_checks),
            "gold_statuses": dict(Counter(g["status"] for g in gold)),
            "total": len(self.rows) * (len(self.questions) - len(self.code_checks)),
            "models": results,
        }


def handler_for(store, token, *, graph_store=None):
    graph_handler = None
    if graph_store is not None:
        if __package__:
            from .server import handler_for as graph_handler_for
        else:
            from server import handler_for as graph_handler_for
        graph_handler = graph_handler_for(graph_store, base_path="/graphs")

    class Handler(graph_handler or BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(
            self,
            data,
            content_type="application/json; charset=utf-8",
            status=200,
            filename=None,
        ):
            if not isinstance(data, str | bytes):
                data = json.dumps(data, ensure_ascii=False)
            if isinstance(data, str):
                data = data.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )
            if filename:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{filename}"'
                )
            self.end_headers()
            self.wfile.write(data)

        def allowed_origin(self):
            host = self.headers.get("Host", "")
            port = self.server.server_port
            if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
                return False
            origin = self.headers.get("Origin")
            return (not origin or origin == f"http://{host}") and self.headers.get(
                "Sec-Fetch-Site"
            ) != "cross-site"

        def do_GET(self):
            if not self.allowed_origin():
                return self.send(
                    {"error": "Local same-origin access required"}, status=403
                )
            try:
                url = urlsplit(self.path)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                path = url.path
                if graph_handler and (path == "/graphs" or path.startswith("/graphs/")):
                    return graph_handler.do_GET(self)
                if path == "/":
                    page = (
                        (ASSETS / "benchmark.html")
                        .read_text()
                        .replace("__REVIEW_TOKEN__", token)
                    )
                    if graph_handler:
                        page = page.replace(
                            '<div class="exports">',
                            '<div class="exports"><a href="/graphs/">Graph viewer ↗</a>',
                            1,
                        )
                    return self.send(
                        page,
                        "text/html; charset=utf-8",
                    )
                if path in {"/benchmark.js", "/benchmark.css"}:
                    return self.send(
                        (ASSETS / path[1:]).read_bytes(),
                        "text/javascript; charset=utf-8"
                        if path.endswith(".js")
                        else "text/css; charset=utf-8",
                    )
                if path == "/api/catalog":
                    return self.send(store.catalog())
                if path == "/api/trace":
                    return self.send(store.trace(q["trace_id"]))
                if path == "/api/message":
                    return self.send(
                        store.message(
                            q["trace_id"], int(q["index"]), int(q.get("start", 0))
                        )
                    )
                if path == "/api/prompt-context":
                    return self.send(store.prompt_context(q["trace_id"]))
                if path == "/api/search":
                    return self.send(
                        store.search(q["trace_id"], q["q"], q.get("unit_id", ""))
                    )
                if path == "/api/review":
                    return self.send(
                        store.get_review(q["trace_id"], q["unit_id"], q["check"])
                    )
                if path == "/api/history":
                    return self.send(
                        store.history(q["trace_id"], q["unit_id"], q["check"])
                    )
                if path == "/api/annotation-history":
                    return self.send(
                        store.annotation_history(
                            q["trace_id"], q["unit_id"], q["check"], q["source_id"]
                        )
                    )
                if path == "/api/annotation-reviews.csv":
                    return self.send(
                        store.export_annotation_reviews_csv(),
                        "text/csv; charset=utf-8",
                        filename="annotation-reviews.csv",
                    )
                if path == "/api/export.json":
                    return self.send(store.export(), filename="benchmark-reviews.json")
                if path == "/api/export.csv":
                    return self.send(
                        store.export_csv(),
                        "text/csv; charset=utf-8",
                        filename="human-review.csv",
                    )
                return self.send({"error": "Not found"}, status=404)
            except Conflict as exc:
                self.send({"error": str(exc)}, status=409)
            except KeyError:
                self.send({"error": "Unknown record or missing parameter"}, status=404)
            except (ValueError, TypeError) as exc:
                self.send({"error": str(exc)}, status=400)
            except OSError:
                self.send({"error": "Could not read the benchmark file"}, status=500)

        def do_POST(self):
            if not self.allowed_origin() or not secrets.compare_digest(
                self.headers.get("X-Review-Token", ""), token
            ):
                return self.send({"error": "Local review token required"}, status=403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/review":
                    return self.send(store.save_review(body))
                if self.path == "/api/annotations":
                    return self.send(
                        store.annotations(body["trace_id"], body["unit_id"])
                    )
                if self.path == "/api/annotation-review":
                    return self.send(store.save_annotation_review(body))
                if self.path == "/api/reveal":
                    return self.send(
                        store.reveal(body["trace_id"], body["unit_id"], body["check"])
                    )
                if self.path == "/api/dashboard":
                    return self.send(store.dashboard())
                return self.send({"error": "Not found"}, status=404)
            except Conflict as exc:
                self.send({"error": str(exc)}, status=409)
            except (KeyError, ValueError, TypeError) as exc:
                self.send({"error": str(exc)}, status=400)
            except (OSError, sqlite3.Error):
                self.send(
                    {
                        "error": "Could not save review; your changes are still in the form"
                    },
                    status=500,
                )

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--database", type=Path, help="Optional separate SQLite review database"
    )
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--graph-run", type=Path, help="Also serve completed graphs at /graphs/"
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Reuse the local review session token on restart",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store = ReviewStore(args.run, args.database)
    graph_store = None
    if args.graph_run:
        if __package__:
            from .graph_run import GraphRunStore
        else:
            from graph_run import GraphRunStore
        graph_store = GraphRunStore(args.graph_run)
    token = (
        args.token_file.read_text().strip()
        if args.token_file
        else secrets.token_urlsafe(32)
    )
    if not token:
        parser.error("Review token must not be empty")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler_for(store, token, graph_store=graph_store)
    )
    logger.info("Trace review: http://127.0.0.1:%s", server.server_port)
    logger.info("Reviews: %s", store.database)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
