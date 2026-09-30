"""Durable generation delivery receipts, stored alongside the commit ledger.

The ledger holds the payload. These tables only distinguish observations that
were never sent from deliveries acknowledged by the exporter or left uncertain.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
from contextlib import closing
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class GenerationDelivery:
    def __init__(self, path: Path, destination: str, execution_id: str):
        # Keep synchronous telemetry writes away from the async commit writer.
        # Docker checkpoints copy the whole directory, including this database.
        self.path = path.with_name(f"{path.name}.langfuse")
        self.destination = destination
        self.execution_id = execution_id
        with closing(self._connect()) as db, db:
            # Keep an empty journal after commits so checkpoint directory copies
            # cannot retain a stale hot journal from an earlier interrupted run.
            db.execute("PRAGMA journal_mode = TRUNCATE")
            db.execute("""CREATE TABLE IF NOT EXISTS langfuse_executions (
                destination TEXT, execution_id TEXT,
                PRIMARY KEY (destination, execution_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS langfuse_generations (
                destination TEXT, commit_hash TEXT, observation_id TEXT,
                status TEXT NOT NULL,
                PRIMARY KEY (destination, commit_hash))""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA synchronous = FULL")
        return db

    def track(self) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO langfuse_executions VALUES (?, ?)",
                (self.destination, self.execution_id),
            )

    def tracked(self) -> bool:
        with closing(self._connect()) as db:
            return (
                db.execute(
                    "SELECT 1 FROM langfuse_executions WHERE destination=? AND execution_id=?",
                    (self.destination, self.execution_id),
                ).fetchone()
                is not None
            )

    def receipt(self, commit_hash: str) -> tuple[str, str | None]:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT status, observation_id FROM langfuse_generations WHERE destination=? AND commit_hash=?",
                (self.destination, commit_hash),
            ).fetchone()
        return tuple(row) if row else ("pending", None)

    def claim(self, commit_hash: str, observation_id: str) -> bool:
        # Commit this BEFORE the network request. A crash leaves 'sending',
        # which requires remote reconciliation, never an automatic resend.
        with closing(self._connect()) as db, db:
            return (
                db.execute(
                    "INSERT OR IGNORE INTO langfuse_generations VALUES (?, ?, ?, 'sending')",
                    (self.destination, commit_hash, observation_id),
                ).rowcount
                == 1
            )

    def confirm(self, commit_hash: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE langfuse_generations SET status='confirmed' WHERE destination=? AND commit_hash=?",
                (self.destination, commit_hash),
            )


class DeliveryExporter:
    """Wrap the SDK's transport so flush is never mistaken for a receipt."""

    def __init__(self, exporter, destination: str):
        self.exporter = exporter
        self.destination = destination
        self.stores: dict[str, GenerationDelivery] = {}

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult

        selected, receipts = [], []
        for span in spans:
            attributes = span.attributes or {}
            commit_hash = attributes.get("langfuse.observation.metadata.commit_hash")
            store = self.stores.get(
                attributes.get("langfuse.trace.metadata.execution_id")
            )
            if commit_hash and store is not None and store.tracked():
                if not store.claim(commit_hash, f"{span.context.span_id:016x}"):
                    continue
                receipts.append((store, commit_hash))
            selected.append(span)
        if not selected:
            return SpanExportResult.SUCCESS
        result = self.exporter.export(selected)
        if result == SpanExportResult.SUCCESS:
            for store, commit_hash in receipts:
                store.confirm(commit_hash)
        return result

    def shutdown(self):
        self.exporter.shutdown()

    def force_flush(self, timeout_millis=30_000):
        return self.exporter.force_flush(timeout_millis=timeout_millis)


@lru_cache
def task_exporter(public_key: str | None) -> DeliveryExporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv(
        "LANGFUSE_HOST", "https://cloud.langfuse.com"
    )
    base_url = base_url.rstrip("/")
    secret = os.environ["LANGFUSE_SECRET_KEY"]
    authorization = base64.b64encode(f"{public_key}:{secret}".encode()).decode()
    export_path = os.getenv(
        "LANGFUSE_OTEL_TRACES_EXPORT_PATH", "api/public/otel/v1/traces"
    )
    exporter = OTLPSpanExporter(
        endpoint=f"{base_url}/{export_path.lstrip('/')}",
        headers={
            "Authorization": f"Basic {authorization}",
            "x-langfuse-public-key": public_key,
            "x-langfuse-ingestion-version": "4",
        },
        timeout=int(os.getenv("LANGFUSE_TIMEOUT", "5")),
    )
    destination = hashlib.sha256(f"{base_url}:{public_key}".encode()).hexdigest()
    return DeliveryExporter(exporter, destination)
