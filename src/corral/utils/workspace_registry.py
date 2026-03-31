from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


@dataclass
class WorkspaceEntry:
    """Metadata for a trial workspace directory."""

    workspace_id: str  # UUID4
    task_id: str
    trial_id: str  # logical "0", "1", etc.
    path: str  # full workspace path
    status: str  # "active" | "completed" | "abandoned"
    created_at: str  # ISO 8601
    completed_at: str | None = None


class WorkspaceRegistry:
    """SQLite-backed registry tracking trial workspaces and their files.

    This is internal infrastructure — agents never interact with it.
    The registry enables:
    - UUID-based workspace identification
    - Persistent tracking of workspaces across sessions
    - Registry-backed file lookup (replacing fragile rglob search)
    - Post-hoc analysis and debugging of benchmark runs
    """

    DB_FILENAME = ".corral_registry.db"

    def __init__(self, base_work_dir: str):
        self.base_work_dir = base_work_dir
        self.db_path = str(Path(base_work_dir) / self.DB_FILENAME)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        Path(self.base_work_dir).mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    trial_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
                    filename TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_files_filename ON files(filename);
                CREATE INDEX IF NOT EXISTS idx_files_workspace ON files(workspace_id);
                CREATE INDEX IF NOT EXISTS idx_workspaces_task ON workspaces(task_id);
                """
            )

    def _get_conn(self) -> sqlite3.Connection:
        """Open a connection. Use as context manager for auto-commit."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def register_workspace(
        self,
        task_id: str,
        trial_id: str,
        workspace_id: str,
        path: str,
    ) -> WorkspaceEntry:
        """Register a new workspace in the registry."""
        now = datetime.now(tz=timezone.utc).isoformat()
        entry = WorkspaceEntry(
            workspace_id=workspace_id,
            task_id=task_id,
            trial_id=trial_id,
            path=path,
            status="active",
            created_at=now,
        )
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO workspaces (workspace_id, task_id, trial_id, path, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.workspace_id,
                    entry.task_id,
                    entry.trial_id,
                    entry.path,
                    entry.status,
                    entry.created_at,
                ),
            )
        logger.debug(
            f"Registered workspace {workspace_id} for {task_id} trial {trial_id}"
        )
        return entry

    def update_status(
        self,
        workspace_id: str,
        status: str,
        completed_at: str | None = None,
    ) -> None:
        """Update workspace status (e.g., active -> completed)."""
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE workspaces SET status = ?, completed_at = ?
                WHERE workspace_id = ?
                """,
                (status, completed_at, workspace_id),
            )

    def register_file(
        self,
        workspace_id: str,
        filename: str,
        rel_path: str,
    ) -> None:
        """Track a file created in a workspace."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO files (workspace_id, filename, rel_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (workspace_id, filename, rel_path, now),
            )

    def find_file(
        self,
        filename: str,
        task_id: str | None = None,
    ) -> str | None:
        """Find a file by name, optionally scoped to a task.

        Returns the full path of the most recently registered match, or None.
        """
        with self._get_conn() as conn:
            if task_id:
                row = conn.execute(
                    """
                    SELECT f.rel_path FROM files f
                    JOIN workspaces w ON f.workspace_id = w.workspace_id
                    WHERE f.filename = ? AND w.task_id = ?
                    ORDER BY f.created_at DESC
                    LIMIT 1
                    """,
                    (filename, task_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT rel_path FROM files
                    WHERE filename = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (filename,),
                ).fetchone()
        if row:
            return row["rel_path"]
        return None

    def get_workspaces(
        self,
        task_id: str | None = None,
        status: str | None = None,
    ) -> list[WorkspaceEntry]:
        """Query workspaces, optionally filtered by task_id and/or status."""
        query = "SELECT * FROM workspaces WHERE 1=1"
        params: list[str] = []
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at"

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            WorkspaceEntry(
                workspace_id=r["workspace_id"],
                task_id=r["task_id"],
                trial_id=r["trial_id"],
                path=r["path"],
                status=r["status"],
                created_at=r["created_at"],
                completed_at=r["completed_at"],
            )
            for r in rows
        ]

    def get_by_id(self, workspace_id: str) -> WorkspaceEntry | None:
        """Look up a workspace by its UUID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        if not row:
            return None
        return WorkspaceEntry(
            workspace_id=row["workspace_id"],
            task_id=row["task_id"],
            trial_id=row["trial_id"],
            path=row["path"],
            status=row["status"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )
