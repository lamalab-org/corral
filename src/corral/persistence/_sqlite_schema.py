"""Core tables for the SQLite commit ledger's existing on-disk format."""

import sqlalchemy as sa

metadata = sa.MetaData()

# Preserve the original nullability, including SQLite's TEXT primary keys.
commits = sa.Table(
    "commits",
    metadata,
    sa.Column("hash", sa.Text, primary_key=True, nullable=True),
    sa.Column("schema_version", sa.Integer, nullable=False),
    sa.Column("execution_id", sa.Text, nullable=False),
    sa.Column("branch_id", sa.Text, nullable=False),
    sa.Column("sequence", sa.Integer, nullable=False),
    sa.Column("branch_sequence", sa.Integer, nullable=False),
    sa.Column("parent_hash", sa.Text, sa.ForeignKey("commits.hash")),
    sa.Column("based_on_hash", sa.Text, sa.ForeignKey("commits.hash")),
    sa.Column("author_json", sa.Text, nullable=False),
    sa.Column("author_kind", sa.Text, nullable=False),
    sa.Column("author_id", sa.Text, nullable=False),
    sa.Column("author_run_id", sa.Text, nullable=False),
    sa.Column("parent_run_id", sa.Text),
    sa.Column("event_type", sa.Text, nullable=False),
    sa.Column("event_json", sa.Text, nullable=False),
    sa.Column("action_id", sa.Text),
    sa.Column("invocation_id", sa.Text),
    sa.Column("requested_by_run_id", sa.Text),
    sa.Column("occurred_at", sa.Text, nullable=False),
    sa.Column("recorded_at", sa.Text, nullable=False),
    sa.UniqueConstraint("execution_id", "sequence"),
    sa.UniqueConstraint("execution_id", "branch_id", "branch_sequence"),
)

branch_heads = sa.Table(
    "branch_heads",
    metadata,
    sa.Column("execution_id", sa.Text, primary_key=True),
    sa.Column("branch_id", sa.Text, primary_key=True),
    sa.Column("origin_hash", sa.Text, sa.ForeignKey("commits.hash")),
    sa.Column("head_hash", sa.Text, sa.ForeignKey("commits.hash")),
)

idempotency_keys = sa.Table(
    "idempotency_keys",
    metadata,
    sa.Column("execution_id", sa.Text, primary_key=True),
    sa.Column("request_id", sa.Text, primary_key=True),
    sa.Column("commit_hash", sa.Text, sa.ForeignKey("commits.hash"), nullable=False),
)

snapshots = sa.Table(
    "snapshots",
    metadata,
    sa.Column("execution_id", sa.Text, nullable=False),
    sa.Column(
        "commit_hash",
        sa.Text,
        sa.ForeignKey("commits.hash"),
        primary_key=True,
        nullable=True,
    ),
    sa.Column("state_json", sa.Text, nullable=False),
    sa.Column("created_at", sa.Text, nullable=False),
)

sa.Index("commits_parent_idx", commits.c.parent_hash)
sa.Index(
    "commits_author_run_idx",
    commits.c.execution_id,
    commits.c.author_run_id,
    commits.c.sequence,
)
sa.Index(
    "commits_action_idx",
    commits.c.execution_id,
    commits.c.action_id,
    commits.c.sequence,
)
sa.Index(
    "commits_invocation_idx",
    commits.c.execution_id,
    commits.c.invocation_id,
    commits.c.sequence,
)
sa.Index(
    "commits_requested_run_idx",
    commits.c.execution_id,
    commits.c.requested_by_run_id,
    commits.c.sequence,
)
