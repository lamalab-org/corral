-- Ledger written by the original sqlite3 store, before the SQLAlchemy refactor.
BEGIN TRANSACTION;
CREATE TABLE branch_heads (
                execution_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                origin_hash TEXT REFERENCES commits(hash),
                head_hash TEXT REFERENCES commits(hash),
                PRIMARY KEY (execution_id, branch_id)
            );
INSERT INTO "branch_heads" VALUES('legacy','main',NULL,'d7169f48668e22c5673ff9423cde3f24fa5b4ca1a497cd5501a0ff48159b880a');
INSERT INTO "branch_heads" VALUES('legacy','experiment','44381f651f1d5b6f1d972088f21aaba9f0fe7d6df5ae12e79da174751e5ec778','44381f651f1d5b6f1d972088f21aaba9f0fe7d6df5ae12e79da174751e5ec778');
CREATE TABLE commits (
                hash TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                execution_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                branch_sequence INTEGER NOT NULL,
                parent_hash TEXT REFERENCES commits(hash),
                based_on_hash TEXT REFERENCES commits(hash),
                author_json TEXT NOT NULL,
                author_kind TEXT NOT NULL,
                author_id TEXT NOT NULL,
                author_run_id TEXT NOT NULL,
                parent_run_id TEXT,
                event_type TEXT NOT NULL,
                event_json TEXT NOT NULL,
                action_id TEXT,
                invocation_id TEXT,
                requested_by_run_id TEXT,
                occurred_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                UNIQUE (execution_id, sequence),
                UNIQUE (execution_id, branch_id, branch_sequence)
            );
INSERT INTO "commits" VALUES('2e2bc791abae4056ef5613f1e938619dc9803c2cb15dbe0ef2ffbb3386b8eebf',1,'legacy','main',0,0,NULL,NULL,'{"actor_id":"corral","kind":"runtime","parent_run_id":null,"run_id":"runtime:legacy"}','runtime','corral','runtime:legacy',NULL,'execution.started','{"dependency_outputs":{},"environment":{},"environment_metadata":{},"model":{},"runtime":{"ended_at":null,"metadata":{},"started_at":"2026-01-01T00:00:00Z","status":"running"},"scaffold":{},"task":{},"type":"execution.started","workspace":{"artifacts":{},"files":{},"id":"fd5cc0dc-1f18-49f7-8c5d-c2a6f7f4128a","revision":0,"schema_version":2}}',NULL,NULL,NULL,'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00');
INSERT INTO "commits" VALUES('44381f651f1d5b6f1d972088f21aaba9f0fe7d6df5ae12e79da174751e5ec778',1,'legacy','main',1,1,'2e2bc791abae4056ef5613f1e938619dc9803c2cb15dbe0ef2ffbb3386b8eebf','2e2bc791abae4056ef5613f1e938619dc9803c2cb15dbe0ef2ffbb3386b8eebf','{"actor_id":"corral","kind":"runtime","parent_run_id":null,"run_id":"runtime:legacy"}','runtime','corral','runtime:legacy',NULL,'agent.started','{"agent_id":"agent","agent_run_id":"agent-run","context_cutoff_hash":null,"handoff":null,"metadata":{},"parent_run_id":null,"type":"agent.started"}',NULL,NULL,NULL,'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00');
INSERT INTO "commits" VALUES('d7169f48668e22c5673ff9423cde3f24fa5b4ca1a497cd5501a0ff48159b880a',1,'legacy','main',2,2,'44381f651f1d5b6f1d972088f21aaba9f0fe7d6df5ae12e79da174751e5ec778','44381f651f1d5b6f1d972088f21aaba9f0fe7d6df5ae12e79da174751e5ec778','{"actor_id":"agent","kind":"agent","parent_run_id":null,"run_id":"agent-run"}','agent','agent','agent-run',NULL,'agent.turn_recorded','{"actions":[],"messages":[{"content":"legacy message","role":"assistant"}],"metadata":{},"parallel_group_id":null,"type":"agent.turn_recorded","usage_delta":{"agent_steps":0,"input_tokens":0,"llm_calls":0,"metadata":{},"output_tokens":0,"reasoning_tokens":0,"tool_calls":0}}',NULL,NULL,NULL,'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00');
CREATE TABLE idempotency_keys (
                execution_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                commit_hash TEXT NOT NULL REFERENCES commits(hash),
                PRIMARY KEY (execution_id, request_id)
            );
INSERT INTO "idempotency_keys" VALUES('legacy','start','2e2bc791abae4056ef5613f1e938619dc9803c2cb15dbe0ef2ffbb3386b8eebf');
INSERT INTO "idempotency_keys" VALUES('legacy','agent','44381f651f1d5b6f1d972088f21aaba9f0fe7d6df5ae12e79da174751e5ec778');
INSERT INTO "idempotency_keys" VALUES('legacy','turn','d7169f48668e22c5673ff9423cde3f24fa5b4ca1a497cd5501a0ff48159b880a');
CREATE TABLE snapshots (
                execution_id TEXT NOT NULL,
                commit_hash TEXT PRIMARY KEY REFERENCES commits(hash),
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
INSERT INTO "snapshots" VALUES('legacy','2e2bc791abae4056ef5613f1e938619dc9803c2cb15dbe0ef2ffbb3386b8eebf','{"through_commit_hash":"2e2bc791abae4056ef5613f1e938619dc9803c2cb15dbe0ef2ffbb3386b8eebf","execution_id":"legacy","branch_id":"main","task":{"metadata":{},"model":{},"scaffold":{},"environment":{},"dependency_outputs":{}},"environment":{"revision":0,"values":{}},"workspace":{"schema_version":2,"id":"fd5cc0dc-1f18-49f7-8c5d-c2a6f7f4128a","revision":0,"files":{},"artifacts":{}},"agent_runs":{},"conversations":{},"actions":{},"tool_invocations":{},"usage_by_run":{},"runtime":{"status":"running","started_at":"2026-01-01T00:00:00Z","ended_at":null,"metadata":{}},"submission":null}','2026-01-01T00:00:00+00:00');
INSERT INTO "snapshots" VALUES('legacy','d7169f48668e22c5673ff9423cde3f24fa5b4ca1a497cd5501a0ff48159b880a','{"through_commit_hash":"d7169f48668e22c5673ff9423cde3f24fa5b4ca1a497cd5501a0ff48159b880a","execution_id":"legacy","branch_id":"main","task":{"metadata":{},"model":{},"scaffold":{},"environment":{},"dependency_outputs":{}},"environment":{"revision":0,"values":{}},"workspace":{"schema_version":2,"id":"fd5cc0dc-1f18-49f7-8c5d-c2a6f7f4128a","revision":0,"files":{},"artifacts":{}},"agent_runs":{"agent-run":{"run_id":"agent-run","actor_id":"agent","parent_run_id":null,"status":"running","context_cutoff_hash":null,"handoff":null,"child_run_ids":[],"algorithm_state":{},"imported_context":[],"result_summary":null,"trace_head":null,"metadata":{}}},"conversations":{"agent-run":[{"role":"assistant","content":"legacy message"}]},"actions":{},"tool_invocations":{},"usage_by_run":{"agent-run":{"input_tokens":0,"output_tokens":0,"reasoning_tokens":0,"llm_calls":0,"tool_calls":0,"agent_steps":0}},"runtime":{"status":"running","started_at":"2026-01-01T00:00:00Z","ended_at":null,"metadata":{}},"submission":null}','2026-01-01T00:00:00+00:00');
CREATE INDEX commits_parent_idx ON commits(parent_hash);
CREATE INDEX commits_author_run_idx
                ON commits(execution_id, author_run_id, sequence);
CREATE INDEX commits_action_idx
                ON commits(execution_id, action_id, sequence);
CREATE INDEX commits_invocation_idx
                ON commits(execution_id, invocation_id, sequence);
CREATE INDEX commits_requested_run_idx
                ON commits(execution_id, requested_by_run_id, sequence);
COMMIT;
