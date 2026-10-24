import asyncio
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from corral.cli import build_parser
from corral.epistemic_analysis import (
    AnnotatedGraph,
    AnnotationBackend,
    AnnotationConfig,
    cli,
    workflow,
)
from corral.epistemic_analysis.annotators import llm
from corral.epistemic_analysis.annotators.jev import (
    DEFAULT_ENDPOINT,
    MAX_WINDOW,
    JevClient,
)
from corral.epistemic_analysis.decisions import DEFAULT_DECISIONS_MODEL
from corral.epistemic_analysis.sources.files import load_annotations, load_path
from corral.epistemic_analysis.sources.inputs import resolve_inputs
from corral.epistemic_analysis.sources.langfuse import (
    download_locator,
    parse_langfuse_url,
)
from corral.epistemic_analysis.trace_checks import TRACE_CHECK_NAMES

URL = "https://cloud.langfuse.com/project/project/sessions/session#observation=one"
MESSAGES = [
    {"role": "user", "content": "Find X"},
    {"role": "assistant", "content": "I think X."},
]


def rows():
    common = {
        "traceId": "trace",
        "sessionId": "session",
        "projectId": "project",
        "startTime": "2026-09-01T00:00:00Z",
        "endTime": "2026-09-01T00:00:05Z",
    }
    return [
        {
            **common,
            "id": "root",
            "parentObservationId": None,
            "type": "AGENT",
            "input": None,
            "output": {"status": "submitted", "answer": "X"},
        },
        {
            **common,
            "id": "generation",
            "parentObservationId": "root",
            "type": "GENERATION",
            "model": "agent-model",
            "input": MESSAGES[:1],
            "output": MESSAGES[1:],
        },
    ]


def write_trace(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"messages": MESSAGES}))
    return path


def invoke(*options):
    args = build_parser().parse_args(["analyze", *map(str, options)])
    return asyncio.run(args._handler(args))


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        [
            "--window",
            "15",
            "--overlap",
            "3",
            "--max-nodes-per-window",
            "80",
            "--tool-output-max-chars",
            "8000",
            "--max-request-tokens",
            "64000",
            "--no-strict-support",
        ],
    ],
)
def test_decisions_graph_shares_llm_graph_configuration(overrides):
    parser = build_parser()
    llm_args = parser.parse_args(
        ["analyze", "--input", "trace.json", "--annotator", "llm", *overrides]
    )
    jev_args = parser.parse_args(
        ["analyze", "--input", "trace.json", "--annotator", "jev", *overrides]
    )
    llm_config = cli.validate_options(llm_args)
    jev_config = cli.validate_options(jev_args)
    assert jev_config.model == DEFAULT_DECISIONS_MODEL
    assert llm_config.model == "anthropic/claude-sonnet-4-6"
    assert replace(jev_config, model=llm_config.model) == llm_config
    if not overrides:
        assert jev_config == AnnotationConfig(DEFAULT_DECISIONS_MODEL)
    for name in ("concurrency", "timeout", "max_retries"):
        assert getattr(jev_args, name) == getattr(llm_args, name)
    assert jev_args.jev_endpoint == DEFAULT_ENDPOINT
    assert cli.create_backend(jev_args).provenance["api"] == "openai-decisions"


@pytest.mark.parametrize("endpoint_flag", ["--decisions-endpoint", "--jev-endpoint"])
def test_decisions_graph_connection_and_model_can_be_overridden(endpoint_flag):
    args = build_parser().parse_args(
        [
            "analyze",
            "--input",
            "trace.json",
            "--annotator",
            "jev",
            "--model",
            "pinned-decisions-model",
            endpoint_flag,
            "https://example.test/v1/decisions",
        ]
    )
    assert cli.validate_options(args).model == "pinned-decisions-model"
    assert (
        cli.create_backend(args).provenance["endpoint"]
        == "https://example.test/v1/decisions"
    )


def test_decisions_window_limit_leaves_room_for_none_support_choice():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze",
            "--input",
            "trace.json",
            "--annotator",
            "jev",
            "--window",
            str(MAX_WINDOW),
        ]
    )
    assert cli.validate_options(args).window == 254
    args.window += 1
    with pytest.raises(ValueError, match="window <= 254"):
        cli.validate_options(args)


def test_relative_mixed_lists_comments_spaces_fragments_and_diagnostics(
    tmp_path, monkeypatch
):
    trace = write_trace(tmp_path / "a trace.json")
    listing = tmp_path / "inputs.txt"
    listing.write_text(f"  # comment\n\n a trace.json \n {URL} \na trace.json\n")
    monkeypatch.chdir(tmp_path.parent)
    entries = resolve_inputs(input_list=str(listing))
    assert [entry.kind for entry in entries] == ["path", "langfuse", "path"]
    assert entries[0].path == trace
    assert entries[0].line_number == 3
    assert entries[1].locator == URL
    assert entries[1].remote.identity == "session"
    normalized = load_path(entries[0])[0]
    assert normalized.source.line_number == 3
    assert normalized.source.locator == "a trace.json"
    assert normalized.message_sources == [{"index": 0}, {"index": 1}]
    listing.write_text(
        "a trace.json\nmissing.json\nhttps://example.org/not-supported\n"
    )
    with pytest.raises(ValueError) as error:
        resolve_inputs(input_list=str(listing))
    assert f"{listing}:2" in str(error.value)
    assert f"{listing}:3" in str(error.value)


def test_source_validation_and_directory_metadata(tmp_path):
    trace = write_trace(tmp_path / "arbitrary" / "nested" / "directory" / "trace.json")
    document = json.loads(trace.read_text())
    document["grouping"] = {
        "agent_model": "agent",
        "environment": "environment",
        "level": "level",
    }
    trace.write_text(json.dumps(document))
    source = load_path(resolve_inputs([str(tmp_path)])[0])[0]
    assert source.grouping.agent_model == "agent"
    assert source.grouping.environment == "environment"
    assert source.grouping.level == "level"
    assert source.source.identity == trace.as_uri()
    for inputs, source in [([str(trace)], "langfuse"), ([URL], "path")]:
        with pytest.raises(ValueError, match="Expected"):
            resolve_inputs(inputs, source=source)
    with pytest.raises(ValueError, match="either"):
        resolve_inputs([str(trace)], "inputs.txt")


@pytest.mark.parametrize(
    ("url", "kind", "identity"),
    [
        (URL, "sessions", "session"),
        (
            "https://host/base/project/project/traces/a%2Fb?display=details#one",
            "traces",
            "a/b",
        ),
    ],
)
def test_supported_langfuse_urls(url, kind, identity):
    selected = parse_langfuse_url(url)
    assert selected.kind == kind
    assert selected.identity == identity


@pytest.mark.parametrize(
    "url",
    [
        "https://host/traces/x",
        "https://host/project/p/traces",
        "ftp://host/project/p/traces/t",
        "https://secret@host/project/p/traces/t",
    ],
)
def test_unsupported_urls(url):
    with pytest.raises(ValueError, match="Unsupported"):
        parse_langfuse_url(url)


def test_url_project_and_host_validation(monkeypatch):
    locator = parse_langfuse_url(URL)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://different.example")
    with pytest.raises(ValueError, match="deployment"):
        download_locator(locator)
    monkeypatch.setenv("LANGFUSE_BASE_URL", locator.base_url)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    sdk = pytest.importorskip("langfuse.api.client")
    projects = SimpleNamespace(
        get=lambda: SimpleNamespace(
            model_dump=lambda **_: {"data": [{"id": "different"}]}
        )
    )
    monkeypatch.setattr(
        sdk, "LangfuseAPI", lambda **kwargs: SimpleNamespace(projects=projects)
    )
    with pytest.raises(ValueError, match="project"):
        download_locator(locator)


@pytest.mark.parametrize("source_kind", ["path", "langfuse"])
@pytest.mark.parametrize("annotator", ["llm", "jev"])
def test_all_four_source_backend_combinations(
    tmp_path, monkeypatch, source_kind, annotator
):
    origin = str(write_trace(tmp_path / "trace.json")) if source_kind == "path" else URL
    monkeypatch.setattr(workflow, "download_locator", lambda *args: rows())
    requests = []

    async def llm_call(model, system, user, **kwargs):
        requests.append(system)
        if system not in {llm.PASS_A_SYSTEM, llm.PASS_B_SYSTEM}:
            assert system == llm.trace_check_system(json.loads(user)["check"]["name"])
            return {
                "trace_check": {
                    "name": json.loads(user)["check"]["name"],
                    "status": "absent",
                    "explanation": "No supported incident.",
                    "support": [],
                }
            }
        return (
            {
                "nodes": [
                    {
                        "type": "H",
                        "time": 1,
                        "text": "I think X.",
                        "support": [{"msg_idx": 1, "quote": "I think X."}],
                    }
                ]
            }
            if system == llm.PASS_A_SYSTEM
            else {"edges": []}
        )

    async def decisions(self, model, state, questions, *, single_shot=False):
        requests.append(state)
        if single_shot:
            assert state["scope"] == "full_trace"
            assert list(questions) == [state["check"]["name"]]
            assert state["check"]["name"] in TRACE_CHECK_NAMES
            return {name: {"choice": "absent", "confidence": 1.0} for name in questions}
        answers = {}
        for key in questions:
            if key.startswith("edge_"):
                choice = "unrelated"
            elif key.startswith("same_"):
                choice = "distinct"
            else:
                span_key, node_type = key.rsplit("_", 1)
                kind = state["candidates"][span_key]["kind"]
                expected = {
                    "prose": "H",
                    "final": "F",
                    "action": "T",
                    "observation": "E",
                    "task": "E",
                }[kind]
                choice = "present" if node_type == expected else "absent"
            answers[key] = {"choice": choice, "confidence": 1.0}
        return answers

    monkeypatch.setattr(llm, "llm_json_call_async", llm_call)
    monkeypatch.setattr(JevClient, "evaluate", decisions)
    output = tmp_path / "out"
    assert invoke("--input", origin, "--annotator", annotator, "--output", output) == 0
    assert requests
    graph = json.loads(next((output / "annotations").glob("*.json")).read_text())
    assert graph["nodes"]
    assert graph["trace_checks_status"] == "complete"
    assert len(graph["trace_checks"]) == 11
    assert graph["source"]["locator"] == origin
    assert len(graph["message_sources"]) == len(graph["messages"])
    summary = json.loads((output / "analysis" / "annotation_summary.json").read_text())
    assert summary["n_selected"] == 1
    assert origin in (output / "analysis" / "annotation_summary.md").read_text()
    assert (output / "analysis" / "tables" / "productive_motifs.tex").exists()
    count = len(requests)
    assert invoke("--input", origin, "--annotator", annotator, "--output", output) == 0
    assert len(requests) == count  # exact fingerprint reuse
    assert (
        invoke(
            "--input",
            origin,
            "--annotator",
            annotator,
            "--model",
            "changed",
            "--output",
            output,
        )
        == 0
    )
    assert len(requests) > count
    assert len(list((output / "annotations").glob("*.json"))) == 2
    # The manifest selects only the most recent invocation; no old config leaks.
    summary = json.loads((output / "analysis" / "annotation_summary.json").read_text())
    assert summary["n_selected"] == 1
    assert len(summary["configurations"]) == 1
    assert (
        next(iter(summary["configurations"].values()))["annotation_configuration"][
            "model"
        ]
        == "changed"
    )

    def forbidden(*args):
        pytest.fail("analysis-only must not initialize clients or download")

    monkeypatch.setattr(cli, "create_backend", forbidden)
    monkeypatch.setattr(workflow, "download_locator", forbidden)
    assert (
        invoke(
            "--input", output, "--skip-annotate", "--output", tmp_path / "reanalysis"
        )
        == 0
    )


def test_deduplication_dry_run_download_only_and_invalid_preflight(
    tmp_path, monkeypatch
):
    local = write_trace(tmp_path / "trace.json")
    listing = tmp_path / "list.txt"
    listing.write_text(f"trace.json\ntrace.json\n{URL}\n{URL}\n")
    monkeypatch.setattr(workflow, "download_locator", lambda *args: rows())
    monkeypatch.setattr(
        cli, "create_backend", lambda *args: pytest.fail("no backend needed")
    )
    output = tmp_path / "out"
    assert invoke("--input-list", listing, "--dry-run", "--output", output) == 0
    assert not output.exists()
    assert invoke("--input-list", listing, "--download-only", "--output", output) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["traces"]) == 2
    broken = tmp_path / "bad.json"
    broken.write_text('{"messages":"not-a-list"}')
    with pytest.raises(ValueError, match="before annotation"):
        invoke("--input", local, "--input", broken, "--output", tmp_path / "bad-output")
    assert not (tmp_path / "bad-output").exists()


def test_failed_annotation_is_excluded_but_recorded(tmp_path, monkeypatch):
    local = write_trace(tmp_path / "trace.json")

    async def failure(**kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        cli,
        "create_backend",
        lambda args: AnnotationBackend(failure, failure, {"annotator": "test"}),
    )
    output = tmp_path / "out"
    assert invoke("--input", local, "--output", output) == 1
    summary = json.loads((output / "analysis" / "annotation_summary.json").read_text())
    stats = next(iter(summary["configurations"].values()))["overall"]
    assert stats["n_traces"] == 0
    assert stats["status_counts"] == {"failed": 1}
    assert stats["n_excluded"] == 1


def test_multiple_saved_configurations_require_explicit_selection(tmp_path):
    for i in range(2):
        (tmp_path / f"{i}.annotated.json").write_text(
            json.dumps(
                AnnotatedGraph(
                    [], [], provenance={"configuration_id": str(i)}
                ).to_dict()
            )
        )
    entry = resolve_inputs([str(tmp_path)])[0]
    with pytest.raises(ValueError, match="Multiple annotation configurations"):
        load_annotations(entry)
    assert len(load_annotations(entry, configuration="0")) == 1


def test_provider_and_cli_dependencies_are_not_imported_by_library():
    import subprocess

    # dotenv is imported transitively by the core agent/tool dependencies.
    # Annotation providers and the CLI must still remain unloaded.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import corral.epistemic_analysis; assert not {'litellm', 'langfuse', 'fire'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_downloaded_traces_preserve_origin_and_mapping_when_loaded_locally(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(workflow, "download_locator", lambda *args: rows())
    output = tmp_path / "download"
    assert invoke("--input", URL, "--download-only", "--output", output) == 0
    saved = next((output / "traces").glob("*.json"))
    trace = load_path(resolve_inputs([str(saved)])[0])[0]
    assert trace.source.locator == URL
    assert trace.source.provider == "langfuse"
    assert trace.source.identity.endswith("/traces/trace")
    assert trace.message_sources[1]["observation_id"] == "generation"
    assert trace.snapshot


def test_resolved_versions_are_preserved_and_pattern_changes_need_no_annotation(
    tmp_path, monkeypatch
):
    from corral.epistemic_analysis import schema
    from corral.epistemic_analysis.annotators.base import record_model

    local = write_trace(tmp_path / "trace.json")
    version = "provider-v1"

    async def extract_nodes(messages, **kwargs):
        record_model(version)
        return [
            {
                "node_id": "N1",
                "type": "H",
                "time": 1,
                "text": "I think X.",
                "support": [{"msg_idx": 1, "quote": "I think X."}],
            }
        ], []

    async def extract_edges(**kwargs):
        return [], []

    monkeypatch.setattr(
        cli,
        "create_backend",
        lambda args: AnnotationBackend(
            extract_nodes, extract_edges, {"annotator": "test"}
        ),
    )
    output = tmp_path / "out"
    assert invoke("--input", local, "--output", output) == 0
    version = "provider-v2"
    assert invoke("--input", local, "--output", output, "--force") == 0
    saved = [
        json.loads(path.read_text()) for path in (output / "annotations").glob("*.json")
    ]
    assert {tuple(g["provenance"]["resolved_models"]) for g in saved} == {
        ("provider-v1",),
        ("provider-v2",),
    }
    monkeypatch.setattr(schema, "PATTERN_VERSION", "future-patterns")
    monkeypatch.setattr(
        cli,
        "create_backend",
        lambda args: pytest.fail("no annotation for changed patterns"),
    )
    reanalysis = tmp_path / "reanalysis"
    assert invoke("--input", output, "--skip-annotate", "--output", reanalysis) == 0
    summary = json.loads(
        (reanalysis / "analysis" / "annotation_summary.json").read_text()
    )
    assert summary["pattern_version"] == "future-patterns"
    assert next(iter(summary["configurations"].values()))["annotation_configuration"][
        "resolved_models"
    ] == ["provider-v2"]


def test_branch_reconstruction_failure_retains_raw_snapshot(tmp_path, monkeypatch):
    records = rows()
    records[-1]["metadata"] = {"agent_run_id": "run-1"}
    records.append(
        {
            **records[-1],
            "id": "another-generation",
            "metadata": {"agent_run_id": "run-2"},
        }
    )
    monkeypatch.setattr(workflow, "download_locator", lambda *args: records)
    monkeypatch.setattr(
        cli,
        "create_backend",
        lambda args: pytest.fail("invalid inputs must not annotate"),
    )
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="separate conversation reconstruction"):
        invoke("--input", URL, "--output", output)
    assert list((output / "snapshots").glob("*.json"))
    assert not (output / "annotations").exists()


def test_analysis_only_cli_never_imports_provider_libraries(tmp_path):
    import subprocess

    saved = tmp_path / "saved.annotated.json"
    saved.write_text(json.dumps(AnnotatedGraph([], []).to_dict()))
    script = """
import sys
class BlockProviders:
    def find_spec(self, fullname, *args):
        if fullname in {"litellm", "langfuse"}:
            raise AssertionError("Unexpected provider import: " + fullname)
sys.meta_path.insert(0, BlockProviders())
from corral.cli import main
assert main(["analyze", "--input", sys.argv[1], "--skip-annotate", "--output", sys.argv[2]]) == 0
"""
    subprocess.run(
        [sys.executable, "-c", script, str(saved), str(tmp_path / "output")], check=True
    )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--concurrency", "0"),
        ("--batch-size", "0"),
        ("--overlap", "20"),
        ("--window", "255"),
        ("--min-confidence", "1.1"),
        ("--timeout", "inf"),
        ("--max-retries", "0"),
        ("--window", "2.5"),
        ("--overlap", "0.5"),
    ],
)
def test_invalid_cli_settings_fail_before_processing(tmp_path, option, value):
    trace = write_trace(tmp_path / "trace.json")
    with pytest.raises((ValueError, SystemExit)):
        invoke("--input", trace, "--annotator", "jev", "--dry-run", option, value)


def test_directory_names_do_not_supply_grouping_metadata(tmp_path):
    write_trace(tmp_path / "agent" / "environment" / "level" / "trace.json")
    trace = load_path(resolve_inputs([str(tmp_path)])[0])[0]
    assert trace.grouping.agent_model is None
    assert trace.grouping.environment is None
    assert trace.grouping.level is None
