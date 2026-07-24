"""One-task, one-trial smoke test for Opus 4.8 against a live server."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _opus_shim  # noqa: F401

from corral.agents import ReActAgent, ToolCallingAgent
from corral.router import CorralRouter
from corral.run import CorralRunner

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
agent_kind = sys.argv[1]
port = int(sys.argv[2])
maxit = int(sys.argv[3]) if len(sys.argv) > 3 else 12

interface = CorralRouter(base_url=f"http://localhost:{port}")
tasks = interface.get_available_tasks()
print(f"SMOKE {agent_kind} port={port} tasks={len(tasks)}")
cls = {"react": ReActAgent, "toolcalling": ToolCallingAgent}[agent_kind]
agent = cls(model=MODEL, max_iterations=maxit, temperature=0.0)
out = Path(__file__).parent / "_smoke_artifacts" / f"{agent_kind}_{port}"
out.mkdir(parents=True, exist_ok=True)
runner = CorralRunner(interface, agent, checkpoint_dir=str(out / "checkpoints"))
result = runner.bench(
    task_ids=tasks[:1],
    trials_per_task=1,
    verbose=True,
    tool_verbosity="workflow",
    run_name=str(out / f"smoke-{agent_kind}-{port}"),
)
print("SMOKE OK", agent_kind, port)
