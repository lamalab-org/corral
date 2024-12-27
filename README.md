# Material Agent Benchmark

The system consists of three main components:

Environment Server (corral)- Hosts tasks and tools
Benchmark Interface (MatAgentBenchmark)- Communicates with the server
Agent - Solves tasks using available tools

## 1. Create environment and add tools example:
```bash
cd tasks/samplemath/samplemath
python -m env # start corral server
```

## 2. Create agent example:
```bash
cd agents/baseline
```


## 3. Benchmark Interface

```python
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark

interface = BenchmarkInterface()
agent = ClaudeAgent(api_key=os.getenv("ANTHROPIC_API_KEY")) # or any other agent
runner = MatAgentBenchmark(interface, agent)

result = runner.bench()
```
