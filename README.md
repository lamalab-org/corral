# Material Agent Benchmark



## Create environment and add tools
```python
env = BaseEnvironment()
# [ available tools in environment]
#tools = env.get_available_tools()
tools = CalculatorTool() 
env.add_tool(tools)

# Create FastAPI app
app = create_environment_server(env)

```

## example

```bash
cd tasks/samplemath
python -m env
```

```
GET /tasks/{task_id}/guide - Get description related to the environment and tools
GET /tasks/{task_id}/prompt - Get the task prompt
GET /tasks/{task_id}/tools - List available tools
GET /tasks/{task_id}/state - Get current state
POST /tasks/{task_id}/tools/execute - Execute a tool
```