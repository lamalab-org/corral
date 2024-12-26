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