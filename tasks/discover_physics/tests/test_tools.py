from discover_physics.tools import create_tools


class TestRunExperimentSchema:
    def test_tool_pool_has_run_experiment(self):
        tools = create_tools()
        assert set(tools) == {"run_experiment"}

    def test_world_config_is_hidden_from_the_agent_schema(self):
        tool = create_tools()["run_experiment"]
        schema = tool.params_json_schema
        assert "world_config" not in schema.get("properties", {})
        assert "world_config" not in schema.get("required", [])

    def test_experiments_is_the_only_visible_argument(self):
        tool = create_tools()["run_experiment"]
        schema = tool.params_json_schema
        assert set(schema.get("properties", {})) == {"experiments"}
        assert schema["properties"]["experiments"]["type"] == "string"
        assert schema.get("required") == ["experiments"]

    def test_hidden_args_declared_on_tool(self):
        tool = create_tools()["run_experiment"]
        assert "world_config" in tool.hidden_args

    def test_openai_tool_format_never_mentions_world_config(self):
        tool = create_tools()["run_experiment"]
        openai_format = tool.get_openai_tool_format()
        assert "world_config" not in str(openai_format)
