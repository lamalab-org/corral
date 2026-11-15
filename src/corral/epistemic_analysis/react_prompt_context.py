"""Recover public agent startup context from a pinned repository source tree.

Only schema declarations, prompt templates and pure rendering/selection functions
are evaluated. Tool bodies are replaced with raising stubs: reconstruction never
starts an environment, reads a sample, executes a tool, or loads a scorer.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from promptstore import Prompt

from .task_prompt_context import ENVIRONMENT_FILES


def _execute(nodes, namespace, filename):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)


def _definition(tree, name):
    matches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        and n.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one pinned definition of {name}")
    return copy.deepcopy(matches[0])


def _schema_declarations(tree, namespace, filename):
    nodes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not any(
            (isinstance(d, ast.Name) and d.id == "tool")
            or (
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id == "tool"
            )
            for d in node.decorator_list
        ):
            continue
        stub = copy.deepcopy(node)
        doc = ast.get_docstring(stub, clean=False)
        if not doc:
            raise ValueError(f"Missing tool documentation: {stub.name}")
        stub.body = [
            ast.Expr(value=ast.Constant(doc)),
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                    args=[ast.Constant("Schema reconstruction cannot execute tools")],
                    keywords=[],
                ),
                cause=None,
            ),
        ]
        nodes.append(stub)
    _execute(nodes, namespace, filename)


class PinnedAgentContext:
    """Render annotation context without constructing or modifying an agent."""

    def __init__(self, root, revision, environment):
        self.root, self.revision, self.environment = Path(root), revision, environment
        self.sources = []
        # Keep each revision's schema implementation isolated from the checkout.
        name = (
            "_corral_recovered_tools_"
            + hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        )
        module = types.ModuleType(name)
        sys.modules[name] = module
        ns = module.__dict__
        description = self._read("src/corral/core/tool_description.py")
        exec(
            compile(
                description,
                str(self.root / "src/corral/core/tool_description.py"),
                "exec",
            ),
            ns,
        )
        tool_path = "src/corral/core/tool.py"
        tool_tree = ast.parse(self._read(tool_path))
        tool_tree.body = [
            n
            for n in tool_tree.body
            if not (
                isinstance(n, ast.ImportFrom)
                and n.module == "corral.core.tool_description"
            )
        ]
        exec(compile(tool_tree, str(self.root / tool_path), "exec"), ns)
        self.namespace = ns
        relative = Path(ENVIRONMENT_FILES[environment]).with_name("tools.py")
        tools_path = f"tasks/{environment}/{relative}"
        tree = ast.parse(self._read(tools_path))
        _schema_declarations(tree, ns, self.root / tools_path)
        _execute([_definition(tree, "create_tools")], ns, self.root / tools_path)
        self.pool = ns["create_tools"]()

        core_path = "src/corral/core/environment.py"
        core = ast.parse(self._read(core_path))
        if environment == "resistor_network":
            fs_path = "src/corral/workspace/filesystem.py"
            fs = ast.parse(self._read(fs_path))
            constants = {
                n.targets[0].id: ast.literal_eval(n.value)
                for n in fs.body
                if isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "PUBLIC_WORKSPACE_ROOT"
            }
            ns.update(constants)
            _schema_declarations(fs, ns, self.root / fs_path)
            default = _definition(core, "default_file_tools")
            selection = next(n.value for n in default.body if isinstance(n, ast.Return))
            workspace_tools = eval(
                compile(ast.Expression(selection), str(self.root / core_path), "eval"),
                {**ns, "tools": ns},
            )
        else:
            workspace_tools = {}
        ns.update(
            dataclass=dataclass,
            field=field,
            logger=logging.getLogger(__name__),
            default_file_tools=lambda _workspace: workspace_tools,
        )
        _execute([_definition(core, "Toolset")], ns, self.root / core_path)
        env_path = f"tasks/{environment}/{ENVIRONMENT_FILES[environment]}"
        env = ast.parse(self._read(env_path))
        calls = [
            n
            for n in ast.walk(env)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "Toolset"
        ]
        if len(calls) != 1:
            raise ValueError("Expected one environment tool selection policy")
        self.toolset = eval(
            compile(ast.Expression(calls[0]), str(self.root / env_path), "eval"),
            {
                **ns,
                "create_tools": lambda: self.pool,
                "tool_pool": self.pool,
                "subtask_level": False,
                "taskgroup_common_tools": None,
            },
        )

        action_path = "src/corral/core/action.py"
        action = ast.parse(self._read(action_path))
        ns.update(SUBMIT_ANSWER_TOOL_NAME="submit_answer", Mapping=Mapping)
        _execute(
            [
                _definition(action, "submit_answer_tool"),
                _definition(action, "with_submit_answer_tool"),
            ],
            ns,
            self.root / action_path,
        )
        prompt_path = "src/corral/agents/prompt_utils.py"
        prompt = ast.parse(self._read(prompt_path))
        ns.update(
            LiteLLMMessage=dict,
            json=json,
            provider_messages=list,
        )
        _execute(
            [
                _definition(prompt, "create_prompt"),
                _definition(prompt, "build_user_content"),
            ],
            ns,
            self.root / prompt_path,
        )
        react_path = "src/corral/agents/react.py"
        _execute(
            [_definition(ast.parse(self._read(react_path)), "_initial_messages")],
            ns,
            self.root / react_path,
        )
        support_path = "src/corral/agents/_epistemic.py"
        _execute(
            [_definition(ast.parse(self._read(support_path)), "_tools")],
            ns,
            self.root / support_path,
        )
        self.system = self._template("system_prompt/system_prompt")
        self.user = self._template("react/user_prompt")

    def _read(self, relative):
        path = self.root / relative
        data = path.read_bytes()
        self.sources.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(data).hexdigest(),
                "git_commit": self.revision,
            }
        )
        return data.decode()

    def _template(self, name):
        text = self._read(f"src/corral/agents/prompts/{name}/prompt.md")
        if not text.startswith("---\n"):
            raise ValueError("Missing prompt template metadata")
        return Prompt.from_markdown(text)

    def recover(
        self,
        task,
        task_prompt,
        task_sources,
        runtime_options,
        *,
        enable_surrender=False,
        agent_name="react",
        max_iterations=None,
        native_workspace=None,
    ):
        forbidden = {
            "session_capabilities",
        } & runtime_options.keys()
        if forbidden:
            raise ValueError(
                f"Reconstruction requires explicit support for runtime prompt overrides: {sorted(forbidden)}"
            )
        public = SimpleNamespace(
            name=task["id"],
            tools=task.get("tools", []),
            excluded_tools=task.get("excluded_tools", []),
        )
        unknown = set(public.tools) - set(self.pool)
        if unknown:
            raise ValueError(f"Task requests missing tools: {sorted(unknown)}")
        tools = self.toolset.resolve(public, "/workspace")
        schemas = [t.get_openai_tool_format() for t in tools.values()]
        session = SimpleNamespace(
            prompt=task_prompt,
            examples=runtime_options.get("examples", ()),
            surrender_allowed=enable_surrender,
            messages=(),
            tools=self.namespace["with_submit_answer_tool"](schemas),
        )
        agent = SimpleNamespace(
            system_prompt=(
                Prompt(content=runtime_options["system_prompt"], version=1).fill({})
                if runtime_options.get("system_prompt") is not None
                else self.system.fill({})
            ),
            user_prompt=self.user,
            surrender_prompt=(
                Prompt(content=runtime_options["surrender_prompt"], version=1)
                if runtime_options.get("surrender_prompt") is not None
                else None
            ),
        )
        agent._tools = lambda s: self.namespace["_tools"](agent, s)
        catalog = agent._tools(session)
        messages = self._render_agent(
            agent_name,
            agent,
            session,
            runtime_options,
            max_iterations,
            native_workspace,
        )
        refs = [*task_sources, *self.sources]
        sources = [
            {
                "index": None,
                "origin": f"reconstructed_{agent_name}_initial_instructions",
                "instruction_kind": message["role"],
                "source_evidence": refs,
                "prompt_context": {
                    "source": "reconstructed",
                    "sdk_hidden_system": agent_name in {"codex", "claude-code"},
                },
                "description": "Recovered from the original task, run configuration and pinned agent prompt renderer; not a recorded model request.",
            }
            for message in messages
        ]
        return messages, sources, catalog

    def _load(self, relative, *names):
        tree = ast.parse(self._read(relative))
        _execute([_definition(tree, name) for name in names], self.namespace, relative)
        return tree

    def _render_agent(self, name, agent, session, options, max_iterations, workspace):
        ns = self.namespace
        templates = {
            "react": "react/user_prompt",
            "tool-calling": "tool_calling/user_prompt",
            "epistemic": "epistemic/user_prompt",
            "llm-planner": "planner_prompt/user_prompt",
        }
        if name in templates:
            template = options.get("user_prompt") or templates[name]
            relative = Path("src/corral/agents/prompts") / template / "prompt.md"
            agent.user_prompt = (
                self._template(template)
                if len(template) < 200
                and "\n" not in template
                and relative.is_relative_to("src/corral/agents/prompts")
                and ".." not in relative.parts
                and (self.root / relative).is_file()
                else Prompt(content=template, version=1)
            )
        if name == "epistemic":
            tree = ast.parse(self._read("src/corral/agents/epistemic.py"))
            constants = {
                node.targets[0].id: ast.literal_eval(node.value)
                for node in tree.body
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id
                in {"EPISTEMIC_SYSTEM_PROMPT", "EPISTEMIC_PROMPT"}
            }
            if options.get("system_prompt") is None:
                agent.system_prompt = Prompt(
                    content=constants["EPISTEMIC_SYSTEM_PROMPT"], version=1
                ).fill({})
            agent.system_prompt += "\n\n" + constants["EPISTEMIC_PROMPT"]
        if name in {"react", "tool-calling", "epistemic"}:
            module = "react" if name == "react" else "tool_calling"
            self._load(f"src/corral/agents/{module}.py", "_initial_messages")
            return ns["_initial_messages"](agent, session)
        if name == "llm-planner":
            if max_iterations is None:
                raise ValueError(
                    "Planner prompt reconstruction requires max_iterations"
                )
            tree = ast.parse(self._read("src/corral/agents/llm_planner.py"))
            initial = next(
                node
                for node in _definition(tree, "run_session").body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "initial" for t in node.targets
                )
            )
            scope = {
                **ns,
                "self": agent,
                "session": session,
                "iteration_limit": max_iterations,
            }
            _execute([initial], scope, "llm_planner.py")
            return scope["initial"]
        if name == "terminus":
            from typing import Annotated, Any

            import pydantic

            ns.update(
                {
                    key: getattr(pydantic, key)
                    for key in (
                        "BaseModel",
                        "ConfigDict",
                        "Field",
                        "WithJsonSchema",
                        "field_validator",
                        "model_validator",
                    )
                },
                Annotated=Annotated,
                Any=Any,
            )
            self._load(
                "src/corral/agents/terminus.py",
                "TerminusToolCall",
                "TerminusResponse",
                "_initial_prompt",
            )
            agent.max_actions_per_turn = options.get("max_actions_per_turn", 1)
            return [
                {"role": "system", "content": agent.system_prompt},
                {
                    "role": "user",
                    "content": ns["_initial_prompt"](
                        agent,
                        session.prompt,
                        ns["with_submit_answer_tool"](session.tools),
                        session.surrender_allowed,
                        session.examples,
                    ),
                },
            ]
        if name in {"codex", "claude-code"}:
            timeout = self._read("src/corral/agents/_mcp_timeout.py")
            exec(compile(timeout, "_mcp_timeout.py", "exec"), ns)
            agent.tool_timeout_s = options.get(
                "tool_timeout_s", ns["DEFAULT_TOOL_TIMEOUT_S"]
            )
            module = name.replace("-", "_")
            method = (
                "_developer_instructions" if name == "codex" else "_build_system_prompt"
            )
            tree = self._load(f"src/corral/agents/{module}.py", method)
            constants = [
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"_MCP_SERVER_NAME", "_FILE_WRITING_TOOLS"}
            ]
            _execute(constants, ns, module)
            ns["SURRENDER_SENTINEL"] = "SURRENDER"
            if name == "codex":
                text = ns[method](
                    agent,
                    session.surrender_allowed,
                    [t["function"]["name"] for t in session.tools],
                )
            else:
                if agent.surrender_prompt is not None:
                    agent.surrender_prompt = agent.surrender_prompt.fill({})
                text = ns[method](agent, session.surrender_allowed)["append"]
            if workspace:
                target = (
                    "developer_instructions"
                    if name == "codex"
                    else "options.system_prompt['append']"
                )
                suffixes = [
                    n.value
                    for n in ast.walk(tree)
                    if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == target
                ]
                if len(suffixes) != 1:
                    raise ValueError("Cannot reconstruct SDK workspace instructions")
                text += eval(
                    compile(ast.Expression(suffixes[0]), module, "eval"),
                    {
                        "run": SimpleNamespace(native_workspace=workspace),
                        "cwd": workspace,
                    },
                )
            return [
                {"role": "developer" if name == "codex" else "system", "content": text},
                {"role": "user", "content": session.prompt},
            ]
        raise ValueError(
            f"{name} requires recorded worker/SDK prompts; they cannot be inferred from a task name"
        )


# Keep the existing ReAct recovery API used by viewer exports.
PinnedReActContext = PinnedAgentContext
