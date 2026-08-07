Design {{count}} {{node_type}} preliminary experiment proposal(s).

Task:
{{task_prompt}}

Formulation:
{{formulation}}

Parent checkpoint:
{{parent}}

Branch workspaces, one per proposal in proposal order:
{{branch_workspaces}}

Research journal:
{{journal}}

Allowed tools:
{{tools}}

Each proposal must pursue a scientifically distinct, executable route and use
at most {{max_actions}} sequential actions. Every action needs a purpose and an
expected observation. Use only listed tools and encode arguments as a JSON
object string. When a tool accepts an output path, place that proposal's output
under its assigned branch workspace. A DEBUG proposal must diagnose the parent's actual failure; a
REFINE proposal must improve a successful route. Return exactly {{count}}
proposal(s) when possible.
