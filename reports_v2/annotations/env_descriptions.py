import ast
import json
import re
from pathlib import Path

# ================= PATHS =================

ROOT_PATH = Path(__file__).parent.parent.parent
TARGET_PATH = Path(__file__).parent / "env_descriptions"
TASKS_PATH = ROOT_PATH / "tasks"

# ================= HELPERS =================


def level_heading(level: str) -> str:
    return rf"\subsubsection{{{level.replace('_', ' ').title()}}}"


def latex_escape(s: str) -> str:
    return (
        s.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("Å", r"\AA{}")
        .replace("g/cm^3", "g/cm$^3$")
    )


def load_single_task(path: Path) -> dict:
    data = json.loads(path.read_text())
    return next(iter(data.values()))


def load_subtasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    # preserve order explicitly
    return [data[k] for k in sorted(data.keys())]


# ================= RENDERERS =================


def render_input_block(input_dict: dict) -> str:
    if not input_dict:
        return r"\textbf{Input Parameters:} None"

    pretty = json.dumps(input_dict, indent=4)
    return rf"""
\textbf{{Input Parameters:}}

\begin{{lstlisting}}[style=json]
{pretty}
\end{{lstlisting}}
"""


def render_main_task(task: dict) -> str:
    return rf"""
\begin{{tcolorbox}}[
    title={latex_escape(task['name'])},
    breakable,
    enhanced,
    colback=blue!5,
    colframe=blue!75!black
]

{latex_escape(task['description'])}

\medskip
{render_input_block(task.get("initial_input", {}))}
"""


def render_subtask(subtask: dict, index: int) -> str:
    return rf"""
% ================= SUBTASK {index} ===================
\begin{{tcolorbox}}[
    title=Subtask {index} : {latex_escape(subtask['name'])},
    breakable,
    enhanced,
    colback=gray!5,
    colframe=gray!60!black
]

\textbf{{Description:}} {latex_escape(subtask['description'])}

\medskip
{render_input_block(subtask.get("initial_input", {}))}

\end{{tcolorbox}}
"""


def generate_level_latex(task_file: Path, subtask_file: Path) -> str:
    task = load_single_task(task_file)
    subtasks = load_subtasks(subtask_file)

    blocks = [render_main_task(task)]

    for i, subtask in enumerate(subtasks, start=1):
        blocks.append(render_subtask(subtask, i))

    blocks.append(r"\end{tcolorbox}")  # close main task box
    return "\n".join(blocks)


def extract_args_and_returns(doc: str):
    result = {"args": {}, "returns": None}
    if not doc:
        return result

    # ---------- Extract ARGS section ----------
    args_block = re.search(r"Args:(.*?)(Returns:|\[RAISES|\Z)", doc, re.S)
    if args_block:
        block = args_block.group(1)

        # capture each param block
        for m in re.finditer(
            r"\n\s*([A-Za-z0-9_]+)\s*:\s([\s\S]*?)(?=\n\s*[A-Za-z0-9_]+\s*:|\nReturns:|\[RAISES|\Z)",
            block,
            re.S,
        ):
            name = m.group(1)
            param_block = m.group(2)

            detailed = re.search(
                r"\[ARGS_DETAILED\](.*?)(?:\[/ARGS_DETAILED\])", param_block, re.S
            )

            if detailed:
                text = detailed.group(1).strip()
                result["args"][name] = " ".join(
                    line.strip() for line in text.splitlines()
                )

    # ---------- Extract RETURNS ----------
    returns_block = re.search(r"Returns:(.*?)(\[RAISES|\Z)", doc, re.S)
    if returns_block:
        block = returns_block.group(1)

        detailed = re.search(
            r"\[ARGS_DETAILED\](.*?)(?:\[/ARGS_DETAILED\])", block, re.S
        )
        if detailed:
            text = detailed.group(1).strip()
            result["returns"] = " ".join(line.strip() for line in text.splitlines())

    return result


def parse_file(path):
    out = []
    with Path.open(path) as f:
        module = ast.parse(f.read())

    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            info = extract_args_and_returns(doc)

            out.append(
                {
                    "function": node.name,
                    "args": info["args"],
                    "returns": info["returns"],
                }
            )
    return out


def render_tools_table_from_results(results):
    def render_single_tool(entry):
        name = entry["function"]
        args = entry["args"]
        returns = entry["returns"] or "Not specified."

        args_lines = (
            "\n".join(
                rf"        \item \texttt{{\detokenize{{{k}}}}}: {latex_escape(v)}"
                for k, v in args.items()
            )
            if args
            else "        \\textit{None}"
        )

        return rf"""
\textbf{{\texttt{{\detokenize{{{name}}}}}}} \\
\begin{{description}}
  \item[\textbf{{Arguments:}}]
  \begin{{description}}
{args_lines}
  \end{{description}}

  \item[\textbf{{Return/Behavior:}}] {latex_escape(returns)}
\end{{description}}
\\\midrule
"""

    tools_body = "\n".join(render_single_tool(r) for r in results)

    return (
        r"""
\subsubsection{Tools}

\Cref{tab:lammps-tools} provides a detailed description of the tools used in the \md environment, including what their arguments are and what they return.

\begin{longtable}{p{0.9\textwidth}}
\caption{\textbf{Tools for the \md environment}. The table describes each tool available in this environment, including its arguments and return values.}
\label{tab:lammps-tools} \\
\toprule
\textbf{Tool} \\
\midrule
\endfirsthead

\toprule
\textbf{Tool} \\
\midrule
\endhead

\bottomrule
\endfoot
"""
        + tools_body
        + r"""
\end{longtable}
"""
    )


# ================= MAIN DRIVER =================

environments = [
    "afm",
    "catalyst",
    "corral_md",
    "kinetic_modeling",
    "ml",
    "retrosynthesis",
    "spectra_elucidation",
]

TARGET_PATH.mkdir(parents=True, exist_ok=True)

all_latex = []

for env in environments:
    if env == "corral_md":
        env_path = TASKS_PATH / env / "environments"
        tasks = ["surface_energy", "melting", "quenching"]
        for task_name in tasks:
            for level in ["level_1", "level_2", "level_3"]:
                all_latex.append(level_heading(level))
                level_path = env_path / task_name / level
                task_dir = level_path / "tasks"
                subtask_dir = level_path / "subtasks"
                if not task_dir.exists():
                    continue
                task_file = next(task_dir.glob("*.json"))
                subtask_file = next(subtask_dir.glob("*.json"))
                latex_block = generate_level_latex(task_file, subtask_file)
                all_latex.append(latex_block)
    else:
        continue
    tool_file = TASKS_PATH / env / "src" / env / "tools.py"
    results = parse_file(tool_file)
    all_latex.append(render_tools_table_from_results(results))
    output_file = TARGET_PATH / f"{env}.tex"
    output_file.write_text("\n\n".join(all_latex))


# for level in levels:
#     all_latex.append(level_heading(level))

#     for env in environments:
#         if env != "corral_md":
#             continue

#         env_path = TASKS_PATH / env / "environments"

#         for task_name in tasks:
#             level_path = env_path / task_name / level
#             task_dir = level_path / "tasks"
#             subtask_dir = level_path / "subtasks"

#             if not task_dir.exists():
#                 continue

#             task_file = next(task_dir.glob("*.json"))
#             subtask_file = next(subtask_dir.glob("*.json"))

#             latex_block = generate_level_latex(task_file, subtask_file)
#             all_latex.append(latex_block)

#         # ==== TOOLS (OPTIONAL) ====
#         tool_file = TASKS_PATH / "corral_md" / "src" / "corral_md" / "tools.py"

#         results = parse_file(tool_file)
#         all_latex.append(render_tools_table_from_results(results))


# # ================= WRITE OUTPUT =================
# TARGET_PATH.mkdir(parents=True, exist_ok=True)
# output_file = TARGET_PATH / "md.tex"
# output_file.write_text("\n\n".join(all_latex))
