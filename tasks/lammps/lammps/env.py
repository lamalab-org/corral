from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from corral.base import Environment, Tool
from corral.server import create_benchmark_server
from tools import create_tools
import uvicorn
import json

from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from corral.base import Environment, Tool

from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from corral.base import Environment, Tool
import os

@dataclass
class TaskDefinition:
    """Definition of a task with its requirements and scoring"""
    name: str
    description: str
    tools: List[str]
    scoring_fn: Callable[[Dict], float]
    submission_format: Dict[str, str]
    # Either use output from another task or custom input
    input_from_task: Optional[str] = None
    initial_input: Optional[Dict[str, Any]] = None

@dataclass
class TaskGroup:
    """Container for related tasks"""
    group_id: str
    tasks: Dict[str, TaskDefinition]
    results: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)

    def get_task_input(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get input for a task either from another task or initial input"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        if task.input_from_task and task.input_from_task in self.results:
            return {"result": self.results[task.input_from_task]}
        return task.initial_input

    def store_result(self, task_id: str, result: Dict[str, Any], score: float) -> None:
        """Store task result and score"""
        self.results[task_id] = result
        self.scores[task_id] = score

    def get_task_dependencies(self) -> Dict[str, List[str]]:
        """Get dictionary of task dependencies"""
        dependencies = {}
        for task_id, task in self.tasks.items():
            deps = []
            if task.input_from_task:
                deps.append(task.input_from_task)
            dependencies[task_id] = deps
        return dependencies

class TaskEnvironment(Environment):
    """Environment that works with a task group"""
    def __init__(
        self,
        task_id: str,
        task_group: TaskGroup,
        available_tools: Dict[str, Tool]
    ):
        self.task_group = task_group
        self.task_id = task_id  # Individual task ID within the group
        self.available_tools = available_tools

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        # Initialize tools and environment
        self.tools = {}
        super().__init__(f"{task_group.group_id}_{task_id}")

        # Add required tools
        for tool_name in self.current_task.tools:
            if tool_name in available_tools:
                self.add_tool(available_tools[tool_name])

    def get_task_prompt(self) -> str:
        input_data = self.task_group.get_task_input(self.task_id)

        prompt = f"""Task: {self.current_task.name}
Description: {self.current_task.description}

Required submission format:
"""
        for key, desc in self.current_task.submission_format.items():
            prompt += f"- {key}: {desc}\n"

        prompt += "\nAvailable input data:\n"

        if self.current_task.input_from_task and self.current_task.input_from_task in self.task_group.results:
            # If this task depends on a previous task, show its result
            previous_result = self.task_group.results[self.current_task.input_from_task]
            if isinstance(previous_result, dict) and "answer" in previous_result:
                prompt += f"Previous task result: {previous_result['answer']}\n"
            else:
                prompt += f"Previous task result: {previous_result}\n"

        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                prompt += f"- {key}: {value}\n"

        if self.current_task.input_from_task:
            status = "available" if self.current_task.input_from_task in self.task_group.results else "not yet available"
            prompt += f"\nThis task uses output from task: {self.current_task.input_from_task} ({status})"

        return prompt

    def score(self) -> float:
        """Score the submitted answer"""
        if not self.state.submitted_answer:
            return 0.0

        try:
            # Clean the submission - take only the numerical answer part
            submission_str = self.state.submitted_answer.strip()
            print(f"Raw submission: {submission_str}")  # Debug print

            # Try to parse as JSON first
            try:
                submission = json.loads(submission_str)
            except json.JSONDecodeError:
                # If not valid JSON, try to create a simple answer dict
                submission = {"answer": submission_str}

            # Store result in task group
            print(f"Parsed submission: {submission}")  # Debug print
            score = self.current_task.scoring_fn(submission)
            self.task_group.store_result(self.task_id, submission, score)

            return score
        except Exception as e:
            print(f"Error scoring submission for task {self.task_id}: {str(e)}")
            print(f"Submission was: {self.state.submitted_answer}")
            return 0.0

def create_catalysis_environments() -> Dict[str, Environment]:
    """Create environments for catalysis tasks"""

    def score_addition(result: Dict) -> float:
        score = 0.0
        if "answer" in result:
            try:
                answer = float(result["answer"])
                score = 1.0
            except ValueError:
                pass
        return score

    def check_structure(result)-> float:
        return 0

    data_path = "./generated_tasks/data/"
    directories = os.listdir(data_path)
    for directory in directories:
        file_path = os.path.join(data_path, directory, f"{directory}.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
        subtasks = data['subtasks']
        # Create task group


        tasks = {}
        for subtask in subtasks:
            if subtask['initial_input'] is not None:
                initial_input = {'directory' : subtask['initial_input']}
            else:
                initial_input = None
            tasks[subtask['subtask_id']] = TaskDefinition(
                name = subtask['name'],
                description=subtask['Description'],
                tools = subtask['tools'],
                submission_format={"answer" : subtask['submission_format']},
                initial_input=initial_input,
                scoring_fn=subtask['scoring_fn'],
                input_from_task=subtask['input_from_task']
            )

        task_group = TaskGroup(group_id=data['task_id'], tasks = tasks)

    # # Create task group
    # task_group = TaskGroup(
    #     group_id="catalyst",
    #     tasks={
    #         "task_1": TaskDefinition(
    #             name="Directory",
    #             description="Make a given directory. If it is done successfully print the absolute location of the directory as the final output.",
    #             tools=["run_bash_command"],
    #             scoring_fn=check_structure,
    #             submission_format={"answer": "/path/to/directory"},
    #             initial_input={"directory" : "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/temp"}
    #         ),

    #         "task_2": TaskDefinition(
    #             name="LAMMPS Simulation",
    #             description="Your task is to generate a face centered cubic (FCC) lattice structure for Aluminum using LAMMPS with lattice constant of 4.05 angstrom. The simulation should use metal units, define a 5x5x5 simulation box, set periodic boundary conditions to True, and dump all the coordinates and lattice parameters in Al.xyz file. Save all the related files to the given directory.",
    #             tools=["run_bash_command", "run_lammps"],
    #             scoring_fn=check_structure,
    #             submission_format={"answer": None},
    #             input_from_task="task_1"
    #         ),
    #     }
    # )

    # Print task dependencies for reference
        print("\nTask Dependencies:")
        for task_id, deps in task_group.get_task_dependencies().items():
            print(f"- {task_id}: depends on {deps}")

        # Create environments for all tasks
        available_tools = create_tools()
        environments = {}

        for task_id in task_group.tasks:
            # All environments share the same task group instance
            #task_id = f"{task_group.group_id}_{_id}"
            environments[task_id] = TaskEnvironment(
                task_id=task_id,
                task_group=task_group,
                available_tools=available_tools
            )

        return environments

if __name__ == "__main__":
    # Create all environments
    environments = create_catalysis_environments()

    print("\nCreated Environments:")
    for env_id, env in environments.items():
        print(f"- {env_id}")
        print(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_task:
            print(f"  Depends on: {env.current_task.input_from_task}")

    # Create and run server
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)