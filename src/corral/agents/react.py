from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
from litellm import completion
from corral.evaluate import BenchmarkInterface

@dataclass
class Thought:
    """Represents agent's reasoning step"""
    content: str

@dataclass
class Action:
    """Represents an action to be taken"""
    tool_name: str
    arguments: Dict[str, Any]

class ReActAgent:
    def __init__(self, model: str = "gpt-4", max_iterations: int = 10, temperature = 0.7):
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature

    def get_llm_response(self, prompt: str) -> str:
        """Get response from LLM using LiteLLM"""
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that solves tasks step by step."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

    def parse_llm_response(self, response: str) -> Tuple[Optional[Thought], Optional[Action]]:
        """Parse LLM response into Thought and Action"""
        thought_match = re.search(r"Thought: (.*?)(?=\nAction:|Final Answer:|$)", response, re.DOTALL)
        action_match = re.search(r"Action: (\w+)\nAction Input: ({.*})", response, re.DOTALL)
        # action_match = re.search(r"Action: (\w+)\nAction Input:\s*({.*?})\s*$", response, re.DOTALL)
        # print("action_match", action_match)
        thought = Thought(thought_match.group(1).strip()) if thought_match else None

        action = None
        if action_match:
            tool_name = action_match.group(1).strip()
            # print("action tool name", tool_name)
            try:
                arguments = json.loads(action_match.group(2).strip())
                # print("action arguments", arguments)
                action = Action(tool_name=tool_name, arguments=arguments)
            except json.JSONDecodeError:
                pass

        return thought, action

    def create_prompt(self, task_guide: str, history: list[str]) -> str:
        """Create prompt for LLM including context and history"""
        return f"""Task Guide: {task_guide}

Previous steps:
{chr(10).join(history)}

Think about what to do next and respond in the following format:

Thought: [your reasoning]
Action: [tool name]
Action Input: [tool arguments as JSON]

If you have the final answer, respond with:
Thought: [your reasoning]
Final Answer: [answer]

Make sure to restrict your output to one Thought, Action, and Action Input per response. Since this is for scientific experiments, do not make up answers.
"""

    def solve_task(self, interface: BenchmarkInterface, task_id: str, directory_path: str, local_directory_path: str) -> str:
        """Main ReAct loop implementation"""
        task_guide = interface.get_task_guide(task_id)
        history: list[str] = []

        import os

        # Ensure the local directory exists
        os.makedirs(local_directory_path, exist_ok=True)

        # Define the file path where all LLM responses and actions will be stored
        response_file_path = os.path.join(local_directory_path, f"task_{task_id}_responses.txt")

        for iteration in range(self.max_iterations):
            # Create prompt and get LLM response
            prompt = self.create_prompt(task_guide, history)
            llm_response = self.get_llm_response(prompt)

            # Parse response
            thought, action = self.parse_llm_response(llm_response)

            # Format the action as a string
            action_str = "No action taken"
            if action:
                action_str = f"Tool: {action.tool_name}\nArguments: {json.dumps(action.arguments, indent=2)}"

            # Append the response and action to the text file with formatting
            with open(response_file_path, "a", encoding="utf-8") as file:
                file.write(f"\n--- ITERATION {iteration + 1} ---\n")
                file.write(f"LLM Response:\n{llm_response}\n")
                file.write(f"\nAction Taken:\n{action_str}\n")

            # Record thought
            if thought:
                history.append(f"Thought: {thought.content}")

            # Check for final answer
            final_answer_match = re.search(r"Final Answer: (.*)", llm_response)
            if final_answer_match:
                return final_answer_match.group(1).strip()

            # Execute tool if action exists
            if action:
                action.arguments["directory_path"] = directory_path
                history.append(
                    f"Action: {action.tool_name}\nAction Input: {json.dumps(action.arguments)}"
                )

                tool_response = interface.execute_tool(
                    task_id, action.tool_name, action.arguments
                )

                # Record observation
                observation = (
                    f"Observation: {tool_response.result}"
                    if tool_response.success
                    else f"Error: {tool_response.error}"
                )
                history.append(observation)
                with open(response_file_path, "a", encoding="utf-8") as file:
                    # file.write(observation + "\n")
                    file.write(f"\nTool Response:\n{observation}\n")

            # If no action or thought was parsed, break the loop
            if not thought and not action:
                break

        return "Unable to solve task within iteration limit"

