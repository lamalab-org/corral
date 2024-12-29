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
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.max_iterations = 10

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
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content

    def parse_llm_response(self, response: str) -> Tuple[Optional[Thought], Optional[Action]]:
        """Parse LLM response into Thought and Action"""
        thought_match = re.search(r"Thought: (.*?)(?=\nAction:|Final Answer:|$)", response, re.DOTALL)
        action_match = re.search(r"Action: (\w+)\nAction Input: ({.*})", response, re.DOTALL)

        thought = Thought(thought_match.group(1).strip()) if thought_match else None

        action = None
        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                arguments = json.loads(action_match.group(2).strip())
                action = Action(tool_name=tool_name, arguments=arguments)
            except json.JSONDecodeError:
                pass

        return thought, action

    def create_prompt(self, task_guide: str, history: List[str]) -> str:
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
Final Answer: [answer]"""

    def solve_task(self, interface: BenchmarkInterface, task_id: str) -> str:
        """Main ReAct loop implementation"""
        task_guide = interface.get_task_guide(task_id)
        history: List[str] = []

        for iteration in range(self.max_iterations):
            # Create prompt and get LLM response
            prompt = self.create_prompt(task_guide, history)
            llm_response = self.get_llm_response(prompt)

            # Parse response
            thought, action = self.parse_llm_response(llm_response)

            # Record thought
            if thought:
                history.append(f"Thought: {thought.content}")

            # Check for final answer
            final_answer_match = re.search(r"Final Answer: (.*)", llm_response)
            if final_answer_match:
                return final_answer_match.group(1).strip()

            # Execute tool if action exists
            if action:
                history.append(f"Action: {action.tool_name}\nAction Input: {json.dumps(action.arguments)}")

                # Execute tool and get response
                tool_response = interface.execute_tool(
                    task_id, 
                    action.tool_name, 
                    action.arguments
                )

                # Record observation
                observation = (
                    f"Observation: {tool_response.result}" 
                    if tool_response.success 
                    else f"Error: {tool_response.error}"
                )
                history.append(observation)

            # If no action or thought was parsed, break the loop
            if not thought and not action:
                break

        return "Unable to solve task within iteration limit"
