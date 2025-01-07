import json

import anthropic
from dotenv import load_dotenv
from prompts import BASELINESYSTEMPROMPT, BASELINEUSERPROMPT

from corral.evaluate import BenchmarkInterface, MatAgentBenchmark

load_dotenv("../.env", override=True)


class ClaudeAgent:
    """Claude-based agent implementation"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def solve_task(self, interface: BenchmarkInterface, task_id: str) -> str:
        guide = interface.get_task_guide(task_id)

        system_prompt = BASELINESYSTEMPROMPT.format(guide=guide)

        # Initialize messages with a starter message
        messages = [{"role": "user", "content": BASELINEUSERPROMPT}]

        while True:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )

            message = response.content[0].text
            messages.append({"role": "assistant", "content": message})

            if "FINAL ANSWER:" in message:
                print(f"Final answer: {message.split('FINAL ANSWER:')[1].strip()}")
                return message.split("FINAL ANSWER:")[1].strip()

            if "TOOL CALL:" in message:
                try:
                    print(f"Tool call: {message.split('TOOL CALL:')[1].strip()}")
                    tool_json = message.split("TOOL CALL:")[1].strip()
                    tool_request = json.loads(tool_json)
                    result = interface.execute_tool(
                        task_id, tool_request["tool_name"], tool_request["arguments"]
                    )
                    if result.success:
                        messages.append(
                            {"role": "user", "content": f"Tool result: {result.result}"}
                        )
                    else:
                        messages.append(
                            {"role": "user", "content": f"Error: {result.error}"}
                        )
                except Exception as e:
                    messages.append({"role": "user", "content": f"Error: {e!s}"})


if __name__ == "__main__":
    import os

    # Create components
    interface = BenchmarkInterface()
    agent = ClaudeAgent(api_key=os.getenv("ANTHROPIC_API_KEY"))
    runner = MatAgentBenchmark(interface, agent)

    # Run benchmark
    result = runner.bench()
    print("Benchmark completed:")
    print(f"Average score: {result.average_score}")
    print(f"Tasks completed: {result.successful_tasks}/{result.total_tasks}")

    # Print detailed results
    for task_id, task_result in result.task_results.items():
        print(f"\nTask {task_id}:")
        print(f"Score: {task_result.score}")
        print(f"Tool usage: {task_result.tool_statistics}")
