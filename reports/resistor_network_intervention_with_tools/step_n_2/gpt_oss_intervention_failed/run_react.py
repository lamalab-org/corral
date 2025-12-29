import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import litellm
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from corral import CorralRouter, CorralRunner
from corral.agents import ReActAgent
from corral.agents.hooks import AgentHooks, HookPoint
from corral.agents.hooks.core import HookContext
from corral.agents.hooks.intervention import create_intervention_hook
from corral.report import CorralWandbLogger

successful_partial = """
<thought>\nLet me analyze the resistance measurements to infer the circuit topology. I have measurements between various nodes: A, B, N_b1, N_c1, N_c2, N_c3, N_d1, N_d2, N_d3, X1, X2, X3.\n\nLooking at the pattern of resistances and the note that resistors are labeled R1, R2, R3, etc. from left to right, this suggests a ladder or chain-like structure.\n\nLet me identify some key observations:\n1. A to B resistance is 125.0 ohms (likely the total resistance through the entire network)\n2. There are intermediate nodes with naming patterns: N_b1, N_c1-N_c3, N_d1-N_d3, X1-X3\n3. The naming suggests a structured topology\n\nLet me look for patterns:\n- A to X1: 6.667 ohms\n- X1 to N_b1: 5.0 ohms\n- A to N_b1: 11.667 ohms (which is approximately 6.667 + 5.0)\n\nThis suggests X1 is between A and N_b1.\n\nLet me trace more connections:\n- N_b1 to X2: 13.333 ohms\n- X2 to N_c1: 5.0 ohms\n- N_b1 to N_c1: 18.333 ohms (approximately 13.333 + 5.0)\n\nThis suggests X2 is between N_b1 and N_c1.\n\nContinuing this pattern:\n- N_c1 to N_c2: 10.0 ohms\n- N_c2 to N_c3: 15.0 ohms\n- N_c3 to X3: 20.0 ohms\n- X3 to N_d1: 5.0 ohms\n- N_d1 to N_d2: 10.0 ohms\n- N_d2 to N_d3: 15.0 ohms\n- N_d3 to B: 20.0 ohms\n\nLet me start by creating a hypothesis about the topology and test it. The structure appears to be a ladder network with series and parallel combinations.\n\nLet me write out my analysis to a file first to organize my thoughts.\n</thought>\n\n<action>write_file</action>\n<action_input>{"path": "analysis.txt", "content": "Circuit Topology Analysis\\n=========================\\n\\nKey Observations:\\n1. Total A-B resistance: 125.0 ohms\\n2. Nodes: A, B, N_b1, N_c1, N_c2, N_c3, N_d1, N_d2, N_d3, X1, X2, X3\\n\\nSeries connections identified:\\n- A to X1: 6.667 ohms\\n- X1 to N_b1: 5.0 ohms\\n- A to N_b1: 11.667 ohms (confirms X1 is between A and N_b1)\\n\\n- N_b1 to X2: 13.333 ohms\\n- X2 to N_c1: 5.0 ohms\\n- N_b1 to N_c1: 18.333 ohms (confirms X2 is between N_b1 and N_c1)\\n\\n- N_c1 to N_c2: 10.0 ohms\\n- N_c2 to N_c3: 15.0 ohms\\n- N_c3 to X3: 20.0 ohms\\n- X3 to N_d1: 5.0 ohms\\n- N_d1 to N_d2: 10.0 ohms\\n- N_d2 to N_d3: 15.0 ohms\\n- N_d3 to B: 20.0 ohms\\n\\nLet me check X2 to X3: 50.0 ohms\\nN_c1 to X3: 45.0 ohms\\nX2 to N_d1: 55.0 ohms\\n\\nThis suggests a more complex topology with parallel paths.\\n"}</action_input>


<thought>\nLooking at the measurements more carefully, I notice that some resistances have fractional values like 6.667, 11.667, 13.333, etc. These suggest parallel combinations (e.g., 6.667 \u2248 20/3, 11.667 \u2248 35/3, 13.333 \u2248 40/3).\n\nLet me hypothesize a ladder network structure. The pattern suggests:\n- From A, there's a path through X1, N_b1, X2, N_c1, N_c2, N_c3, X3, N_d1, N_d2, N_d3 to B\n- There might be parallel resistors creating the fractional values\n\nLet me try to build a topology hypothesis. Looking at the fractional resistances:\n- A to X1: 6.667 = 20/3 (could be 10 || 20 = 6.667)\n- X1 to N_b1: 5.0\n- N_b1 to X2: 13.333 = 40/3 (could be 20 || 40 = 13.333)\n- X2 to N_c1: 5.0\n\nLet me propose a ladder network with series resistors along the main path and parallel resistors branching off. I'll start with a simple hypothesis and test it.\n\nBased on the pattern, let me propose:\n- R1 (10 ohms) from A to X1\n- R2 (20 ohms) from A to X1 (parallel to R1)\n- R3 (5 ohms) from X1 to N_b1\n- R4 (20 ohms) from N_b1 to X2\n- R5 (40 ohms) from N_b1 to X2 (parallel to R4)\n- R6 (5 ohms) from X2 to N_c1\n- And so on...\n\nActually, let me reconsider. The measurements suggest a more systematic structure. Let me look at the direct series resistances:\n- N_c1 to N_c2: 10.0\n- N_c2 to N_c3: 15.0\n- N_c3 to X3: 20.0\n- X3 to N_d1: 5.0\n- N_d1 to N_d2: 10.0\n- N_d2 to N_d3: 15.0\n- N_d3 to B: 20.0\n\nThese are clean integer values, suggesting they are direct series connections.\n\nFor the fractional values at the beginning, they suggest parallel combinations. Let me build a topology step by step.\n</thought>\n\n<action>write_file</action>\n<action_input>{\"path\": \"topology_hypothesis.txt\", \"content\": \"Topology Hypothesis v1\\n======================\\n\\nBased on measurements, I propose a ladder network:\\n\\nMain path (series resistors):\\n- R3: X1 to N_b1 (5 ohms)\\n- R6: X2 to N_c1 (5 ohms)\\n- R7: N_c1 to N_c2 (10 ohms)\\n- R8: N_c2 to N_c3 (15 ohms)\\n- R9: N_c3 to X3 (20 ohms)\\n- R10: X3 to N_d1 (5 ohms)\\n- R11: N_d1 to N_d2 (10 ohms)\\n- R12: N_d2 to N_d3 (15 ohms)\\n- R13: N_d3 to B (20 ohms)\\n\\nParallel branches:\\n- R1 and R2 in parallel from A to X1\\n  - A to X1 measured: 6.667 = 20/3\\n  - If R1 = 10 and R2 = 20, then 10||20 = 6.667 \u2713\\n\\n- R4 and R5 in parallel from N_b1 to X2\\n  - N_b1 to X2 measured: 13.333 = 40/3\\n  - If R4 = 20 and R5 = 40, then 20||40 = 13.333 \u2713\\n\\nLet me verify this hypothesis with simulations.\\n\"}</action_input>


<thought>\nLet me verify the parallel resistor calculations first, then build the complete topology JSON and test it.\n\nFor R1 and R2 in parallel from A to X1:\n- If R1 = 10 and R2 = 20, then 1/(1/10 + 1/20) = 1/(0.1 + 0.05) = 1/0.15 = 6.667 \u2713\n\nFor R4 and R5 in parallel from N_b1 to X2:\n- If R4 = 20 and R5 = 40, then 1/(1/20 + 1/40) = 1/(0.05 + 0.025) = 1/0.075 = 13.333 \u2713\n\nNow let me construct the full topology JSON and test it.\n</thought>\n\n<action>calculate_parallel_resistance</action>\n<action_input>{\"resistances\": [10.0, 20.0]}</action_input>


<action>calculate_parallel_resistance</action>\n<action_input>{"resistances": [20.0, 40.0]}</action_input>'


<thought>\nGood! The parallel calculations match. Now let me construct the complete topology JSON. Based on my analysis:\n\nResistors (labeled R1, R2, R3, ... from left to right):\n- R1: 10 ohms (A to X1)\n- R2: 20 ohms (A to X1, parallel to R1)\n- R3: 5 ohms (X1 to N_b1)\n- R4: 20 ohms (N_b1 to X2)\n- R5: 40 ohms (N_b1 to X2, parallel to R4)\n- R6: 5 ohms (X2 to N_c1)\n- R7: 10 ohms (N_c1 to N_c2)\n- R8: 15 ohms (N_c2 to N_c3)\n- R9: 20 ohms (N_c3 to X3)\n- R10: 5 ohms (X3 to N_d1)\n- R11: 10 ohms (N_d1 to N_d2)\n- R12: 15 ohms (N_d2 to N_d3)\n- R13: 20 ohms (N_d3 to B)\n\nLet me create the topology JSON and test it.\n</thought>\n\n<action>write_file</action>\n<action_input>{"path": "topology_v1.json", "content": "{\\"resistors\\": {\\"R1\\": 10, \\"R2\\": 20, \\"R3\\": 5, \\"R4\\": 20, \\"R5\\": 40, \\"R6\\": 5, \\"R7\\": 10, \\"R8\\": 15, \\"R9\\": 20, \\"R10\\": 5, \\"R11\\": 10, \\"R12\\": 15, \\"R13\\": 20}, \\"connections\\": [[\\"A\\", \\"X1\\", \\"R1\\"], [\\"A\\", \\"X1\\", \\"R2\\"], [\\"X1\\", \\"N_b1\\", \\"R3\\"], [\\"N_b1\\", \\"X2\\", \\"R4\\"], [\\"N_b1\\", \\"X2\\", \\"R5\\"], [\\"X2\\", \\"N_c1\\", \\"R6\\"], [\\"N_c1\\", \\"N_c2\\", \\"R7\\"], [\\"N_c2\\", \\"N_c3\\", \\"R8\\"], [\\"N_c3\\", \\"X3\\", \\"R9\\"], [\\"X3\\", \\"N_d1\\", \\"R10\\"], [\\"N_d1\\", \\"N_d2\\", \\"R11\\"], [\\"N_d2\\", \\"N_d3\\", \\"R12\\"], [\\"N_d3\\", \\"B\\", \\"R13\\"]]}"}</action_input>'


<action>read_file</action>\n<action_input>{"path": "topology_v1.json"}</action_input>


<thought>\nNow let me test this topology by simulating the resistance between A and B, which should be 125.0 ohms according to the measurements.\n</thought>\n\n<action>simulate_circuit_resistance</action>\n<action_input>{"topology": "{\\"resistors\\": {\\"R1\\": 10, \\"R2\\": 20, \\"R3\\": 5, \\"R4\\": 20, \\"R5\\": 40, \\"R6\\": 5, \\"R7\\": 10, \\"R8\\": 15, \\"R9\\": 20, \\"R10\\": 5, \\"R11\\": 10, \\"R12\\": 15, \\"R13\\": 20}, \\"connections\\": [[\\"A\\", \\"X1\\", \\"R1\\"], [\\"A\\", \\"X1\\", \\"R2\\"], [\\"X1\\", \\"N_b1\\", \\"R3\\"], [\\"N_b1\\", \\"X2\\", \\"R4\\"], [\\"N_b1\\", \\"X2\\", \\"R5\\"], [\\"X2\\", \\"N_c1\\", \\"R6\\"], [\\"N_c1\\", \\"N_c2\\", \\"R7\\"], [\\"N_c2\\", \\"N_c3\\", \\"R8\\"], [\\"N_c3\\", \\"X3\\", \\"R9\\"], [\\"X3\\", \\"N_d1\\", \\"R10\\"], [\\"N_d1\\", \\"N_d2\\", \\"R11\\"], [\\"N_d2\\", \\"N_d3\\", \\"R12\\"], [\\"N_d3\\", \\"B\\", \\"R13\\"]]}", "terminal_nodes": ["A", "B"]}</action_input>
"""


failed_partial = """
<thought>To infer the circuit topology and resistor values, I need to analyze the given node-to-node resistance measurements. The task involves identifying the connections and estimating the resistor values that match these measurements. I will start by examining the measurements to identify potential series and parallel connections. Then, I will propose a topology and validate it using the available tools.</thought>\n<action>list_files</action>\n<action_input>{\"path\": \".\"}</action_input>


<thought>There are no files in the current directory, which means I need to work directly with the provided measurements and tools. I will start by analyzing the measurements to identify any obvious series or parallel connections. This will help in proposing an initial circuit topology.</thought>\n<action>calculate_parallel_resistance</action>\n<action_input>{\"resistances\": [11.667, 6.667]} </action_input>


<thought>The parallel resistance calculation of 11.667 ohms and 6.667 ohms results in approximately 4.24 ohms. This suggests that these two resistors could be in parallel in the circuit. I will continue analyzing other measurements to identify more connections and propose a complete topology.</thought>\n<action>calculate_series_resistance</action>\n<action_input>{\"resistances\": [30.0, 40.0]} </action_input>


<thought>The series resistance calculation of 30.0 ohms and 40.0 ohms results in 70.0 ohms. This suggests that these two resistors could be in series in the circuit. I will continue to analyze the measurements to identify more connections and propose a complete topology.</thought>\n<action>calculate_parallel_resistance</action>\n<action_input>{\"resistances\": [25.0, 75.0]} </action_input>


<thought>The parallel resistance calculation of 25.0 ohms and 75.0 ohms results in 18.75 ohms. This suggests that these two resistors could be in parallel in the circuit. I will now attempt to propose a complete circuit topology based on these observations and validate it using the available tools.</thought>\n<action>simulate_circuit_resistance</action>\n<action_input>{\"topology\": \"{\\\"resistors\\\": {\\\"R1\\\": 11.667, \\\"R2\\\": 6.667, \\\"R3\\\": 30.0, \\\"R4\\\": 40.0, \\\"R5\\\": 25.0, \\\"R6\\\": 75.0}, \\\"connections\\\": [[\\\"A\\\", \\\"N_b1\\\", \\\"R1\\\"], [\\\"A\\\", \\\"X1\\\", \\\"R2\\\"], [\\\"N_c1\\\", \\\"N_c2\\\", \\\"R3\\\"], [\\\"N_c2\\\", \\\"N_c3\\\", \\\"R4\\\"], [\\\"X2\\\", \\\"X3\\\", \\\"R5\\\"], [\\\"X1\\\", \\\"X3\\\", \\\"R6\\\"]]}\", \"terminal_nodes\": [\"A\", \"B\"]}</action_input>


<thought>The error indicates that the node 'B' is not connected in the proposed topology. I need to revise the connections to ensure that all nodes, especially the terminal nodes 'A' and 'B', are properly connected. I will re-evaluate the measurements and adjust the topology accordingly.</thought>\n<action>calculate_parallel_resistance</action>\n<action_input>{\"resistances\": [113.333, 118.333]} </action_input>


<thought>The parallel resistance calculation of 113.333 ohms and 118.333 ohms results in approximately 57.89 ohms. This suggests that these two resistors could be in parallel in the circuit. I will now attempt to propose a revised circuit topology that includes node 'B' and validate it using the available tools.</thought>\n<action>simulate_circuit_resistance</action>\n<action_input>{\"topology\": \"{\\\"resistors\\\": {\\\"R1\\\": 11.667, \\\"R2\\\": 6.667, \\\"R3\\\": 30.0, \\\"R4\\\": 40.0, \\\"R5\\\": 25.0, \\\"R6\\\": 75.0, \\\"R7\\\": 113.333, \\\"R8\\\": 118.333}, \\\"connections\\\": [[\\\"A\\\", \\\"N_b1\\\", \\\"R1\\\"], [\\\"A\\\", \\\"X1\\\", \\\"R2\\\"], [\\\"N_c1\\\", \\\"N_c2\\\", \\\"R3\\\"], [\\\"N_c2\\\", \\\"N_c3\\\", \\\"R4\\\"], [\\\"X2\\\", \\\"X3\\\", \\\"R5\\\"], [\\\"X1\\\", \\\"X3\\\", \\\"R6\\\"], [\\\"B\\\", \\\"N_b1\\\", \\\"R7\\\"], [\\\"B\\\", \\\"X1\\\", \\\"R8\\\"]]}\", \"terminal_nodes\": [\"A\", \"B\"]}</action_input>


<thought>The error indicates that the circuit is not solvable, possibly due to disconnections or an ill-conditioned setup. I need to ensure that all nodes are properly connected and that the topology is consistent with the measurements. I will re-evaluate the connections and propose a new topology.</thought>\n<action>calculate_series_resistance</action>\n<action_input>{\"resistances\": [45.0, 35.0, 20.0]} </action_input>


<thought>The series resistance calculation of 45.0 ohms, 35.0 ohms, and 20.0 ohms results in 100.0 ohms. This suggests that these three resistors could be in series in the circuit. I will now attempt to propose a revised circuit topology that includes these series connections and validate it using the available tools.</thought>\n<action>simulate_circuit_resistance</action>\n<action_input>{\"topology\": \"{\\\"resistors\\\": {\\\"R1\\\": 11.667, \\\"R2\\\": 6.667, \\\"R3\\\": 30.0, \\\"R4\\\": 40.0, \\\"R5\\\": 25.0, \\\"R6\\\": 75.0, \\\"R7\\\": 45.0, \\\"R8\\\": 35.0, \\\"R9\\\": 20.0}, \\\"connections\\\": [[\\\"A\\\", \\\"N_b1\\\", \\\"R1\\\"], [\\\"A\\\", \\\"X1\\\", \\\"R2\\\"], [\\\"N_c1\\\", \\\"N_c2\\\", \\\"R3\\\"], [\\\"N_c2\\\", \\\"N_c3\\\", \\\"R4\\\"], [\\\"X2\\\", \\\"X3\\\", \\\"R5\\\"], [\\\"X1\\\", \\\"X3\\\", \\\"R6\\\"], [\\\"B\\\", \\\"N_d1\\\", \\\"R7\\\"], [\\\"N_d1\\\", \\\"N_d2\\\", \\\"R8\\\"], [\\\"N_d2\\\", \\\"N_d3\\\", \\\"R9\\\"]]}\", \"terminal_nodes\": [\"A\", \"B\"]}</action_input>


<thought>The circuit is still not solvable, indicating that there might be a fundamental issue with the proposed topology. I need to ensure that all nodes are properly connected and that the topology is consistent with the measurements. I will re-evaluate the connections and propose a new topology.</thought>\n<action>calculate_parallel_resistance</action>\n<action_input>{\"resistances\": [80.0, 90.0, 105.0]} </action_input>


<thought>The parallel resistance calculation of 80.0 ohms, 90.0 ohms, and 105.0 ohms results in approximately 30.18 ohms. This suggests that these three resistors could be in parallel in the circuit. I will now attempt to propose a revised circuit topology that includes these parallel connections and validate it using the available tools.</thought>\n<action>simulate_circuit_resistance</action>\n<action_input>{\"topology\": \"{\\\"resistors\\\": {\\\"R1\\\": 11.667, \\\"R2\\\": 6.667, \\\"R3\\\": 30.0, \\\"R4\\\": 40.0, \\\"R5\\\": 25.0, \\\"R6\\\": 75.0, \\\"R7\\\": 80.0, \\\"R8\\\": 90.0, \\\"R9\\\": 105.0}, \\\"connections\\\": [[\\\"A\\\", \\\"N_b1\\\", \\\"R1\\\"], [\\\"A\\\", \\\"X1\\\", \\\"R2\\\"], [\\\"N_c1\\\", \\\"N_c2\\\", \\\"R3\\\"], [\\\"N_c2\\\", \\\"N_c3\\\", \\\"R4\\\"], [\\\"X2\\\", \\\"X3\\\", \\\"R5\\\"], [\\\"X1\\\", \\\"X3\\\", \\\"R6\\\"], [\\\"B\\\", \\\"N_d1\\\", \\\"R7\\\"], [\\\"N_d1\\\", \\\"N_d2\\\", \\\"R8\\\"], [\\\"N_d2\\\", \\\"N_d3\\\", \\\"R9\\\"]]}\", \"terminal_nodes\": [\"A\", \"B\"]}</action_input>


<thought>The circuit is still not solvable, indicating that there might be a fundamental issue with the proposed topology. I need to ensure that all nodes are properly connected and that the topology is consistent with the measurements. I will re-evaluate the connections and propose a new topology.</thought>\n<action>calculate_series_resistance</action>\n<action_input>{\"resistances\": [18.333, 28.333, 43.333]} </action_input>


<thought>The series resistance calculation of 18.333 ohms, 28.333 ohms, and 43.333 ohms results in approximately 90.0 ohms. This suggests that these three resistors could be in series in the circuit. I will now attempt to propose a revised circuit topology that includes these series connections and validate it using the available tools.</thought>\n<action>simulate_circuit_resistance</action>\n<action_input>{\"topology\": \"{\\\"resistors\\\": {\\\"R1\\\": 11.667, \\\"R2\\\": 6.667, \\\"R3\\\": 30.0, \\\"R4\\\": 40.0, \\\"R5\\\": 25.0, \\\"R6\\\": 75.0, \\\"R7\\\": 18.333, \\\"R8\\\": 28.333, \\\"R9\\\": 43.333}, \\\"connections\\\": [[\\\"A\\\", \\\"N_b1\\\", \\\"R1\\\"], [\\\"A\\\", \\\"X1\\\", \\\"R2\\\"], [\\\"N_c1\\\", \\\"N_c2\\\", \\\"R3\\\"], [\\\"N_c2\\\", \\\"N_c3\\\", \\\"R4\\\"], [\\\"X2\\\", \\\"X3\\\", \\\"R5\\\"], [\\\"X1\\\", \\\"X3\\\", \\\"R6\\\"], [\\\"B\\\", \\\"N_d1\\\", \\\"R7\\\"], [\\\"N_d1\\\", \\\"N_d2\\\", \\\"R8\\\"], [\\\"N_d2\\\", \\\"N_d3\\\", \\\"R9\\\"]]}\", \"terminal_nodes\": [\"A\", \"B\"]}</action_input>


<thought>The circuit is still not solvable, indicating that there might be a fundamental issue with the proposed topology. I need to ensure that all nodes are properly connected and that the topology is consistent with the measurements. I will re-evaluate the connections and propose a new topology.</thought>\n<action>calculate_parallel_resistance</action>\n<action_input>{\"resistances\": [50.0, 100.0]} </action_input>


<thought>The parallel resistance calculation of 50.0 ohms and 100.0 ohms results in approximately 33.33 ohms. This suggests that these two resistors could be in parallel in the circuit. I will now attempt to propose a revised circuit topology that includes these parallel connections and validate it using the available tools.</thought>\n<action>simulate_circuit_resistance</action>\n<action_input>{\"topology\": \"{\\\"resistors\\\": {\\\"R1\\\": 11.667, \\\"R2\\\": 6.667, \\\"R3\\\": 30.0, \\\"R4\\\": 40.0, \\\"R5\\\": 25.0, \\\"R6\\\": 75.0, \\\"R7\\\": 50.0, \\\"R8\\\": 100.0}, \\\"connections\\\": [[\\\"A\\\", \\\"N_b1\\\", \\\"R1\\\"], [\\\"A\\\", \\\"X1\\\", \\\"R2\\\"], [\\\"N_c1\\\", \\\"N_c2\\\", \\\"R3\\\"], [\\\"N_c2\\\", \\\"N_c3\\\", \\\"R4\\\"], [\\\"X2\\\", \\\"X3\\\", \\\"R5\\\"], [\\\"X1\\\", \\\"X3\\\", \\\"R6\\\"], [\\\"B\\\", \\\"N_d1\\\", \\\"R7\\\"], [\\\"N_d1\\\", \\\"N_d2\\\", \\\"R8\\\"]]}\", \"terminal_nodes\": [\"A\", \"B\"]}</action_input>


<thought>The circuit is still not solvable, indicating that there might be a fundamental issue with the proposed topology. I need to ensure that all nodes are properly connected and that the topology is consistent with the measurements. I will re-evaluate the connections and propose a new topology.</thought>\n<action>calculate_series_resistance</action>\n<action_input>{\"resistances\": [10.0, 15.0, 25.0]} </action_input>


<thought>The series resistance calculation of 10.0 ohms, 15.0 ohms, and 25.0 ohms results in 50.0 ohms. This suggests that these three resistors could be in series in the circuit. I will now attempt to propose a revised circuit topology that includes these series connections and validate it using the available tools.</thought>\n<action>simulate_circuit_resistance</action>\n<action_input>{\"topology\": \"{\\\"resistors\\\": {\\\"R1\\\": 11.667, \\\"R2\\\": 6.667, \\\"R3\\\": 30.0, \\\"R4\\\": 40.0, \\\"R5\\\": 25.0, \\\"R6\\\": 75.0, \\\"R7\\\": 10.0, \\\"R8\\\": 15.0, \\\"R9\\\": 25.0}, \\\"connections\\\": [[\\\"A\\\", \\\"N_b1\\\", \\\"R1\\\"], [\\\"A\\\", \\\"X1\\\", \\\"R2\\\"], [\\\"N_c1\\\", \\\"N_c2\\\", \\\"R3\\\"], [\\\"N_c2\\\", \\\"N_c3\\\", \\\"R4\\\"], [\\\"X2\\\", \\\"X3\\\", \\\"R5\\\"], [\\\"X1\\\", \\\"X3\\\", \\\"R6\\\"], [\\\"B\\\", \\\"N_d1\\\", \\\"R7\\\"], [\\\"N_d1\\\", \\\"N_d2\\\", \\\"R8\\\"], [\\\"N_d2\\\", \\\"N_d3\\\", \\\"R9\\\"]]}\", \"terminal_nodes\": [\"A\", \"B\"]}</action_input>


<thought>The circuit is still not solvable, indicating that there might be a fundamental issue with the proposed topology. I need to ensure that all nodes are properly connected and that the topology is consistent with the measurements. I will re-evaluate the connections and propose a new topology.</thought>\n<action>calculate_parallel_resistance</action>\n<action_input>{\"resistances\": [35.0, 65.0]} </action_input>
"""

successful_map = {
    "task_4": successful_partial,
}
failed_map = {"task_4": failed_partial}


successful_hook = create_intervention_hook(successful_map, execute_tools=False)
failed_hook = create_intervention_hook(failed_map, execute_tools=False)


def logprobs_hook(context: HookContext) -> None:
    """Extract, convert to dict, and save logprobs."""
    llm_response = context.iteration_data.get("llm_response")
    if not llm_response:
        return

    log_probs_obj = getattr(llm_response, "logprobs", None)
    if not log_probs_obj:
        return

    # Convert Pydantic object to a standard Python Dict once
    try:
        if hasattr(log_probs_obj, "model_dump"):
            lp_dict = log_probs_obj.model_dump()
        elif hasattr(log_probs_obj, "dict"):
            lp_dict = log_probs_obj.dict()
        else:
            lp_dict = log_probs_obj

        # Store in metadata so other hooks can use the clean DICT
        context.metadata["current_logprobs_dict"] = lp_dict

    except Exception as e:
        logger.error(f"Failed to convert logprobs to dict: {e}")
        return

    # Save to disk as an artifact (optional, but good for records)
    lp_data = {
        "id": llm_response.id,
        "task_id": context.task_id,
        "iteration": context.iteration,
        "logprobs": lp_dict,
    }

    out_dir = Path("logprobs")
    out_dir.mkdir(exist_ok=True, parents=True)
    log_prob_hook_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    context.metadata["logprob_hook_timestamp"] = log_prob_hook_timestamp
    file_path = out_dir / f"{llm_response.id}_{log_prob_hook_timestamp}.json"

    with file_path.open("w") as f:
        json.dump(lp_data, f, indent=2)
    logger.info(f"Saved logprobs to {file_path}")


def compute_logprob_metrics(
    logprobs: dict, log_prob_hook_timestamp
) -> dict[str, float]:
    if not logprobs or "content" not in logprobs:
        return {"avg_logprob": 0, "entropy": 0, "variance": 0, "avg_top_entropy": 0}

    token_logprobs = [tok["logprob"] for tok in logprobs["content"] if "logprob" in tok]
    if not token_logprobs:
        return {"avg_logprob": 0, "entropy": 0, "variance": 0, "avg_top_entropy": 0}

    avg_logprob = np.mean(token_logprobs)
    probs = np.exp(token_logprobs)
    entropy = -np.sum(probs * np.log(probs + 1e-10)) / len(probs)
    variance = np.var(token_logprobs)

    # Optional: Avg entropy over top_logprobs per token (for alternative sharpness)
    top_entropies = []
    for tok in logprobs["content"]:
        if "top_logprobs" in tok:
            top_lps = [tl["logprob"] for tl in tok["top_logprobs"]]
            if top_lps:
                top_probs = np.exp(top_lps) / np.sum(np.exp(top_lps))  # Normalize
                top_ent = -np.sum(top_probs * np.log(top_probs + 1e-10))
                top_entropies.append(top_ent)
    avg_top_entropy = np.mean(top_entropies) if top_entropies else 0

    return {
        "avg_logprob": avg_logprob,
        "entropy": entropy,
        "variance": variance,
        "avg_top_entropy": avg_top_entropy,
        "path_length": len(token_logprobs),
        "log_prob_hook_timestamp": log_prob_hook_timestamp,
    }


def save_metrics_to_file(
    task_id: str,
    iteration: int,
    metrics: dict[str, float],
    log_prob_hook_timestamp=None,
) -> None:
    """
    Save computed metrics to a JSON file.

    Args:
        task_id: The current task identifier
        iteration: The iteration number
        metrics: Dictionary of metrics (e.g., {'avg_logprob': -0.5, 'entropy': 1.2, ...})
    """
    out_dir = Path("metrics")
    out_dir.mkdir(exist_ok=True, parents=True)

    file_path = (
        out_dir / f"{task_id}_iteration_{iteration}_{log_prob_hook_timestamp}.json"
    )
    try:
        with file_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")


def metrics_hook(context: HookContext) -> None:
    """Compute and save metrics using the dict already in memory."""
    # 1. Try to get the pre-converted dict from the previous hook
    lp_dict = context.metadata.get("current_logprobs_dict")

    # 2. Fallback: If logprobs_hook didn't run, try to get it from the response object
    if not lp_dict:
        llm_response = context.iteration_data.get("llm_response")
        if not llm_response or not getattr(llm_response, "logprobs", None):
            return

        # (Conversion logic if needed)
        obj = llm_response.logprobs
        lp_dict = obj.model_dump() if hasattr(obj, "model_dump") else obj

    # 3. Compute metrics using the in-memory dictionary
    log_prob_hook_timestamp = context.metadata["logprob_hook_timestamp"]
    metrics = compute_logprob_metrics(lp_dict, log_prob_hook_timestamp)

    # Store metrics in context for logging or final report
    context.metadata[f"iteration_{context.iteration}_metrics"] = metrics

    # Save metrics to file
    save_metrics_to_file(
        context.task_id, context.iteration, metrics, log_prob_hook_timestamp
    )


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def move_work_dir_files(work_dir: str, target_subdir: str):
    """
    Moves all contents of work_dir into a new subdirectory within work_dir,
    and creates the target subdirectory if it doesn't exist.
    """
    if not work_dir:
        logger.warning("CORRAL_WORK_DIR is not set. Skipping file movement.")
        return

    target_path = Path(work_dir) / target_subdir
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Moving work directory contents to: {target_path}")

    # Move all files and folders *except* the target subdirectory itself
    for item_path in Path(work_dir).iterdir():
        if item_path.name != target_subdir:
            try:
                shutil.move(item_path, target_path)
            except Exception as e:
                logger.error(f"Failed to move {item_path}: {e!s}")

    logger.success("Work directory files moved successfully.")


def run_benchmark(
    model: str = "claude-3-5-sonnet-20241022",
    _task_ids: list | None = None,
    temperature: float = 0.7,
    run_name: str = "corral_benchmark_run",
    verbose: str = "brief",
    hooks=None,
):
    """Run the benchmark with specified model and tasks"""

    interface = CorralRouter(base_url="http://localhost:8000")
    wandblogger = CorralWandbLogger(
        project="corral_resistor_oss",
        group="gpt_oss",
        name=run_name,
    )
    agent = ReActAgent(
        model=model,
        logprobs=True,
        top_logprobs=2,
        max_iterations=20,
        temperature=temperature,
        api_endpoint="https://api.helmholtz-blablador.fz-juelich.de/v1",
    )
    runner = CorralRunner(
        interface,
        agent,
        logger=wandblogger,
    )

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        task_ids=[
            "task_4",
        ],
        trials_per_task=50,
        k_values=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
        ],
        verbose=True,
        tool_verbosity=verbose,
        hooks=hooks,
    )
    result.generate_report(f"{run_name}_try.json")
    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()
    os.environ["OPENAI_API_KEY"] = os.getenv("BLABLADOR_API_KEY_TEST", "")

    corral_work_dir = os.getenv("CORRAL_WORK_DIR")
    if not corral_work_dir:
        logger.error("CORRAL_WORK_DIR environment variable is not set!")

    verbose = "workflow"
    conditions = ["failed"]
    for condition in conditions:
        hooks = AgentHooks()  # Fresh hooks per condition
        if condition == "successful":
            hooks.register(HookPoint.BEFORE_TASK, successful_hook)
        elif condition == "failed":
            hooks.register(HookPoint.BEFORE_TASK, failed_hook)
        hooks.register(HookPoint.AFTER_ITERATION, logprobs_hook)
        hooks.register(HookPoint.AFTER_ITERATION, metrics_hook)  # Always on

        run_name = f"gpt_oss_120-react-resistor_{condition}-{verbose}_verbosity-single"
        model = (
            "openai/1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
        )
        run_benchmark(model=model, run_name=run_name, verbose=verbose, hooks=hooks)
        move_work_dir_files(corral_work_dir, verbose)
