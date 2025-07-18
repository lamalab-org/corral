RUBRICS_v1 = {
    "task_rubrics": {
        "correctness": {
            "trajectory_correctness": {
                "question": "Do the agent makes the right trajectory?",
                "description": "If the agent makes the right trajectory without committing silly mistakes (for example this could be agent making wrong submission path), then this is correct, or the box is checked. Otherwise, it is incorrect. This checks the correctness of the trajectory, not the final answer.",
                "examples": [
                    "Answer returned by the agent: 'Final Answer: path/to/final_answer.cif'\nCorrectness: Correct",
                ],
                "automatic_check": False,
                "step_wise": False,
            },
            "final_answer": {
                "question": "Do the agent logs lead to the correct answer?",
                "description": "If the answer returned by the agent is correct, then this is correct, or the box is checked. Otherwise, it is incorrect. This only checks the final answer, not the intermediate steps.",
                "examples": [
                    "Answer returned by the agent: 'Final Answer: [B-](CCC1=CC=CC=C1)(F)(F)F.[K+]'\nCorrectness: Correct",
                    "Answer returned by the agent: 'Final Answer: CCO=F'\nCorrectness: Incorrect",
                ],
                "automatic_check": True,
                "step_wise": False,
            },
        },
        "insanity": {
            "repeated_message": {
                "question": "Did the agent avoid repeating the same exact message two or more iterations?",
                "description": "If the agent returned the same exact message two or more consecutive iterations. Note the negative character of the question. This is the checkbox that would stick unchecked if insanity is observed. Otherwise, we check the box if the agent keeps a normal and rational behavior.",
                "examples": [
                    "`iteration 3`: 'Now I need to provide the final answer with the correct format.' `iteration 4`: 'Now I need to provide the final answer with the correct format.' `iteration 5`: 'Now I need to provide the final answer with the correct format.': incorrect since the agent repeated the same message three times."
                ],
                "automatic_check": True,
                "step_wise": False,
            },
            "repeated_tokens": {
                "question": "Did the agent avoid repeating the same tokens until the max output tokens limit?",
                "description": "If for one message, the agent reproduced the same repeated tokens until the max output tokens limit. Note the negative character of the question. This is the checkbox that would stick unchecked if insanity is observed. Otherwise, we check the box if the agent keeps a normal and rational behavior.",
                "examples": [
                    "'The spectra suggest that the molecule could be the one with SMILES: COCCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC...': incorrect since the agent repeated the same tokens until the max output tokens limit."
                ],
                "automatic_check": False,
                "step_wise": False,
            },
        },
        "task_understanding": {
            "task_decomposition": {
                "question": "Is the task decomposed correctly?",
                "description": "If the agent decomposed the task correctly before proposing the respective steps, then this is correct, or the box is checked. Otherwise, it is incorrect.",
                "examples": [
                    '\'Based on the input data, let me try a structure: ethyl 4-((5-chloropyridin-2-yl)amino)benzoate\n\nLet me verify this with the formula tool first.\nAction: get_formula_from_smiles\nAction Input: {"smiles": "O=C(OCC)c1ccc(Nc2ccc(Cl)cn2)cc1"\': incorrect since the agent did not decompose the task correctly. The task decomposition is to find the functional groups in the spectra, not to verify a structure.'
                ],
                "automatic_check": False,
                "step_wise": False,
            },
            "input_understanding": {
                "question": "Is there a mistake that could have been avoided by using knowledge provided in the context?",
                "description": "If the agent made a mistake that could have been avoided by using knowledge provided in the context, then this is incorrect, or the box is checked. Otherwise, it is correct.",
                "examples": [
                    "For example, in spectra, if some of the spectra are not used, resulting in the agent not finding the correct functional groups, then this is incorrect. If the agent uses all the spectra and finds the correct functional groups, then this is correct."
                ],
                "automatic_check": False,
                "step_wise": False,
            },
        },
        "tool_usage": {
            "tool_sampling": {
                "question": "Did the agent incur into sampling a tool with different inputs?",
                "description": "If the agent used the tool with different inputs (being the inputs always correct) two or more consecutive times, then this is correct, or the box is checked. Otherwise, it is incorrect.",
                "examples": [
                    "`iteration 2`: {'tool_name': 'simulate_spectra', 'arguments': {'smiles': 'CC1=C(C(=O)O)C=CC=C1C(=O)C'}\n`iteration 3`: {'tool_name': 'simulate_spectra', 'arguments': {'smiles': 'CC1=CC=CC=C1C(=O)OC(=O)C'}}\n`iteration 4`: {'tool_name': 'simulate_spectra', 'arguments': {'smiles': 'CC1=CC=C(C(=O)C)C(C(=O)O)=C1'}}: incorrect",
                    "`iteration 2`: {'tool_name': 'get_formula_from_smiles', 'arguments': {'smiles': 'C#CCC#CCCCCC'}\n`iteration 3`: {'tool_name': 'get_formula_from_smiles', 'arguments': {'smiles': 'C#CC#CCCCCCC'}}: incorrect",
                ],
                "automatic_check": True,
                "step_wise": False,
            },
            "tool_usage_reasoning": {
                "question": "Did the agent provide a reasoning for using a specific tool?",
                "description": "If the agent used the tool explaining the reasoning behind why that tool is used, then this is correct, or the box is checked. Otherwise, it is incorrect. Note that for this we do not care about if the reasoning is correct or not, just that the agent provided a reasoning.",
                "examples": [
                    '\'Action: get_formula_from_smiles\nAction Input: {"smiles": "O=C(OCC)c1ccc(Nc2ccc(Cl)cn2)cc1"}\': incorrect since the agent did not explain why it is using the tool get_formula_from_smiles.',
                    '\'Let me verify the structure further by simulating the spectra to compare with the given data.\nAction: simulate_spectra\nAction Input: {"smiles": "O=C(OCC)c1ccc(Nc2ccc(Cl)cn2)cc1"}\': correct since the agent explained why it is using the tool simulate_spectra.',
                ],
                "automatic_check": False,
                "step_wise": True,
            },
            "rational": {
                "logical_tool_usage": {
                    "question": "Is the step meaningful at this point in the trajectory and conductive toward the end goal? Hint: A step would be non-meaningful if the information obtained in there can be used in no way for the final solution.",
                    "description": "If the tool used by the agent is logical for that step, taking into account the end goal and the information known at that step, then this is correct or the box is checked. Otherwise, it is incorrect.",
                    "examples": [
                        '\'The final answer is the SMILES CC1=C(C(=O)O)C=CC=C1C(=O)C. I will know retrieve the carbon shifts and then I will provide the final answer.\nAction: get_carbon_shifts\nAction Input: {"smiles": "CC1=C(C(=O)O)C=CC=C1C(=O)C"}\': incorrect since the agent is retrieving some prior knowledge that is not needed to provide the final answer.',
                        '\'I will start by retrieving the carbon shifts of the molecule to better understand the spectra provided.\nAction: get_carbon_shifts\nAction Input: {"smiles": "CC1=C(C(=O)O)C=CC=C1C(=O)C"}\': correct since the agent is retrieving some prior knowledge that is needed to fully understand the input.',
                    ],
                    "automatic_check": False,
                    "step_wise": True,
                },
                "efficient_execution": {
                    "question": "Is the action performed in the most efficient way? For example, was an optimized tool used when it was available?",
                    "description": "If the agent executed the action in the most efficient way, then this is correct, or the box is checked. Otherwise, it is incorrect.",
                    "examples": [
                        '\'I need to join the different datasets: Action: read_file\nAction Input: {"file_path": "dataset1.csv"}... Action: write_file\nAction Input: {"file_path": "joined_dataset.csv", "data": "..."}\': incorrect since the agent is reading and writing files, which is not the most efficient way to join datasets.',
                        '\'I will write a python script to join the datasets: Action: write_file\nAction Input: {"file_path": "join_datasets.py", "data": "import pandas as pd...\': correct since the agent is using a python script to join the datasets, which is the most efficient way to do it.',
                    ],
                    "automatic_check": False,
                    "step_wise": True,
                },
            },
            "tool_calling_error": {
                "question": "Did the agent call the tool correctly without incurring any errors?",
                "description": "If the agent called the tool correctly, i.e., without incurring in argument format errors or an incorrect tool name, then this is correct or the box is checked. Otherwise, it is incorrect. Note that this only checks if the tool was called correctly, not if the tool returned the expected output.",
                "examples": [
                    '\'Action: get_formula_from_smiles\nAction Input: {"smiles": "[C#CCC#CCCCCC]"}\': incorrect since the agent passed an incorrect argument format to the tool get_formula_from_smiles list instead of string.',
                    '\'Action: wrong_tool_name\nAction Input: {"smiles": "O=C(OCC)c1ccc(Nc2ccc(Cl)cn2)cc1"}\': incorrect since the agent used a wrong tool name.',
                    '\'Action: get_formula_from_smiles\nAction Input: {"smiles": "O=C(OCC)c1ccc(Nc2ccc(Cl)cn2)cc1"}\': correct since the agent passed the correct argument format to the tool get_formula_from_smiles string.',
                ],
                "automatic_check": False,
                "step_wise": True,
            },
        },
    }
}
