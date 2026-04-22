from corral.backend.schema import ToolArgument
from corral.backend.tool import (
    Tool,
    arguments_to_schema,
    tool,
)


@tool
def calculator(operation: str, x: float, y: float) -> float:
    """[BRIEF] Perform basic arithmetic operations (add, subtract, multiply, divide) on two numbers. [/BRIEF]

    [DETAILED] This tool takes two numerical inputs and applies one of four basic arithmetic operations: addition, subtraction, multiplication, or division. It validates the requested operation and raises an error for unsupported operations. Division by zero returns an error string rather than raising an exception. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to perform a basic arithmetic calculation on two numbers.
    - When the task requires step-by-step numerical computation.
    - Recommended as the primary tool for any addition, subtraction, multiplication, or division. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Identify the two numerical values and the operation needed from the task description. [/PREREQUISITE]
        2. [CURRENT] Call this tool with the operation and the two numbers. [/CURRENT]
        3. [FOLLOW_UP] Use the result for further calculations (e.g., feed into `percentage_calculator`) or report it as the final answer. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Maps the operation string to the corresponding arithmetic function.
    - Executes the operation on the two input numbers.
    - For division, checks for zero divisor and returns an error string if y is 0.
    - Raises ValueError if the operation is not one of the supported choices. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `calculator("add", 3.0, 5.0)`,
        `calculator("subtract", 10.0, 4.0)`,
        `calculator("multiply", 2.5, 3.0)`,
        `calculator("divide", 10.0, 2.0)`,
    ]
    [/SYNTACTICAL]

    Args:
        operation (str):
            [ARGS_BRIEF] The arithmetic operation to perform. [/ARGS_BRIEF]
            [ARGS_DETAILED] A string specifying which arithmetic operation to apply to the two input numbers. Must be one of the supported operations. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] One of: "add", "subtract", "multiply", "divide" [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "add", "subtract", "multiply", "divide" [/ARGS_EXAMPLES]
            (choices: ["add", "subtract", "multiply", "divide"])
        x (float):
            [ARGS_BRIEF] The first operand. [/ARGS_BRIEF]
            [ARGS_DETAILED] The first numerical value in the arithmetic operation. Acts as the left-hand operand. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Any valid floating-point number. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] 3.0, 10.0, -5.5, 0.0 [/ARGS_EXAMPLES]
        y (float):
            [ARGS_BRIEF] The second operand. [/ARGS_BRIEF]
            [ARGS_DETAILED] The second numerical value in the arithmetic operation. Acts as the right-hand operand. For division, must be non-zero. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Any valid floating-point number. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] 5.0, 2.0, -1.0, 100.0 [/ARGS_EXAMPLES]

    Returns:
        float:
            [RETURNS_BRIEF] The result of the arithmetic operation. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The numerical result of applying the specified operation to x and y. For division by zero, returns the string "Error: Division by zero" instead of a number. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] 8.0 for add(3,5), 6.0 for subtract(10,4), 7.5 for multiply(2.5,3), 5.0 for divide(10,2) [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the operation is not one of "add", "subtract", "multiply", or "divide". [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the provided operation string does not match any of the supported arithmetic operations. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the operation parameter is exactly one of: "add", "subtract", "multiply", "divide". [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
    - Division by zero returns an error string rather than raising an exception, which may cause type inconsistency.
    - Only supports four basic operations; does not handle exponentiation, modulo, or other mathematical functions.
    [/LIMITATIONS]
    """
    operations = {
        "add": lambda: x + y,
        "subtract": lambda: x - y,
        "multiply": lambda: x * y,
        "divide": lambda: x / y if y != 0 else "Error: Division by zero",
    }
    if operation not in operations:
        raise ValueError(f"Invalid operation: {operation}")
    return operations[operation]()


@tool
def percentage_calculator(value: float, percentage: float = 100.0) -> float:
    """[BRIEF] Calculate a given percentage of a numerical value. [/BRIEF]

    [DETAILED] This tool computes the result of applying a percentage to a base value using the formula: (value * percentage) / 100. It is useful for markup/discount calculations, proportional scaling, and any task requiring percentage-based computation. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to calculate a percentage of a given number.
    - When performing discount, tax, or proportion calculations.
    - Recommended for any task that involves percentage-based arithmetic. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Identify the base value and the percentage from the task description. [/PREREQUISITE]
        2. [CURRENT] Call this tool with the base value and the desired percentage. [/CURRENT]
        3. [FOLLOW_UP] Use the result for further calculations or report it as the final answer. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Multiplies the base value by the percentage and divides by 100.
    - The percentage parameter defaults to 100.0, which returns the original value. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `percentage_calculator(200.0, 15.0)`,
        `percentage_calculator(50.0, 50.0)`,
        `percentage_calculator(1000.0)`,
    ]
    [/SYNTACTICAL]

    Args:
        value (float):
            [ARGS_BRIEF] The base value to calculate the percentage of. [/ARGS_BRIEF]
            [ARGS_DETAILED] The numerical value on which the percentage calculation is performed. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Any valid floating-point number. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] 200.0, 50.0, 1000.0, -30.0 [/ARGS_EXAMPLES]
        percentage (float):
            [ARGS_BRIEF] The percentage to apply (defaults to 100.0). [/ARGS_BRIEF]
            [ARGS_DETAILED] The percentage value to calculate. A value of 50.0 means 50%, 100.0 means 100% (returns the original value). Defaults to 100.0 if not specified. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Any valid floating-point number representing a percentage. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] 15.0, 50.0, 100.0, 200.0 [/ARGS_EXAMPLES]

    Returns:
        float:
            [RETURNS_BRIEF] The calculated percentage of the value. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The result of (value * percentage) / 100. For example, 15% of 200 returns 30.0. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] 30.0 for (200, 15), 25.0 for (50, 50), 1000.0 for (1000, 100) [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        TypeError:
            [ERROR_WHEN] If the inputs are not numerical types. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when value or percentage cannot be used in arithmetic operations, e.g., if a string is passed instead of a number. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure both value and percentage are valid numbers (int or float). [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
    - Does not handle non-numeric inputs gracefully; relies on Python's type system.
    - No built-in rounding; results may have floating-point precision artifacts.
    [/LIMITATIONS]
    """
    return (value * percentage) / 100.0


class UnitConverterTool(Tool):
    def __init__(self):
        super().__init__(
            name="unit_converter",
            description="Convert between different units",
            params_json_schema=arguments_to_schema(
                [
                    ToolArgument("value", "float", "Value to convert"),
                    ToolArgument("from_unit", "str", "Original unit (m, kg, s)"),
                    ToolArgument("to_unit", "str", "Target unit (cm, g, ms)"),
                ]
            ),
        )

    def execute(self, value: float, from_unit: str, to_unit: str) -> str:
        conversions = {
            ("m", "cm"): lambda x: x * 100,
            ("kg", "g"): lambda x: x * 1000,
            ("s", "ms"): lambda x: x * 1000,
        }

        key = (from_unit, to_unit)
        if key not in conversions:
            raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")

        return str(conversions[key](value))
