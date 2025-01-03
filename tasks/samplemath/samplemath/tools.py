import modal
from modal import Image

from corral.base import Tool, ToolArgument
from corral.utils import MODAL_TOOL_REGISTRY, modal_tool, tool

app = modal.App("cooral-test")


@tool
def calculator(operation: str, x: float, y: float) -> float:
    """Perform basic math operations.

    Args:
        operation: Operation to perform (choices: ["add", "subtract", "multiply", "divide"])
        x: First number
        y: Second number
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
    """Calculate percentage of a value.

    Args:
        value: The base value
        percentage: The percentage to calculate (defaults to 100.0)

    Returns:
        float: The calculated result
    """
    return (value * percentage) / 100.0


@modal_tool(app=app, image=Image.debian_slim().pip_install("numerizer"), memory=512)
def number_convert(text: str, return_float: bool = False) -> str:
    """
    Convert number words to numeric representation.

    Args:
        text: Text containing number words (e.g. 'forty two', 'one million')
        return_float: Whether to return float for decimal values (choices: [True, False])

    Returns:
        String containing the numeric representation
    """
    from numerizer import numerize

    result = numerize(text)
    return str(float(result)) if return_float and "." in result else result


# TODO: decorator could not return the tool instance
number_converter = MODAL_TOOL_REGISTRY["number_convert"]


class UnitConverterTool(Tool):
    def __init__(self):
        super().__init__(
            name="unit_converter",
            description="Convert between different units",
            arguments=[
                ToolArgument("value", "float", "Value to convert"),
                ToolArgument("from_unit", "str", "Original unit (m, kg, s)"),
                ToolArgument("to_unit", "str", "Target unit (cm, g, ms)"),
            ],
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
