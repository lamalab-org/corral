from corral.base import Tool, ToolArgument


class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform basic math operations",
            arguments=[
                ToolArgument(
                    "operation",
                    "str",
                    "Operation to perform: add, subtract, multiply, divide",
                ),
                ToolArgument("x", "float", "First number"),
                ToolArgument("y", "float", "Second number"),
            ],
        )

    def execute(self, operation: str, x: float, y: float) -> str:
        operations = {
            "add": lambda: x + y,
            "subtract": lambda: x - y,
            "multiply": lambda: x * y,
            "divide": lambda: x / y if y != 0 else "Error: Division by zero",
        }

        if operation not in operations:
            raise ValueError(f"Invalid operation: {operation}")

        return str(operations[operation]())


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
