from corral.core.tool import tool


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


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """[BRIEF] Convert a value between supported metric units. [/BRIEF]

    [DETAILED] This tool converts a numerical value from a source unit to a target unit for a fixed set of metric conversions: metres to centimetres, kilograms to grams, and seconds to milliseconds. It looks up the (from_unit, to_unit) pair in a conversion table and applies the corresponding scaling factor. Unsupported unit pairs raise an error rather than returning a best-effort guess. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to convert a value between one of the supported metric unit pairs.
    - When a task expresses quantities in one unit but requires them in another (e.g. metres vs centimetres).
    - Recommended as the primary tool for any supported metric unit conversion. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Identify the value to convert and its current and target units from the task description. [/PREREQUISITE]
        2. [CURRENT] Call this tool with the value, the source unit, and the target unit. [/CURRENT]
        3. [FOLLOW_UP] Use the converted value in further calculations (e.g. feed it into `calculator`) or report it as the final answer. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Builds a lookup table keyed by the (from_unit, to_unit) pair.
    - Retrieves the scaling function for the requested pair and applies it to the value.
    - Raises ValueError if the requested (from_unit, to_unit) pair is not in the table.
    - Returns the converted value as a string. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `unit_converter(1.5, "m", "cm")`,
        `unit_converter(2.0, "kg", "g")`,
        `unit_converter(0.25, "s", "ms")`,
    ]
    [/SYNTACTICAL]

    Args:
        value (float):
            [ARGS_BRIEF] The numerical value to convert. [/ARGS_BRIEF]
            [ARGS_DETAILED] The magnitude expressed in the source unit that will be scaled into the target unit. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Any valid floating-point number. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] 1.5, 2.0, 0.25, 100.0 [/ARGS_EXAMPLES]
        from_unit (str):
            [ARGS_BRIEF] The unit the value is currently expressed in. [/ARGS_BRIEF]
            [ARGS_DETAILED] The source unit of the value. Must be the first element of a supported conversion pair. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] One of: "m", "kg", "s" [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "m", "kg", "s" [/ARGS_EXAMPLES]
        to_unit (str):
            [ARGS_BRIEF] The unit to convert the value into. [/ARGS_BRIEF]
            [ARGS_DETAILED] The target unit of the conversion. Must pair with from_unit as a supported conversion (m->cm, kg->g, s->ms). [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] One of: "cm", "g", "ms" [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "cm", "g", "ms" [/ARGS_EXAMPLES]

    Returns:
        str:
            [RETURNS_BRIEF] The converted value as a string. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The result of applying the scaling factor for the (from_unit, to_unit) pair to value, formatted as a string. For example, converting 1.5 m to cm returns "150.0". [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "150.0" for (1.5, "m", "cm"), "2000.0" for (2.0, "kg", "g"), "250.0" for (0.25, "s", "ms") [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the (from_unit, to_unit) pair is not one of the supported conversions. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the requested source and target unit combination has no entry in the conversion table (only m->cm, kg->g, and s->ms are supported). [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure from_unit and to_unit form one of the supported pairs: m->cm, kg->g, or s->ms. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
    - Supports only three fixed conversion pairs (m->cm, kg->g, s->ms); no inverse or chained conversions.
    - Returns the result as a string rather than a numeric type, which may require parsing before further arithmetic.
    [/LIMITATIONS]
    """
    conversions = {
        ("m", "cm"): lambda x: x * 100,
        ("kg", "g"): lambda x: x * 1000,
        ("s", "ms"): lambda x: x * 1000,
    }

    key = (from_unit, to_unit)
    if key not in conversions:
        raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")

    return str(conversions[key](value))
