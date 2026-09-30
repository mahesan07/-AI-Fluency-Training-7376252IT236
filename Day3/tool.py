"""Day 3: the one and only tool of this project - a small calculator.

The tool takes a normal arithmetic expression as a string, calculates it, and
returns the answer as plain text. That is all it does.
"""
import ast
import operator

# Only these operations are allowed, so the model cannot run any Python code.
ALLOWED_OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def calculate(node):
    """Walk through the expression tree and do the maths."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATIONS:
        left = calculate(node.left)
        right = calculate(node.right)
        return ALLOWED_OPERATIONS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATIONS:
        return ALLOWED_OPERATIONS[type(node.op)](calculate(node.operand))
    raise ValueError("only numbers and + - * / ** and brackets are allowed")


def calculator(expression: str) -> str:
    """Calculate an expression such as '4999 * 0.9 + 250' and return plain text."""
    try:
        value = calculate(ast.parse(expression, mode="eval").body)
        # we round money answers to 2 decimals, otherwise Python shows 80987.54999999999
        if isinstance(value, float):
            value = round(value, 2)
        return str(value)
    except Exception as error:
        # We return the problem as text instead of crashing, so the model can read it
        # and correct itself.
        return f"Calculator error: {error}. Use only numbers and + - * / ** and brackets."


# The schema we send to the model. It tells the model the tool name, what the tool
# does and which parameters it needs.
TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate an arithmetic expression. Use it for any maths such as "
                       "percentages, discounts, totals and multi-step sums. "
                       "Example: '4999 * 0.9 + 250'.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to calculate, for example (1200 + 800) * 0.9",
                }
            },
            "required": ["expression"],
        },
    },
}


if __name__ == "__main__":
    # quick test of the tool, no LLM needed
    print(calculator("4999 * 0.9 + 250"))
    print(calculator("12345.67 - 2345.10 + 678.90 - 89.99"))
    print(calculator("5 / 0"))
    print(calculator("import os"))
