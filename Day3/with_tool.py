"""Day 3 - Part 2: the same questions, but now the LLM has the calculator tool.

The model decides on its own whether it needs the tool. We only run the tool when the
model asks for it, in one small loop, until the model writes a normal answer.
"""
import json

from config import client, MODEL, QUESTIONS, banner
from tool import calculator, TOOL

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the calculator tool for any arithmetic, "
    "especially percentages, discounts and multi-step sums. "
    "For questions that need no calculation, just answer directly."
)


def ask_llm_with_tool(question):
    """Ask the question, run the tool when the model asks for it, return the final answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    while True:
        # 1. ask the model, and let it use the tool if it wants to
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=[TOOL], temperature=0
        )
        message = response.choices[0].message

        # 2. no tool call means the model is done, so this is the final answer
        if not message.tool_calls:
            print("   tool used: NO (answered directly)")
            return message.content.strip()

        # 3. the model asked for the tool, so we run it and give the result back
        for call in message.tool_calls:
            arguments = json.loads(call.function.arguments or "{}")
            result = calculator(**arguments)

            print(f"   tool used: YES -> calculator({arguments})")
            print(f"   tool result: {result}")

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                ],
            })
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        # the loop goes round again so the model can use the result, or call the tool again


if __name__ == "__main__":
    banner("WITH TOOL (LLM + calculator)")
    for number, question in enumerate(QUESTIONS, start=1):
        print(f"Q{number}: {question}")
        print(f"A{number}: {ask_llm_with_tool(question)}")
        print("-" * 70)
