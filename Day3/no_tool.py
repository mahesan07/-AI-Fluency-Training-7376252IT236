"""Day 3 - Part 1: the questions sent to the LLM with NO tool.

The model only gets the question. It cannot use the calculator, so it has to
answer from its own memory.
"""
from config import client, MODEL, QUESTIONS, banner

SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question clearly and shortly."


def ask_llm(question):
    """Send one question to the LLM and return the text answer."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    banner("NO TOOL (plain LLM)")
    for number, question in enumerate(QUESTIONS, start=1):
        print(f"Q{number}: {question}")
        print(f"A{number}: {ask_llm(question)}")
        print("-" * 70)
