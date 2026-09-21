from workflow import workflow
from agent import agent

QUESTION = "I have Rs. 10,000. Which two student accessories can I buy together within this budget?"

print("Q:", QUESTION)

print("\n=== WORKFLOW (fixed steps, no thinking) ===")
print("Workflow answer :", workflow(QUESTION))

print("\n=== AGENT (decides which tools to call) ===")
print("Agent answer    :", agent(QUESTION))