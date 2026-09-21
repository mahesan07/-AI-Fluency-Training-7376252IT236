# Analysis: Student Budget & Purchase Assistant

## Scenario

A student with a fixed budget (e.g. Rs. 10,000) wants to know which two accessories (keyboard, headphones, mouse, backpack, USB drive, notebook, calculator, webcam) they can buy together. Item prices are **private data** — stored only in a Python dictionary in our program. The model must never invent a price.

---

## 3.1 Explanation of each approach

### Plain chatbot (chatbot.py)

LLM only. We send the user's question to the model, plus a system message containing the price list. The model answers from what it saw in the prompt.
- **Data/private access:** gets prices only because we paste them into the prompt; it cannot read our dictionary.
- **Tools/rules:** none.
- **Request flow:** question → single LLM call → text answer.
- **Limitation:** it cannot really calculate. For "which two items fit Rs. 10,000?" it describes, it doesn't verify. It guesses instead of computing.

### Rule-based workflow (workflow.py)

No LLM at all. A fixed Python function with hard-coded steps.
- **Data/private access:** reads the price dictionary directly each run.
- **Tools/rules:** uses the tools, but in a fixed sequence written by the programmer.
- **Request flow:** extract budget → get all prices → loop over every pair → calculate total → check budget → keep pairs that fit → return plain-text list.
- **Limitation:** rigid. Only handles questions with a rupee amount, only two-item pairs. New question type = new code.

### AI agent (agent.py)

LLM + Tools + Loop. The model chooses and calls tools by itself.
- **Data/private access:** reads private prices only through the tools we expose (get_item_price, calculate_total, check_budget).
- **Request flow:** question → model requests a tool call → we run it on real data → result sent back → model decides next step → loop repeats until it returns a text answer.
- **Limitation:** more API calls and a slightly unpredictable path, but the numbers are always real because they come from the tools.

---

## 3.2 Comparison table

| Basis | Plain chatbot | Rule-based workflow | AI agent |
| --- | --- | --- | --- |
| **Flexibility** | Low — answers from memory, can't adapt method. | Very low — fixed steps, new questions need new code. | High — reads the question and picks tools accordingly. |
| **Decision-making** | None — model just generates text. | None — every decision pre-made in code. | Yes — decides which tool, what args, when to stop. |
| **Tool usage** | No tools. | Tools, but in a fixed hard-coded script. | Same tools, chosen and sequenced by the model via the loop. |
| **Private-data access** | Only what we paste into the prompt; otherwise guesses. | Reads the real dictionary every time. | Only through exposed tools; never invents prices. |
| **Multi-step handling** | No — single-shot answer. | Yes, but same fixed path every time. | Yes — chains steps, observes results, continues until done. |
| **Automation** | Minimal — each question is separate. | Fully automatic, but only for its one problem. | Self-directed multi-step automation. |
| **Reliability** | Low — can give a wrong total confidently. | High — identical arithmetic every run. | Medium-high — real numbers, slight loop variability. |

---

## 3.3 Suitability analysis

The **rule-based workflow** is most suitable for the core question "which two items fit this budget": the task is fixed, the prices are static, and correctness matters most. As the table shows, it reads private data directly, gives identical results every run, and can never invent a price — best reliability and zero model cost.

The **AI agent** is the better choice once the question is open-ended (e.g. "what combination can I buy?") because it adds flexibility and decision-making while still getting prices from real tools. The **plain chatbot** is the least suitable here — it looks confident but cannot verify totals, which is dangerous for a budget question.

**Best fit:** rule-based workflow for the narrow, repeated task; AI agent when questions vary; chatbot for nothing in this scenario.

---

## 3.4 Conclusion

- **Chatbot** is right for conversational, open-ended problems where exact data/calculations aren't critical (explaining, brainstorming, advice).
- **Rule-based workflow** is right for fixed, well-known problems needing exact answers every time (formulas, validations, fixed conditions) — but only when the question set is small and stable.
- **AI agent** is right for multi-step problems needing real private data and decisions the programmer can't foresee (live prices, comparisons, varied questions) — most flexible, at the cost of more API calls.

This project shows the trade-off clearly: the same budget question solved three ways — LLM only, predefined steps, and LLM + Tools + Loop.