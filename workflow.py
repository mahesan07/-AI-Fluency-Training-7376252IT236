import re
from config import ITEMS, QUESTIONS, banner
from tools import TOOL_FUNCTIONS

def get_budget(question):
    match = re.search(r"(\d[\d,]*)", question)
    if match:
        return int(match.group(1).replace(",", ""))
    return None

def workflow(question):
    budget = get_budget(question)
    if budget is None:
        return "Sorry, I can only answer questions that include a budget in rupees."

    step = 0
    combos = []
    names = list(ITEMS.keys())

    step += 1
    print(f"step {step}: read budget Rs. {budget} from the question")

    for name in names:
        step += 1
        args = {"item_name": name}
        result = TOOL_FUNCTIONS["get_item_price"](**args)
        print(f"step {step}: get_item_price({args}) -> {result}")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = [names[i], names[j]]

            step += 1
            total_args = {"items": pair}
            total = TOOL_FUNCTIONS["calculate_total"](**total_args)
            print(f"step {step}: calculate_total({total_args}) -> {total}")

            step += 1
            budget_args = {"total": int(total), "budget": budget}
            check = TOOL_FUNCTIONS["check_budget"](**budget_args)
            print(f"step {step}: check_budget({budget_args}) -> {check}")

            if int(total) <= budget:
                combos.append((names[i], names[j], int(total)))

    if combos:
        lines = [f"Within a budget of Rs. {budget}, these pairs of items fit:"]
        for a, b, total in combos:
            lines.append(f"- {a} + {b} = Rs. {total}")
        result_text = "\n".join(lines)
    else:
        result_text = f"No pair of items fits inside a budget of Rs. {budget}."

    step += 1
    print(f"step {step}: build a plain-text answer from the found combinations")
    return result_text

if __name__ == "__main__":
    banner("SYSTEM 2: RULE-BASED WORKFLOW (no LLM)")
    for question in QUESTIONS:
        print("Q:", question)
        print("A:", workflow(question))
        print("-" * 70)