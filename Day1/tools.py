from config import ITEMS

def get_item_price(item_name):
    price = ITEMS.get(item_name.strip().lower())
    if price is None:
        return f"Unknown item: {item_name}. Available items: {', '.join(ITEMS)}"
    return str(price)

def calculate_total(items):
    total = 0
    for item in items:
        price = ITEMS.get(item.strip().lower())
        if price is None:
            return f"Unknown item: {item}"
        total += price
    return str(total)

def check_budget(total, budget):
    total = int(total)
    budget = int(budget)
    difference = budget - total
    if difference >= 0:
        return f"Yes, within budget. Rs. {difference} left."
    return f"No, Rs. {abs(difference)} over budget."

TOOL_FUNCTIONS = {
    "get_item_price": get_item_price,
    "calculate_total": calculate_total,
    "check_budget": check_budget,
}
TOOLS = [
    {"type": "function", "function": {
        "name": "get_item_price",
        "description": "Get the price in rupees of a single student item, for example 'keyboard'.",
        "parameters": {"type": "object",
                       "properties": {"item_name": {"type": "string"}},
                       "required": ["item_name"]}}},
    {"type": "function", "function": {
        "name": "calculate_total",
        "description": "Add up the prices of a list of item names, for example ['keyboard', 'headphones'].",
        "parameters": {"type": "object",
                       "properties": {"items": {"type": "array",
                                                "items": {"type": "string"}}},
                       "required": ["items"]}}},
    {"type": "function", "function": {
        "name": "check_budget",
        "description": "Check whether a total price fits inside a budget in rupees.",
        "parameters": {"type": "object",
                       "properties": {"total": {"type": "number"},
                                      "budget": {"type": "number"}},
                       "required": ["total", "budget"]}}},
]

if __name__ == "__main__":
    print("get_item_price('keyboard') ->", get_item_price("keyboard"))
    print("get_item_price('mouse') ->", get_item_price("mouse"))
    print("calculate_total(['keyboard', 'headphones']) ->", calculate_total(["keyboard", "headphones"]))
    print("check_budget(8000, 10000) ->", check_budget(8000, 10000))
    print("check_budget(8000, 7000) ->", check_budget(8000, 7000))
    print("get_item_price('laptop') ->", get_item_price("laptop"))