from config import client, MODEL, ITEMS, QUESTIONS, banner

PRICE_LIST = ", ".join(f"{item} = Rs. {price}" for item, price in ITEMS.items())

def chatBot(question):
    response = client.chat.completions.create(
        model = MODEL,
        messages=[
            {"role": "system", "content": f"You are a helpful student budget assistant. "
                                          f"These are the only items we sell with their prices: {PRICE_LIST}. "
                                          f"Answer using only these prices, in one or two short sentences."},
            {"role": "user", "content": question}
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

if(__name__ == "__main__"):
    banner("SYSTEM 1: CHATBOT")
    for question in QUESTIONS:
        print("Q: ", question)
        print("A: ", chatBot(question))
        print("-" * 70)