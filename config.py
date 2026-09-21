import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()

if PROVIDER == "ollama":
    BASE_URL = "http://localhost:11434/v1"
    API_KEY = "ollama"
    MODEL = os.getenv("MODEL", "qwen2.5:1.5b")
elif PROVIDER == "groq":
    BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY = os.getenv("GROQ_API_KEY")
    MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")
elif PROVIDER == "huggingface":
    BASE_URL = "https://router.huggingface.co/v1"
    API_KEY = os.getenv("HF_TOKEN")
    MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")
else :
    raise SystemExit(f"Unknown Provider `{PROVIDER}`.Use ollama, groq, huggingface.")

if not API_KEY:
    raise SystemExit(f"No API key found for PROVIDER={PROVIDER}. Check your .env file.")
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

ITEMS = {
    "keyboard": 3000,
    "headphones": 5000,
    "mouse": 1500,
    "backpack": 2000,
    "usb_drive": 1200,
    "notebook": 300,
    "calculator": 800,
    "webcam": 2500
}

QUESTIONS = {
    "What is the price of a keyboard?",
    "I have Rs. 10000. Which two items can I buy together?",
    "Can I buy a keyboard and headphones with Rs. 7000?",
    "I have Rs. 12000. What combination of student accessories can I buy?"
}
def banner(system_name):
    print(f"\n==={system_name} | provider: {PROVIDER} | model : {MODEL} ===\n")