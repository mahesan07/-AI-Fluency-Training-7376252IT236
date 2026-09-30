"""Day 3 - small setup file: the LLM connection and the three questions."""
import os
from dotenv import load_dotenv
from openai import OpenAI

# The real .env file is kept one folder up, in Assignment_Day1, so Day3 reuses it.
# load_dotenv() with no argument walks up the folders and finds that file.
load_dotenv()

PROVIDER = os.getenv("PROVIDER", "groq").strip().lower()

if PROVIDER == "groq":
    BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY = os.getenv("GROQ_API_KEY")
    MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")
elif PROVIDER == "huggingface":
    BASE_URL = "https://router.huggingface.co/v1"
    API_KEY = os.getenv("HF_TOKEN")
    MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")
elif PROVIDER == "ollama":
    BASE_URL = "http://localhost:11434/v1"
    API_KEY = "ollama"
    MODEL = os.getenv("MODEL", "qwen2.5:1.5b")
else:
    raise SystemExit(f"Unknown PROVIDER '{PROVIDER}'. Use groq, huggingface or ollama.")

if not API_KEY:
    raise SystemExit(f"No API key found for PROVIDER={PROVIDER}. Please check your .env file.")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# The same three questions are sent to both scripts, so the answers can be compared.
# Q1 needs the calculator, Q2 does not need it at all, Q3 is a small one-step sum.
QUESTIONS = [
    "A shop bill has these lines: Rs. 234.56, Rs. 1234.78, Rs. 89.90, Rs. 4567.89, "
    "Rs. 345.67, Rs. 6789.54, Rs. 12.35, Rs. 9876.54, Rs. 111.11 and Rs. 2222.22. "
    "Add 18% GST on the item total, then add a Rs. 150 delivery charge. "
    "How much do I pay in total?",

    "In one short sentence, what is Python used for?",

    "A jacket costs Rs. 4999. The shop gives 10% off and then adds Rs. 250 delivery charge. "
    "How much do I pay in total?",
]


def banner(run_name):
    print(f"\n=== {run_name} | provider: {PROVIDER} | model: {MODEL} ===\n")
