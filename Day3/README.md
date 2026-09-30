# Day 3 - From Prompt to Action: LLM With and Without a Tool

## What this project shows

The same three questions are sent to the same LLM twice:

1. **Without a tool** (`no_tool.py`) - the model can only answer from its own memory.
2. **With one tool** (`with_tool.py`) - the model can call a small `calculator` function
   whenever it thinks a calculation is needed.

The scenario is small money maths: a long shop bill with GST, a jacket with a discount,
and one normal question that needs no maths at all. The difference you can see clearly is
that the tool version prints every expression it hands to the calculator, so the number in
the answer can be checked, while the plain version only gives you the final text.

## Folder structure

```
Assignment_Day1/            <- shared .env, requirements.txt and .venv
├── .env                   # API key (already set up)
├── requirements.txt       # openai + python-dotenv (already installed)
├── .venv/                 # virtual environment to activate
└── Day3/
    ├── config.py          # LLM connection (provider, key, model) and the three questions
    ├── tool.py            # the one tool: calculator() + its schema for the model
    ├── no_tool.py         # run 1: plain LLM, no tool
    ├── with_tool.py       # run 2: LLM that may call the calculator once
    ├── analysis.md        # concepts, comparison table, observations (fill in after running)
    ├── README.md
    └── screenshots/       # put your real terminal screenshots here
```

## Install

Nothing to install for this folder. The `.env`, `requirements.txt` and `.venv` are already
set up one folder up in `Assignment_Day1`, and Day3 uses the same ones as the earlier
days. Just activate that venv:

```bash
cd Day3
source ../.venv/bin/activate
```

The only two packages this project needs are `openai` and `python-dotenv`, and both are
already listed in `../requirements.txt`.

## API key setup

No new setup needed either. `config.py` calls `load_dotenv()`, which walks up the folders
and finds `../.env`, the same key file the earlier days use:

```
PROVIDER=groq
GROQ_API_KEY=your_real_key_here
MODEL=openai/gpt-oss-20b
```

The key stays in that one `.env` file and is never committed (`.env` is in `.gitignore`).

## Run

Run this first, it only tests the tool, no API call:

```bash
python tool.py
```

Then the two runs:

```bash
python no_tool.py
python with_tool.py
```

## What you should observe

- **Question 2** ("In one short sentence, what is Python used for?") needs no maths. Both
  versions answer it correctly, and the tool version prints `tool used: NO` and answers
  directly. That is the case where a tool is not needed.
- **Questions 1 and 3** need arithmetic. The tool version prints
  `tool used: YES -> calculator({'expression': ...})` and then `tool result: ...`, so you
  can see the exact maths behind the final number. Question 1 needs two tool calls, so the
  model adds up the items first and then applies the GST and delivery.
- The plain LLM may still get these numbers right - `gpt-oss-120b` is a strong model. If it
  does, the honest conclusion is that the tool did not change the answer, it made the
  answer checkable. `analysis.md` Section 4 reports what actually happened in my run.
- If the model ever sends a bad expression, the tool returns a message like
  `Calculator error: ...` instead of crashing, and the program keeps going.
