# Day 3 Analysis - From Prompt to Action: LLM With and Without a Tool

## The scenario

A student does some small money maths:

1. A ten-line shop bill, then 18% GST, then Rs. 150 delivery.
2. "In one short sentence, what is Python used for?"
3. A jacket for Rs. 4999 with 10% off and Rs. 250 delivery.

The only tool is a small `calculator` function. The same three questions go to the same
model twice: once with no tool, once with the tool.

---

## Section 1 - The concepts

### 1.1 What is an LLM?

A program that has read a lot of text and learned its patterns. It answers by predicting
the next words, over and over. It never looks anything up and it never calculates, it
only writes text.

- **Good at:** language questions it has seen much text about, so "What is Python used
  for?" is easy for it.
- **Weak at:** exact arithmetic. A ten-line bill means ten additions and a multiplication
  done in its head, and one wrong digit changes the answer. It also never says "I am not
  sure" about a number.

Key point: for a plain LLM, *answering* and *calculating* are the same action, so we
cannot trust it with a number that has to be exact.

### 1.2 What is an agent?

A plain LLM is just this:

```
question  ->  LLM  ->  text answer
```

It reads and writes. Nothing else runs.

An agent is the **same model** plus two things:

1. It knows a tool exists (name, description, parameters).
2. It can answer "call this tool with these arguments" instead of answering the user.

Our Python code runs the tool, gives the result back, and the model answers from that
result. The model does not become smarter, it just gains a way to **act** and to **check**
itself instead of only speaking.

### 1.3 What is a tool?

An ordinary Python function that our program exposes to the model. The model cannot run
Python, so our program runs it for them.

A **tool call** is only a request from the model to our program: which tool, which
arguments. Nothing has happened yet.

The `TOOL` schema in `tool.py` has three parts:

- **name** - `calculator`. The word the model must use.
- **description** - "Calculate an arithmetic expression. Use it for any maths such as
  percentages, discounts, totals and multi-step sums." This tells it *when* to use it.
- **parameters** - one string parameter, `expression`, and it is required. This tells it
  *what to send*. It cannot guess the argument name.

The model needs all three because it has to make three decisions: which tool, whether to
use it at all, and with what arguments.

### 1.4 One tool call, step by step

Using question 3 (jacket, 10% off, Rs. 250 delivery):

1. The user asks the question, and it is sent to the model along with the tool schema.
2. The model sees the question and knows a calculator is available.
3. The model decides a tool is needed, so it returns a tool call instead of an answer.
   This is the moment where speaking turns into acting.
4. The model requests `calculator({"expression": "4999 * 0.9 + 250"})`. Only a request so
   far, nothing is calculated.
5. Our code reads the name and the arguments and calls the Python function. Python does the
   real maths.
6. The function returns the text `"4749.1"`.
7. We add that text back as a `tool` message and call the model again. Now the model can
   see a real number instead of guessing one.
8. The model writes the final answer, "You pay Rs. 4749.10", and that number came from
   Python.

`no_tool.py` only does steps 1, 2 and 8, because there is no tool to ask for.

**One thing I learned by running it:** the model can need the tool more than once. For
question 1 it called it once to add the ten items, then again for
`25484.56 * 1.18 + 150`. That is why `with_tool.py` uses a small loop instead of one extra
call.

### 1.5 Why the tool returns text instead of raising an error

If `calculator` raised an exception, the program would crash with a traceback and the model
would never get a chance to fix its own mistake.

Instead it catches the problem and returns a normal string:
`Calculator error: ... Use only numbers and + - * / ** and brackets.`

- the program keeps running, so the next question is still asked;
- the model can read the message and correct the expression;
- the failure is visible, instead of a wrong number quietly appearing in the answer.

---

## Section 2 - Comparison table

| Basis for comparison | Plain LLM prompt (no tool) | LLM with one tool |
| --- | --- | --- |
| Source of the answer | The model writes the number by predicting text. | Python calculates, and the model writes the sentence around the real result. |
| Can it fetch or compute information outside its own memory? | No. It can only guess, in a confident tone. | Yes for arithmetic only. Sums, percentages and discounts become exact and visible. |
| Reliability on factual or numeric questions | Unreliable on long maths, and there is no way to check it yourself. | High, because Python is exact and returns the same answer every run. |
| Transparency (can you see how the answer was reached?) | Low. Only the final text, so you cannot see which step went wrong. | High. The exact expression, the exact result and the final answer are all printed. |
| Speed / cost of getting an answer | One API call. Fastest and cheapest. | More calls when the tool is used (question 1 used three). Costs more, but the number is provable. |

---

## Section 3 - Minimal implementation

Four Python files and nothing else.

- **`tool.py`** - one function, `calculator(expression)`. It parses the expression with
  Python's `ast` module and allows only `+ - * / **` and brackets, so nothing else can
  sneak in. Bad input returns a `Calculator error:` string. The same file holds the `TOOL`
  schema.
- **`no_tool.py`** - loops over the three questions and sends each one to the model with no
  `tools=` argument, so the model does not know a calculator even exists.
- **`with_tool.py`** - sends the same questions with `tools=[TOOL]` and runs one small
  loop: if the model asks for the tool, print the call, run the function, send the result
  back and ask again; if the model sends no tool call, print the answer and stop.
- **`config.py`** - the API connection and the three questions.

**How the call works:** the model cannot run code, so it returns JSON containing the tool
name and the arguments. `with_tool.py` reads it with `json.loads`, runs the Python
function, appends the result as a `tool` message, and calls the API again.

---

## Section 4 - Observation

> These are the real outputs from my run of `no_tool.py` and `with_tool.py` (provider
> `groq`, model `openai/gpt-oss-120b`, temperature 0). I still have to add the terminal
> screenshots to `screenshots/`.

| # | Question | No-tool result | Tool used? | Tool-enabled result | Observation |
| --- | --- | --- | --- | --- | --- |
| 1 | Ten-line bill + 18% GST + Rs. 150 delivery | Rs 30,221.78 | YES, twice | Rs 30,221.78 | Both correct. The tool version called `234.56 + ... + 2222.22` giving 25484.56, then `25484.56 * 1.18 + 150` giving 30221.78. |
| 2 | What is Python used for? | Correct answer | NO | Correct answer | The model chose not to use the tool, which is right, no maths needed. |
| 3 | Jacket Rs. 4999, 10% off + Rs. 250 delivery | Rs 4,749.10 | YES, once | Rs 4,749.10 | Both correct. The tool version called `4999 * 0.9 + 250` giving 4749.1. |

**The honest result is that both versions gave the same answers.** I expected the plain LLM
to get question 1 wrong, but `gpt-oss-120b` is a strong reasoning model and it did the
maths correctly. I am not going to invent a wrong answer just to make the table look
better.

What the tool still gave me:

- **Proof instead of trust.** The exact expressions and results are printed, so I did not
  have to check the sums myself, and if an answer had been wrong I would have seen exactly
  which expression was wrong.
- **Repeatability.** Python gives the same result every run.
- **A visible decision.** `tool used: YES` and `tool used: NO` show the model choosing.

**A failure I hit earlier, worth reporting.** My first wording of question 1 was "bill Rs.
12345.67, paid Rs. 2345.10, a friend gave me Rs. 678.90, paid Rs. 89.99". The model decided
the money it received should be *added*, and called the tool with
`2345.10 + 678.90 + 89.99`, which correctly returned 3113.99. The calculator worked
perfectly and the final answer was still wrong, because the model chose the wrong
expression. A tool fixes calculation, not thinking.

**A real API error I hit.** My first version of `with_tool.py` did not pass `tools=` on the
final call, and the API replied `400 - Tool choice is none, but model called a tool`.
Passing the tools on every call and letting the small loop repeat fixed it.

---

## Section 5 - Suitability and conclusion

**When the plain LLM was enough:** question 2, and honestly question 3 as well, since the
model got both right on its own. A tool there only added cost.

**When the tool was needed:** question 1. Ten additions, a percentage and a fee is exactly
where one wrong digit changes the answer and nothing signals it. The tool costs extra API
calls, so it buys exactness and visibility, and I pay for that in time and tokens.

**Why it improved reliability:** the number stopped coming from the model's head and came
from Python. This run the two answers matched, so the improvement is not that the answer
changed, it is that the tool version is provable and repeatable.

**A plain LLM is enough when** the answer is about language, the question is general
knowledge, being roughly right is acceptable, and speed matters.

**Give it a tool when** the answer depends on exact maths, money, dates or counts, when it
depends on data the model cannot know, or when a wrong answer costs something and the
working has to be shown.

**Conclusion:** the same model gave the same answer with and without the calculator, which
is not what I expected. The real difference was proof, not accuracy. The biggest lesson for
me is that an agent is not a better LLM, it is the same LLM plus a tool, the model stays in
charge of when to use it, and a tool fixes calculation, not thinking.
