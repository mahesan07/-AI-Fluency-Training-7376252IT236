# Day 4 — Local LLM Memory Estimator

A beginner-level project that estimates how much RAM a local LLM would need on a
CPU-only laptop, and compares four small models using official web information.

> **No model was downloaded or run locally for this assignment.**
> `ollama pull`, `ollama run` and `ollama rm` were **not** used. All model facts
> come from official Hugging Face model cards and the official Ollama library pages.

---

## What this project does

It answers one question: *could a small LLM fit on a ~15 GB Linux Mint laptop used
as a personal chatbot / coding helper?*

It does this in three steps:

1. **Estimate memory** using a simple formula (weights + KV cache + overhead).
2. **Run two experiments** — one for context length, one for quantization.
3. **Compare four model families** using only official web pages and licences.

---

## Folder structure

```
Day_4/
│
├── estimator.py      # memory estimator: weights, KV cache, overhead, fit / no-fit
├── comparison.py     # context experiment, quantization experiment, model comparison
├── analysis.md       # the full written analysis (13 sections)
├── README.md         # this file
└── screenshots/      # (empty — you take the screenshots yourself)
```

---

## How to run

The scripts use only plain Python (no extra packages needed). Run them from inside
the `Day_4` folder:

```bash
cd Day_4
python3 estimator.py
python3 comparison.py
```

`comparison.py` imports `estimator.py`, so run it from inside the `Day_4` folder so
Python can find the module.

---

## The memory formula

```
Weights    = parameters × bytes_per_weight
KV cache   = 2 × layers × kv_heads × head_dim × context_tokens × 2 bytes
Overhead   = 0.5 GB + 10% of the weights
Total      = Weights + KV cache + Overhead
Fits?      = Total ≤ 12 GB   (15 GB RAM minus ~3 GB for the OS)
```

Bytes per parameter used: FP16 = 2.0, Q8 = 1.0, Q4 = 0.5.

---

## Models compared (web research only, checked 1 October 2026)

| Model | Publisher | Params | Context | Licence | Ollama Q4 |
|---|---|---|---|---|---|
| Llama 3.2 3B | Meta | 3.21B | 128K | Llama 3.2 Community License | 2.0 GB |
| Qwen3 4B | Alibaba | 4.02B | 32K (131K w/ YaRN) | Apache License 2.0 | 2.5 GB |
| Gemma 3 4B | Google | 4.30B | 128K | Gemma Terms of Use | 3.3 GB |
| Phi-4-mini | Microsoft | 3.80B | 128K | MIT License | 2.5 GB |

All four are **dense** (not MoE) models and all four allow personal local use.

---

## Screenshots

The `screenshots/` folder is intentionally **empty** — no screenshots were
fabricated. You take them manually. Useful things to capture:

1. `python3 estimator.py` output (the memory tables).
2. `python3 comparison.py` output (the two experiments + model comparison).
3. Official web pages (open these in a browser and screenshot them):
   - `huggingface.co/meta-llama/Llama-3.2-3B`
   - `huggingface.co/Qwen/Qwen3-4B`
   - `huggingface.co/google/gemma-3-4b-it`
   - `huggingface.co/microsoft/Phi-4-mini-instruct`
   - `ollama.com/library/llama3.2:3b`, `qwen3:4b`, `gemma3:4b`, `phi4-mini`

Optional, only if Ollama is runnable for your user (it may print "Permission
denied"): `ollama list` and `ollama ps` show what models are already present and
running. **Do not** run `ollama pull` or `ollama run` to create a screenshot. On
this particular machine the `ollama` binary is not executable by the user account,
so no `ollama list` / `ollama ps` output could be produced — this is documented
honestly in `analysis.md` (Section 11).

---

## Key takeaway

Based on the estimated memory usage, **Qwen3 4B at Q4 with an 8K context** is the
best fit for the scenario (~3.68 GB estimated RAM, Apache 2.0 licence, strong tool
calling). **Phi-4-mini at Q4** is a close runner-up (~3.45 GB, MIT licence). These
are calculations and web facts only — no model was actually run locally. See
`analysis.md` for the full reasoning.
