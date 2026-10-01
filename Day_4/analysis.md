# Day 4 Analysis — Local LLM Memory Estimation on a CPU Laptop

> **No model was downloaded or executed locally for this assignment.**
> All model information (parameters, context window, licence, tool calling, Ollama
> size) was taken from official web sources: Hugging Face model cards and the
> official Ollama library pages. The memory numbers are calculations, not
> measurements from a running model.

---

## 1. Scenario

The machine in this scenario is a **Linux Mint laptop with about 15 GB of RAM** and
**no dedicated GPU**, so any local LLM would run on the CPU. The purpose is a
**personal chatbot / coding helper** — something I would use for myself, not a
product for other people.

It is important to be clear about what this project is. It is **not** downloading
or running these models. It is a study of whether models **could** fit on this kind
of machine, using a simple memory formula and public information about the models.
Nothing was pulled with `ollama pull`, nothing was started with `ollama run`, and
no existing models were removed or changed.

For the estimate, I assume the laptop has 15 GB of RAM, but about **3 GB is
reserved** for the operating system, the desktop, and a browser. That leaves an
**available memory budget of about 12 GB** for the model. This 12 GB figure is the
number that every "fits / does not fit" decision in this project is compared against.

The simple memory formula I used throughout is:

```
Weights    = parameters × bytes_per_weight
KV cache   = 2 × layers × kv_heads × head_dim × context_tokens × 2 bytes
Overhead   = 0.5 GB runtime buffer + 10% of the weights
Total      = Weights + KV cache + Overhead
```

If `Total ≤ 12 GB` the configuration **fits**; otherwise it **does not fit**.

---

## 2. Model Weights

The "weights" are the numbers that the model has learned — every parameter in the
network. Each parameter is just one number, and the memory it takes depends on how
many bits that number is stored in.

The formula is simply:

```
Weights memory = parameter count × bytes per parameter
```

So two things decide the weight memory: **how many parameters the model has**, and
**what precision those parameters are stored in**.

For example, the Qwen3 4B model has about 4.02 **billion** parameters. If each
parameter is stored in FP16 (2 bytes), the weights need about 8 GB. If each
parameter is stored in Q4 (about 0.5 bytes), the same model needs only about 2 GB.
The parameter count never changed — only the number of bytes per parameter. That
is the whole idea behind quantization.

The relationship is linear. Doubling the parameters roughly doubles the weight
memory. Halving the precision roughly halves it. This is why a 3B model is
comfortable on a laptop but a 30B model in FP16 is not: 30 billion parameters ×
2 bytes is about 60 GB, which no ordinary laptop has.

---

## 3. Quantization

Quantization means storing each parameter using fewer bits than the original full
precision. The table below shows the bytes per parameter I used.

| Precision | Bytes per parameter | Bits | Comment |
|---|---|---|---|
| FP32 | 4.0 | 32 | Full precision, far too big for a laptop |
| FP16 | 2.0 | 16 | The standard "original" release precision |
| Q8 | 1.0 | 8 | Almost no quality loss |
| Q6 | 0.65 | ~6 | Middle ground |
| Q4 | 0.50 | 4 | The usual default on laptops (Q4_K_M) |

Quantization **primarily reduces weight memory**. That is its main job. When a
model does not fit in FP16, dropping to Q4 is usually the easiest way to make it
fit, because it cuts the largest part of the memory bill.

There is a trade-off. Fewer bits means each parameter carries less detail, so the
model can become slightly less accurate or slightly less coherent. Q8 is nearly as
good as FP16 and still saves half the weight memory. Q4 saves the most memory but
loses the most quality. For a personal chatbot or coding helper, Q4 is normally the
sweet spot on a 15 GB laptop.

One important detail: the **KV cache is not reduced by quantization**. Even when
the weights are stored as Q4, the KV cache is normally kept in FP16. So quantization
shrinks the weights but leaves the KV cache as it is. I explain the KV cache in the
next section.

---

## 4. KV Cache and Context Length

The KV cache is the memory that grows while the model is generating a reply.

A transformer reads the prompt one token at a time. For each token it calculates a
"Key" and a "Value" for every layer, and it **keeps** them so it does not have to
recalculate them later. That stored collection is the KV cache.

```
KV cache = 2 × layers × kv_heads × head_dim × context_tokens × 2 bytes
```

- The **2** at the start is because we store both Keys and Values.
- `context_tokens` is the length of the conversation being processed.

The key insight is that the KV cache **grows with the context window**. If I allow
an 8K context, the model may cache up to 8,192 tokens. If I allow a 16K context, it
may cache up to 16,384 tokens — twice as many. So the KV cache doubles.

This is different from the weights. The **weights are always the same size**,
no matter how long the context is. Only the KV cache changes with context length.
That is exactly what Experiment 1 in `comparison.py` demonstrates: as context grows
from 2K to 16K, the Weights column stays fixed while the KV Cache and Total columns
grow.

Modern models use **Grouped-Query Attention (GQA)**, where several query heads share
one Key/Value head. This is why the `kv_heads` number is smaller than the attention
head count — it keeps the KV cache from becoming enormous.

---

## 5. Model Cards

I looked at four model families from different publishers. For each one I recorded
the facts from the official Hugging Face model card and the official Ollama
library page. **Date checked: 1 October 2026.**

All the sources I used were:

- Hugging Face model card: `meta-llama/Llama-3.2-3B`
- Hugging Face model card: `Qwen/Qwen3-4B`
- Hugging Face model card: `google/gemma-3-4b-it`
- Hugging Face model card: `microsoft/Phi-4-mini-instruct`
- Official Ollama library pages: `ollama.com/library/llama3.2:3b`,
  `ollama.com/library/qwen3:4b`, `ollama.com/library/gemma3:4b`,
  `ollama.com/library/phi4-mini`

All four are **dense** models (not Mixture-of-Experts). None of the four is MoE, so
there is no separate "active parameters" figure — every parameter is used. (For
example, the wider Qwen3 family includes MoE models like Qwen3-30B-A3B where only a
few billion of the 30B parameters are active per token, but the Qwen3-4B I compared
is a normal dense model.)

---

## 6. Open-Weight vs Open-Source

This is an important distinction that is often confused.

**Open-weight** means the trained model numbers (the weights) are published so that
anyone can download and run them. All four models here are open-weight.

**Open-source** is a stronger idea. It means the model is released under a standard
open-source software licence, recognised by the Open Source Initiative (OSI). A
proper open-source licence gives broad rights to use, study, modify and redistribute
the software, with no field-of-use restrictions.

Being **downloadable does not automatically mean open source**. A publisher can give
you the weights but attach a custom licence with extra conditions. That is the case
for Llama and Gemma.

Here is how the four licences compare:

| Model | Exact licence | Open-weight? | Standard OSI open-source licence? |
|---|---|---|---|
| Phi-4-mini-instruct | MIT License | Yes | **Yes** — MIT is an OSI licence |
| Qwen3 4B | Apache License 2.0 | Yes | **Yes** — Apache 2.0 is an OSI licence |
| Llama 3.2 3B | Llama 3.2 Community License | Yes | **No** — custom licence with extra conditions |
| Gemma 3 4B | Gemma Terms of Use | Yes | **No** — custom terms with a use policy |

The important conditions:

- **Llama 3.2 Community License** — commercial use is allowed, but you must
  prominently display **"Built with Llama"**, you must include the licence with any
  redistribution, and if your product has more than 700 million monthly active users
  you must request a separate licence from Meta. It also comes with an Acceptable
  Use Policy that restricts certain uses.

- **Gemma Terms of Use** — commercial use is allowed, but usage is governed by a
  **Gemma Prohibited Use Policy** that bans certain applications, and you must
  accept the terms on Hugging Face before you can download the weights.

- **Apache 2.0 (Qwen3)** — a standard permissive licence. Commercial use, modification
  and redistribution are allowed with normal attribution and notice requirements.

- **MIT (Phi-4-mini)** — the shortest and most permissive standard licence. Commercial
  use, modification and redistribution are allowed with just an attribution notice.

For my personal, non-commercial local chatbot, **all four licences allow the use I
have in mind**. But if I ever wanted to build a product and share it, the MIT and
Apache 2.0 models are the cleanest, because they carry no user-count limits and no
"built with" branding requirement.

---

## 7. Memory Estimate Table

This is the main output of `estimator.py`. It uses the 12 GB available-memory
budget described in Section 1.

| Model | Parameters | Quantization | Context | Weights | KV Cache | Total | Fits? |
|---|---|---|---|---|---|---|---|
| Llama 3.2 3B | 3.21B | Q4 | 8K | 1.49 GB | 0.88 GB | 3.02 GB | Yes |
| Llama 3.2 3B | 3.21B | FP16 | 8K | 5.98 GB | 0.88 GB | 7.95 GB | Yes |
| Qwen3 4B | 4.02B | Q4 | 8K | 1.87 GB | 1.12 GB | 3.68 GB | Yes |
| Qwen3 4B | 4.02B | FP16 | 8K | 7.49 GB | 1.12 GB | 9.86 GB | Yes |
| Gemma 3 4B | 4.30B | Q4 | 8K | 2.00 GB | 1.06 GB | 3.77 GB | Yes |
| Phi-4-mini | 3.80B | Q4 | 8K | 1.77 GB | 1.00 GB | 3.45 GB | Yes |
| Qwen3 4B | 4.02B | Q4 | 32K | 1.87 GB | 4.50 GB | 7.06 GB | Yes |
| Qwen3 4B | 4.02B | FP16 | 32K | 7.49 GB | 4.50 GB | 13.24 GB | **NO** |
| Gemma 3 4B | 4.30B | FP32 | 8K | 16.02 GB | 1.06 GB | 19.18 GB | **NO** |

This table shows the three ideas working together:

- Small models at **Q4** (the green rows) are very comfortable, needing only about
  3–4 GB.
- The same models at **FP16** need roughly twice as much weight memory. Qwen3 4B in
  FP16 at 8K (9.86 GB) still fits, but only just.
- **Qwen3 4B in FP16 at 32K does not fit** (13.24 GB > 12 GB), because the big
  FP16 weights plus a 32K KV cache together pass the budget.
- **Gemma 3 4B in FP32 is far too big** (19.18 GB). FP32 is never used for local
  models on a laptop.

---

## 8. Three-Model Comparison Table

I compared four model families using **only** web sources. **Date checked:
1 October 2026.** No model was downloaded.

| | Llama 3.2 3B Instruct | Qwen3 4B | Gemma 3 4B IT | Phi-4-mini-instruct |
|---|---|---|---|---|
| Publisher | Meta | Alibaba (Qwen Team) | Google | Microsoft |
| HF model ID | `meta-llama/Llama-3.2-3B-Instruct` | `Qwen/Qwen3-4B` | `google/gemma-3-4b-it` | `microsoft/Phi-4-mini-instruct` |
| Total parameters | 3.21B | 4.02B | 4.30B | 3.80B |
| Active parameters (MoE) | N/A — dense | N/A — dense | N/A — dense | N/A — dense |
| Is it MoE? | No | No | No | No |
| Context window | 128K (8K quantized) | 32K native, 131K w/ YaRN | 128K input, 8K output | 128K |
| Exact licence | Llama 3.2 Community License | Apache License 2.0 | Gemma Terms of Use | MIT License |
| Commercial use allowed? | Yes, with conditions | Yes | Yes, with a use policy | Yes |
| Extra licence conditions | "Built with Llama" notice; >700M MAU needs separate Meta licence; Acceptable Use Policy | Standard Apache 2.0 terms only | Gemma Prohibited Use Policy; must accept terms on HF | Standard MIT terms only |
| Tool calling on model card? | Yes (BFCL tool-use score reported) | Yes (card says it excels at tool calling) | No tool-calling section; chat/vision model | Yes (function-calling format documented) |
| GGUF / Ollama build? | Yes — `llama3.2:3b` | Yes — `qwen3:4b` | Yes — `gemma3:4b` | Yes — `phi4-mini` |
| Published Q4 size | 2.0 GB (Q4_K_M) | 2.5 GB (Q4_K_M) | 3.3 GB (Q4_K_M) | 2.5 GB |
| My memory estimate (Q4, 8K) | 3.02 GB total | 3.68 GB total | 3.77 GB total | 3.45 GB total |
| Theoretically fits scenario? | Yes | Yes | Yes | Yes |
| Date checked | 1 Oct 2026 | 1 Oct 2026 | 1 Oct 2026 | 1 Oct 2026 |

A few observations:

- All four are dense 3–4B models, so their memory needs are similar (roughly 3–4 GB
  at Q4). Size is not the main difference between them.
- **Gemma 3 4B has the largest download** (3.3 GB) partly because it is multimodal
  and includes a vision tower, which adds parameters beyond the language model.
- **Licence simplicity differs a lot.** Phi-4-mini (MIT) and Qwen3 (Apache 2.0) are
  the cleanest. Llama and Gemma add conditions.
- **Tool calling** is stated on the model cards for Llama 3.2, Qwen3 and Phi-4-mini.
  The Gemma 3 4B card does not present a tool-calling feature.

---

## 9. Context Length Experiment

In `comparison.py`, Experiment 1, I fixed the model (**Qwen3 4B**) and the
quantization (**Q4**) and changed only the context length.

| Context | Weights | KV Cache | Total | Fits? |
|---|---|---|---|---|
| 2K | 1.87 GB | 0.28 GB | 2.84 GB | Yes |
| 4K | 1.87 GB | 0.56 GB | 3.12 GB | Yes |
| 8K | 1.87 GB | 1.12 GB | 3.68 GB | Yes |
| 16K | 1.87 GB | 2.25 GB | 4.81 GB | Yes |

The results clearly show the three claims:

- **Weights remain approximately constant** at 1.87 GB for every context length.
  Context length does not change how big the model itself is.
- **KV cache increases steadily** as context grows: 0.28 GB → 0.56 GB → 1.12 GB →
  2.25 GB. Each doubling of context roughly doubles the cache.
- **Total memory increases** as a result: 2.84 GB → 4.81 GB. Almost all of that
  growth comes from the KV cache.

If I ever needed more memory, reducing the context window is the easiest saving,
because it only shrinks the KV cache and leaves the weights untouched.

---

## 10. Quantization Experiment

In `comparison.py`, Experiment 2, I fixed the model (**Qwen3 4B**) and the context
(**8K**) and changed only the precision.

| Precision | Weights | KV Cache | Total | Fits? |
|---|---|---|---|---|
| FP16 | 7.49 GB | 1.12 GB | 9.86 GB | Yes |
| Q8 | 3.74 GB | 1.12 GB | 5.74 GB | Yes |
| Q4 | 1.87 GB | 1.12 GB | 3.68 GB | Yes |

This shows that quantization **primarily reduces weight memory**:

- Weights drop from 7.49 GB (FP16) to 1.87 GB (Q4) — a 4× reduction.
- The KV cache stays at 1.12 GB in all three rows, because the KV cache is normally
  kept in FP16 regardless of weight quantization.
- Total drops from 9.86 GB to 3.68 GB.

At 8K context, all three precisions happen to fit in the 12 GB budget. But the
saving is what makes the difference in practice: a model at FP16 uses 2.7× more
memory than the same model at Q4. On a smaller machine, or at a longer context,
FP16 would not fit while Q4 still would. For example, in the main table, Qwen3 4B
in FP16 at 32K **does not fit** (13.24 GB) while the same model in Q4 at 32K fits
comfortably (7.06 GB). That is quantization letting a model fit into a smaller
memory budget. The cost is a small reduction in quality.

---

## 11. Estimate vs Available Web Information

> **No model was downloaded or executed locally.**

Since I was not allowed to download or run a model, I cannot show a real measured
memory reading. Instead, this section compares my **theoretical estimate** with the
**published model/Ollama size** that is available on the web.

| Model | My Q4 weights estimate | Published Ollama Q4_K_M size | Difference |
|---|---|---|---|
| Llama 3.2 3B | 1.49 GB | 2.0 GB | −0.51 GB |
| Qwen3 4B | 1.87 GB | 2.5 GB | −0.63 GB |
| Gemma 3 4B | 2.00 GB | 3.3 GB | −1.30 GB |
| Phi-4-mini | 1.77 GB | 2.5 GB | −0.73 GB |

My estimates are in the same range as the published sizes but consistently a little
lower. That is expected, and here is why:

1. **Quantization overhead.** My formula uses a clean 0.5 bytes per parameter for
   Q4. The real Q4_K_M format is not exactly 4 bits per weight — it also stores
   small scale and metadata values used to de-quantize the weights, plus the
   embedding and output layers are often kept at higher precision. This makes the
   real file bigger than my simple estimate. Gemma 3 shows the biggest gap partly
   because its vision tower is not just a language model.

2. **Runtime overhead is real but approximate.** The running process uses more than
   just the weights and the KV cache. There are buffers for computation, the
   tokenizer, and internal allocations. I added a rough allowance, but a real
   runtime number depends on the specific version of Ollama or llama.cpp.

3. **Architecture differences.** Models use Grouped-Query Attention differently and
   some share embeddings between layers. Two models with the same parameter count
   can therefore have different KV caches and different real memory footprints.

4. **Context settings at runtime.** My KV cache estimate assumes the cache grows to
   the full allowed context. In practice the cache grows as the conversation fills
   up, so a short chat uses much less than the maximum.

5. **It is a formula, not a prediction.** The formula is a simplified model that
   captures the main relationships. It is designed to answer "roughly could this
   fit?", not "exactly how many megabytes will the process use?".

So the honest conclusion is: **the estimate is close enough to be useful for
deciding whether a model is in the right ballpark, but it is not an exact
prediction.** A model that clearly fits with a good margin (like a 3B model at Q4
needing ~3 GB on a 12 GB budget) is a safe bet. A model that only *just* fits should
be treated with caution.

### About the existing Ollama environment

I checked whether Ollama could report anything without downloading or changing
anything. On this machine the `ollama` binary is not readable/executable by my
user account (it returns "Permission denied"), so I could **not** run `ollama list`
or `ollama ps` at all. I have therefore **not reported any `ollama list` or
`ollama ps` output**, because I have none and I will not invent any.

I also did **not** attempt `ollama pull`, `ollama run`, or `ollama rm`, and I did not
modify any Ollama files. Because no local runtime comparison was possible, this
document compares the formula's estimate against the published web sizes only.

---

## 12. Suitability Analysis

Based on the estimated memory usage, the most suitable configuration for this
hypothetical scenario — a personal local chatbot / coding helper on a Linux Mint
laptop with ~15 GB RAM, CPU-only — is:

### Recommended: Qwen3 4B at Q4 with an 8K context

| Property | Value |
|---|---|
| Model | Qwen3 4B (Qwen/Qwen3-4B) |
| Size | 4.02B parameters, dense |
| Quantization | Q4 (Q4_K_M) |
| Context length | 8K |
| Licence | Apache License 2.0 |
| Estimated memory | ~3.68 GB total (1.87 GB weights + 1.12 GB KV cache + 0.69 GB overhead) |

Why this one:

- It fits with a large margin (3.68 GB out of a 12 GB budget), so there is plenty of
  room left for the operating system and other applications.
- The **Apache 2.0 licence** is a clean, standard open-source licence with no
  user-count limits and no branding requirement, which is the most convenient if
  the project ever grows.
- The model card states strong **tool-calling** support, which is useful for a
  coding helper.
- Its 32K native context gives room to grow later, and Qwen3-4B stays the smallest
  of the 4B-class options at Q4.

### Runner-up: Phi-4-mini-instruct at Q4

Phi-4-mini comes out very close, at an estimated ~3.45 GB total, and it actually has
the **simplest licence of all (MIT)** and a documented 128K context. The reason it
is the runner-up rather than the top choice is small but real: the Qwen3 4B card
explicitly highlights tool-calling ability, which is directly useful for a coding
helper, while Phi-4-mini is a general reasoning model. Both would work well; the
choice between them is close.

The other two are also usable but score slightly lower for this scenario:

- **Llama 3.2 3B** is the smallest and lightest (2.0 GB download), but it uses the
  **Llama 3.2 Community License** with its "Built with Llama" requirement and the
  700-million-MAU clause, which is less convenient than MIT/Apache if the project
  ever becomes something more than personal.
- **Gemma 3 4B** is the largest download (3.3 GB) because it includes a vision
  tower, and its tool-calling is not stated on the model card, so it is a poorer fit
  for a coding-helper use case despite a fine licence for personal use.

I want to be clear that this recommendation is **based on the estimated memory usage
and the published model cards**, not on running the models. I did not test them
locally.

---

## 13. Conclusion

This project set out to answer a practical question: which small language models
could reasonably run on a CPU-only Linux Mint laptop with about 15 GB of RAM, for
use as a personal chatbot or coding helper? It answered the question using a simple
memory formula and official web information, without downloading or running
anything.

The main things I learned are:

1. **Weight memory is decided by parameter count and precision.** A parameter is one
   number; how many bits it takes decides how much RAM it uses. More parameters or
   higher precision means more memory.

2. **Quantization is the main memory lever.** Going from FP16 to Q4 cuts weight
   memory by about 4×. This is what allows a model that would not fit in full
   precision to fit comfortably in quantized form on a small machine. The trade-off
   is a small drop in quality.

3. **Context length affects the KV cache, not the weights.** The weights stay the
   same size no matter the context; only the KV cache grows. This gives a second,
   independent way to save memory if needed.

4. **The KV cache is normally kept in FP16**, so quantization reduces the weights but
   not the cache.

5. **Downloadable does not mean open-source.** All four models are open-weight, but
   only Qwen3 (Apache 2.0) and Phi-4-mini (MIT) use standard OSI open-source
   licences. Llama and Gemma use custom licences with extra conditions.

6. **The estimate is a guide, not a prediction.** My calculated sizes were in the
   right ballpark but a little lower than the published Ollama sizes, because real
   quantization formats and runtime overheads add extra bytes.

For my scenario, based on the estimated memory usage, **Qwen3 4B at Q4 with an 8K
context** is the best fit — around 3.68 GB of estimated RAM, a clean Apache 2.0
licence, and strong tool-calling support — with **Phi-4-mini at Q4** a very close
runner-up. Both would comfortably fit, and neither was actually run; these are
calculations and web facts only.

---

### Sources (all checked 1 October 2026)

- `huggingface.co/meta-llama/Llama-3.2-3B` — model card and licence
- `huggingface.co/Qwen/Qwen3-4B` — model card and Apache 2.0 licence
- `huggingface.co/google/gemma-3-4b-it` — model card and Gemma Terms of Use
- `huggingface.co/microsoft/Phi-4-mini-instruct` — model card and MIT licence
- `ollama.com/library/llama3.2:3b` — Q4_K_M size (2.0 GB)
- `ollama.com/library/qwen3:4b` — Q4_K_M size (2.5 GB)
- `ollama.com/library/gemma3:4b` — Q4_K_M size (3.3 GB)
- `ollama.com/library/phi4-mini` — size (2.5 GB)
