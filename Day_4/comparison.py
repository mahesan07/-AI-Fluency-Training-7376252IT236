"""
Day 4 - Comparison experiments (beginner level)

This script reuses the estimator functions from estimator.py and runs three
experiments:

    EXPERIMENT 1 - Context length experiment
        Keep the model and the quantization fixed, change only the context
        length (2K, 4K, 8K, 16K). This shows that the Weights stay the same
        while the KV cache and the Total grow.

    EXPERIMENT 2 - Quantization experiment
        Keep the model and the context fixed, change only the precision
        (FP16, Q8, Q4). This shows that quantization mainly reduces the
        Weights memory, which is what lets a model fit in a smaller budget.

    EXPERIMENT 3 - Three-model comparison
        Compare four model families using ONLY official web information
        (model cards, licence pages, Ollama library pages). No model is
        downloaded or run.

IMPORTANT: No model was downloaded, started, or modified for this assignment.
"""

import estimator as est

GB = est.GB
AVAILABLE = est.AVAILABLE_RAM_GB


def print_table(results, title):
    """Print one comparison table."""
    print(title)
    print("-" * 84)
    header = (
        f"{'Model':<16}{'Quant':>6}{'Context':>9}"
        f"{'Weights':>9}{'KV Cache':>10}{'Overhead':>10}{'Total':>8}{'Fits?':>7}"
    )
    print(header)
    print("-" * 84)
    for r in results:
        print(
            f"{r['model']:<16}"
            f"{r['quant']:>6}"
            f"{r['context_tokens'] // 1024:>7}K"
            f"{r['weights_gb']:>9.2f}"
            f"{r['kv_cache_gb']:>10.2f}"
            f"{r['overhead_gb']:>10.2f}"
            f"{r['total_gb']:>8.2f}"
            f"{r['fits']:>7}"
        )
    print("-" * 84)
    print("  (memory columns are in GB;  Fits? = Yes when Total <= "
          f"{AVAILABLE} GB)")
    print()


# ---------------------------------------------------------------------------
# EXPERIMENT 1 - Context length
# ---------------------------------------------------------------------------

def experiment_context():
    """Same model, same quantization, growing context length."""
    print("=" * 84)
    print("EXPERIMENT 1 - CONTEXT LENGTH")
    print("=" * 84)
    print()
    print("Setup: keep the model (Qwen3 4B) and the quantization (Q4) fixed.")
    print("       Only the context length changes: 2K, 4K, 8K, 16K.")
    print()

    results = []
    for ctx in [2048, 4096, 8192, 16384]:
        results.append(est.estimate("Qwen3 4B", "Q4", ctx))

    print_table(results, "  Context length results (Qwen3 4B, Q4):")

    first = results[0]
    last = results[-1]

    print("What this shows:")
    print(f"  - Weights stayed the same at {first['weights_gb']:.2f} GB "
          f"for every context length.")
    print("    Context length does NOT change the size of the model weights.")
    print(f"  - KV cache grew from {first['kv_cache_gb']:.2f} GB (2K) to "
          f"{last['kv_cache_gb']:.2f} GB (16K).")
    print("    That is because the cache stores one Key and one Value per token.")
    print(f"  - Total grew from {first['total_gb']:.2f} GB to "
          f"{last['total_gb']:.2f} GB, almost entirely because of the KV cache.")
    if last["fits"] == "NO":
        print(f"  - At 16K context this configuration would NOT fit in "
              f"{AVAILABLE} GB of available RAM.")
    else:
        print(f"  - Even at 16K context this configuration still fits in "
              f"{AVAILABLE} GB.")
    print("  - Conclusion: if memory is tight, a smaller context window is an")
    print("    easy way to save RAM, because it only shrinks the KV cache.")
    print()


# ---------------------------------------------------------------------------
# EXPERIMENT 2 - Quantization
# ---------------------------------------------------------------------------

def experiment_quantization():
    """Same model, same context, changing precision."""
    print("=" * 84)
    print("EXPERIMENT 2 - QUANTIZATION")
    print("=" * 84)
    print()
    print("Setup: keep the model (Qwen3 4B) and the context (8K) fixed.")
    print("       Only the precision changes: FP16, Q8, Q4.")
    print()

    results = []
    for quant in ["FP16", "Q8", "Q4"]:
        results.append(est.estimate("Qwen3 4B", quant, 8192))

    print_table(results, "  Quantization results (Qwen3 4B, 8K context):")

    fp16 = results[0]
    q8 = results[1]
    q4 = results[-1]

    print("What this shows:")
    print(f"  - Weights dropped from {fp16['weights_gb']:.2f} GB (FP16) to "
          f"{q4['weights_gb']:.2f} GB (Q4).")
    print("    Quantization mainly reduces the memory used by the weights.")
    print(f"  - KV cache stayed at {fp16['kv_cache_gb']:.2f} GB in all three rows.")
    print("    That is expected: the KV cache is normally kept in FP16 even when")
    print("    the weights are quantized.")
    print(f"  - Total dropped from {fp16['total_gb']:.2f} GB to "
          f"{q4['total_gb']:.2f} GB.")
    print(f"  - Fit result: FP16 = {fp16['fits']}, Q8 = {q8['fits']}, "
          f"Q4 = {q4['fits']}.")
    if fp16["fits"] == "NO" and q4["fits"] == "Yes":
        print("    This is the important point: the SAME model does not fit in")
        print(f"    FP16 but does fit in Q4, because quantization cut the weights")
        print(f"    from {fp16['weights_gb']:.2f} GB down to {q4['weights_gb']:.2f} GB.")
    print("  - The trade-off: lower quantization saves RAM but can reduce answer")
    print("    quality, because the weights keep less detail.")
    print()


# ---------------------------------------------------------------------------
# EXPERIMENT 3 - Three (four) model families, from web sources only
# ---------------------------------------------------------------------------

# Everything below was read from official pages on the date in analysis.md.
# Fields match the official model cards and Ollama library pages.

WEB_FACTS = [
    {
        "name": "Llama 3.2 3B Instruct",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "publisher": "Meta",
        "params": "3.21B",
        "moe": "No (dense)",
        "context": "128K (8K for the quantized release)",
        "licence": "Llama 3.2 Community License",
        "commercial": "Yes, with conditions",
        "extra": "'Built with Llama' notice; >700M MAU needs a separate Meta license",
        "tools": "Yes (BFCL tool-use score reported on the model card)",
        "gguf": "Yes - ollama.com/library/llama3.2:3b",
        "q4_size": "2.0 GB (Q4_K_M)",
        "est_q4": est.estimate("Llama 3.2 3B", "Q4", 8192),
        "fits": "Yes (estimated)",
    },
    {
        "name": "Qwen3 4B",
        "hf_id": "Qwen/Qwen3-4B",
        "publisher": "Alibaba (Qwen Team)",
        "params": "4.02B",
        "moe": "No (dense; the Qwen3 family also has MoE models)",
        "context": "32K native, up to 131K with YaRN",
        "licence": "Apache License 2.0",
        "commercial": "Yes",
        "extra": "None beyond the standard Apache 2.0 terms",
        "tools": "Yes (model card says it excels at tool calling)",
        "gguf": "Yes - ollama.com/library/qwen3:4b",
        "q4_size": "2.5 GB (Q4_K_M)",
        "est_q4": est.estimate("Qwen3 4B", "Q4", 8192),
        "fits": "Yes (estimated)",
    },
    {
        "name": "Gemma 3 4B IT",
        "hf_id": "google/gemma-3-4b-it",
        "publisher": "Google",
        "params": "4.30B",
        "moe": "No (dense)",
        "context": "128K input, 8K output",
        "licence": "Gemma Terms of Use",
        "commercial": "Yes, with a use policy",
        "extra": "Gemma Prohibited Use Policy; must accept terms on Hugging Face",
        "tools": "No tool-calling section on the model card; it is a chat/vision model",
        "gguf": "Yes - ollama.com/library/gemma3:4b",
        "q4_size": "3.3 GB (Q4_K_M)",
        "est_q4": est.estimate("Gemma 3 4B", "Q4", 8192),
        "fits": "Yes (estimated)",
    },
    {
        "name": "Phi-4-mini-instruct",
        "hf_id": "microsoft/Phi-4-mini-instruct",
        "publisher": "Microsoft",
        "params": "3.80B",
        "moe": "No (dense decoder-only)",
        "context": "128K",
        "licence": "MIT License",
        "commercial": "Yes",
        "extra": "None beyond the MIT terms",
        "tools": "Yes (function calling supported; card shows a tool-call format)",
        "gguf": "Yes - ollama.com/library/phi4-mini",
        "q4_size": "2.5 GB (Q4_K_M)",
        "est_q4": est.estimate("Phi-4-mini", "Q4", 8192),
        "fits": "Yes (estimated)",
    },
]


def experiment_models():
    """Compare model families using official web information only."""
    print("=" * 84)
    print("EXPERIMENT 3 - THREE (FOUR) MODEL FAMILIES FROM WEB SOURCES")
    print("=" * 84)
    print()
    print("No model was downloaded or run. Every fact below comes from the official")
    print("Hugging Face model card and the official Ollama library page.")
    print()

    for f in WEB_FACTS:
        print(f"  {f['name']}  ({f['hf_id']})")
        print(f"    Publisher           : {f['publisher']}")
        print(f"    Total parameters    : {f['params']}")
        print(f"    MoE?                : {f['moe']}")
        print(f"    Context window      : {f['context']}")
        print(f"    Licence (exact)     : {f['licence']}")
        print(f"    Commercial use      : {f['commercial']}")
        print(f"    Extra conditions    : {f['extra']}")
        print(f"    Tool calling on card: {f['tools']}")
        print(f"    GGUF / Ollama build : {f['gguf']}")
        print(f"    Published Q4 size   : {f['q4_size']}")
        print(f"    Our Q4 estimate @8K : {f['est_q4']['weights_gb']:.2f} GB weights, "
              f"{f['est_q4']['total_gb']:.2f} GB total")
        print(f"    Fits the scenario?  : {f['fits']} "
              f"(available RAM {AVAILABLE} GB)")
        print()

    print("  Summary table")
    header = (
        f"{'Model':<24}{'Licence':<32}{'Q4 size':>10}{'Est total':>11}{'Fits?':>7}"
    )
    print("  " + header)
    print("  " + "-" * (len(header) - 2))
    for f in WEB_FACTS:
        print(
            f"  {f['name']:<24}{f['licence']:<32}"
            f"{f['q4_size'].split()[0]:>10}"
            f"{f['est_q4']['total_gb']:>10.2f}G"
            f"{f['fits'].split()[0]:>7}"
        )
    print()

    print("  What stands out:")
    print("  - All four are dense (not MoE) models of 3-4B parameters, so their")
    print("    memory needs are very similar.")
    print("  - Phi-4-mini and Qwen3 have the simplest licences (MIT and Apache 2.0).")
    print("    Llama and Gemma have custom licences with extra conditions, so")
    print("    'downloadable' does not automatically mean 'open source'.")
    print("  - Gemma 3 4B is the largest download here (3.3 GB) partly because it")
    print("    is multimodal and carries a vision tower.")
    print("  - Based on the estimated memory usage, every one of these fits on a")
    print(f"    ~{est.TOTAL_RAM_GB} GB laptop when quantized to Q4 with a modest context.")
    print()


def experiment_licences():
    """A short, focused look at the licences."""
    print("=" * 84)
    print("EXPERIMENT 3b - LICENCE COMPARISON")
    print("=" * 84)
    print()
    print("  Model              Licence                      OSI-style open source?")
    print("  " + "-" * 68)
    print(f"  {'Phi-4-mini':<18}{'MIT License':<30}Yes (MIT is an OSI licence)")
    print(f"  {'Qwen3 4B':<18}{'Apache 2.0':<30}Yes (Apache 2.0 is an OSI licence)")
    print(f"  {'Llama 3.2 3B':<18}{'Llama 3.2 Community':<30}No (custom licence with use limits)")
    print(f"  {'Gemma 3 4B':<18}{'Gemma Terms of Use':<30}No (custom terms with a use policy)")
    print()
    print("  Key point: all four models are downloadable, and all four give you the")
    print("  weights, so they are all 'open-weight'. But only Phi-4-mini (MIT) and")
    print("  Qwen3 4B (Apache 2.0) use standard OSI open-source licences. Llama and")
    print("  Gemma use custom licences that add conditions a normal open-source")
    print("  licence would not have, so calling them 'open source' is not accurate.")
    print()


def main():
    print()
    print("DAY 4 - COMPARISON EXPERIMENTS")
    print("(No model is downloaded or run. This only compares numbers.)")
    print()

    experiment_context()
    experiment_quantization()
    experiment_models()
    experiment_licences()

    print("=" * 84)
    print("OVERALL SUMMARY")
    print("=" * 84)
    print("  1. Weights memory is set by parameters x precision. It does not change")
    print("     with context length.")
    print("  2. KV cache memory grows with the context window.")
    print("  3. Quantization mainly shrinks the weights, which is the main lever for")
    print("     making a model fit in a small RAM budget.")
    print("  4. All four researched models are small, dense, 3-4B models with official")
    print("     GGUF/Ollama builds and licences that permit personal local use.")
    print("  5. Remember: these are estimates and web facts. Nothing was run locally.")
    print()


if __name__ == "__main__":
    main()
