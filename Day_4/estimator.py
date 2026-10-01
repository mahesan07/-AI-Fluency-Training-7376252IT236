"""
Day 4 - Local LLM Memory Estimator (beginner level)

Purpose:
    Estimate how much RAM a local LLM needs on a CPU-only laptop, so we can
    decide if a model COULD fit. Nothing is downloaded here. This script only
    does arithmetic based on published model sizes.

Scenario (hypothetical):
    Linux Mint laptop, ~15 GB RAM, CPU-based local LLM,
    used as a personal chatbot / coding helper.

The memory formula used in this lab:

    Weights    = parameters x bytes_per_weight
    KV cache   = 2 x layers x kv_heads x head_dim x context_tokens x bytes_per_value
    Overhead   = runtime buffer + 10% of weights (a rough allowance)
    Total      = Weights + KV cache + Overhead

    If Total <= available RAM  ->  FITS
    If Total >  available RAM  ->  DOES NOT FIT

All model numbers (parameters, layers, KV heads, head size, Ollama Q4 size,
context window, licence) were read from official sources on the date shown in
analysis.md. No model was downloaded or run locally.
"""

# ---------------------------------------------------------------------------
# Scenario settings
# ---------------------------------------------------------------------------

TOTAL_RAM_GB = 15          # the laptop has about 15 GB of RAM
RESERVED_FOR_OS_GB = 3     # Linux Mint + desktop + browser need some RAM too
AVAILABLE_RAM_GB = TOTAL_RAM_GB - RESERVED_FOR_OS_GB   # 12 GB we can really use

RUNTIME_BASE_GB = 0.5      # the runtime (Ollama/llama.cpp) itself needs some RAM
OVERHEAD_PERCENT = 0.10    # extra ~10% on the weights for buffers/metadata

GB = 1024 ** 3              # 1 GiB in bytes


# ---------------------------------------------------------------------------
# How many bytes each weight takes, based on precision / quantization
# ---------------------------------------------------------------------------
# This is the KEY idea: a parameter is just a number. Fewer bits = less memory.

BYTES_PER_WEIGHT = {
    "FP32": 4.0,    # 32 bits  - rarely used locally, too big
    "FP16": 2.0,    # 16 bits  - the "original" model precision
    "Q8":   1.0,    #  8 bits  - nearly lossless, small quality drop
    "Q6":   0.65,   #  6 bits  - a middle ground
    "Q4":   0.50,   #  4 bits  - the usual default for laptops (Q4_K_M)
}

# KV cache is normally kept in FP16 even when the weights are quantized,
# so each cached value takes 2 bytes.
BYTES_PER_KV_VALUE = 2.0


# ---------------------------------------------------------------------------
# Model information (from official model cards / Ollama library pages)
# ---------------------------------------------------------------------------
# params_b   : total parameters, in billions (from the model card)
# layers     : number of transformer layers
# kv_heads   : number of Key/Value heads (Grouped-Query Attention reduces this)
# head_dim   : size of one attention head
# ollama_q4  : the Q4_K_M download size shown on the official Ollama page
# context    : the context window the model supports
# licence    : exact licence name from the official model card

MODELS = {
    "Llama 3.2 3B": {
        "publisher": "Meta",
        "params_b": 3.21,
        "layers": 28,
        "kv_heads": 8,
        "head_dim": 128,
        "context": 131072,       # 128K context window
        "ollama_q4_gb": 2.0,     # ollama.com/library/llama3.2:3b
        "licence": "Llama 3.2 Community License",
    },
    "Qwen3 4B": {
        "publisher": "Alibaba (Qwen Team)",
        "params_b": 4.02,
        "layers": 36,
        "kv_heads": 8,
        "head_dim": 128,
        "context": 32768,        # 32K native (131K with YaRN)
        "ollama_q4_gb": 2.5,     # ollama.com/library/qwen3:4b
        "licence": "Apache License 2.0",
    },
    "Gemma 3 4B": {
        "publisher": "Google",
        "params_b": 4.30,
        "layers": 34,
        "kv_heads": 4,
        "head_dim": 256,
        "context": 131072,       # 128K context window
        "ollama_q4_gb": 3.3,     # ollama.com/library/gemma3:4b
        "licence": "Gemma Terms of Use",
    },
    "Phi-4-mini": {
        "publisher": "Microsoft",
        "params_b": 3.80,
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "context": 131072,       # 128K context window
        "ollama_q4_gb": 2.5,     # ollama.com/library/phi4-mini
        "licence": "MIT License",
    },
}


# ---------------------------------------------------------------------------
# The estimator functions
# ---------------------------------------------------------------------------

def weight_memory(params_billion, quant):
    """
    Model weights memory.

    weights = parameters x bytes_per_weight

    More parameters OR higher precision = more weight memory.
    Lower quantization (Q4 instead of FP16) = less weight memory.
    """
    bytes_each = BYTES_PER_WEIGHT[quant]
    total_bytes = params_billion * 1_000_000_000 * bytes_each
    return total_bytes / GB


def kv_cache_memory(model, context_tokens):
    """
    KV cache memory.

    KV cache = 2 x layers x kv_heads x head_dim x context_tokens x bytes_per_value

    Why 2?  Because we store both the Keys and the Values.
    Why does context matter?  The cache grows with the number of tokens,
        so a longer context window uses more RAM for the cache.
    """
    m = MODELS[model]
    total_bytes = (
        2
        * m["layers"]
        * m["kv_heads"]
        * m["head_dim"]
        * context_tokens
        * BYTES_PER_KV_VALUE
    )
    return total_bytes / GB


def runtime_overhead(weights_gb):
    """A rough runtime allowance: a small fixed buffer + a % of the weights."""
    return RUNTIME_BASE_GB + OVERHEAD_PERCENT * weights_gb


def estimate(model, quant, context_tokens):
    """
    Put the whole estimate together for one model configuration.

    Returns a small dictionary with weights, KV cache, overhead, total, and
    whether it fits in the available RAM.
    """
    m = MODELS[model]
    weights = weight_memory(m["params_b"], quant)
    kv_cache = kv_cache_memory(model, context_tokens)
    overhead = runtime_overhead(weights)
    total = weights + kv_cache + overhead
    fits = "Yes" if total <= AVAILABLE_RAM_GB else "NO"

    return {
        "model": model,
        "params_b": m["params_b"],
        "quant": quant,
        "context_tokens": context_tokens,
        "weights_gb": weights,
        "kv_cache_gb": kv_cache,
        "overhead_gb": overhead,
        "total_gb": total,
        "fits": fits,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_estimate_table(results):
    """Print a clean text table of estimates (good for a screenshot)."""
    header = (
        f"{'Model':<16}{'Params':>8}{'Quant':>7}{'Context':>9}"
        f"{'Weights':>9}{'KV Cache':>10}{'Overhead':>10}{'Total':>8}{'Fits?':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<16}"
            f"{r['params_b']:>7.2f}B"
            f"{r['quant']:>7}"
            f"{r['context_tokens'] // 1024:>7}K"
            f"{r['weights_gb']:>9.2f}"
            f"{r['kv_cache_gb']:>10.2f}"
            f"{r['overhead_gb']:>10.2f}"
            f"{r['total_gb']:>8.2f}"
            f"{r['fits']:>7}"
        )


def print_info():
    """Print the scenario and a short explanation before the table."""
    print("=" * 84)
    print("DAY 4 - LOCAL LLM MEMORY ESTIMATOR")
    print("=" * 84)
    print()
    print("Scenario:")
    print(f"  Machine              : Linux Mint laptop, ~{TOTAL_RAM_GB} GB RAM, CPU only")
    print(f"  Reserved for OS      : {RESERVED_FOR_OS_GB} GB")
    print(f"  Available for the LLM: {AVAILABLE_RAM_GB} GB")
    print(f"  Purpose              : personal local chatbot / coding helper")
    print()
    print("NOTE: No model was downloaded or run locally. This is a calculation only.")
    print("      All model facts come from official model cards and Ollama pages.")
    print()
    print("Formula:")
    print("  Weights  = parameters x bytes_per_weight")
    print("  KV cache = 2 x layers x kv_heads x head_dim x context x 2 bytes")
    print("  Total    = Weights + KV cache + Runtime overhead")
    print()


# ---------------------------------------------------------------------------
# The configurations we will show (at least four realistic choices)
# ---------------------------------------------------------------------------

CONFIGS = [
    ("Llama 3.2 3B", "Q4",  8192),    # the usual default for a laptop
    ("Llama 3.2 3B", "FP16", 8192),   # same model, unquantized (still fits here)
    ("Qwen3 4B",      "Q4",  8192),
    ("Qwen3 4B",      "FP16", 8192),  # unquantized Qwen3 at 8K (fits, but heavy)
    ("Gemma 3 4B",    "Q4",  8192),
    ("Phi-4-mini",    "Q4",  8192),
    ("Qwen3 4B",      "Q4",  32768),  # Qwen3 at its full 32K context
    ("Qwen3 4B",      "FP16", 32768),  # unquantized AND full context -> too big
    ("Gemma 3 4B",    "FP32", 8192),   # FP32 is far too big for a laptop
]


def main():
    print_info()

    print("STEP 1 - Bytes per parameter for each precision")
    print("-" * 84)
    for quant in ["FP16", "Q8", "Q4"]:
        print(f"  {quant:<5} -> {BYTES_PER_WEIGHT[quant]:>4.2f} bytes per parameter")
    print("  (Quantization lowers bytes per parameter, so weights use less RAM.)")
    print()

    print("STEP 2 - Memory estimate for several realistic configurations")
    print()
    results = []
    for model, quant, ctx in CONFIGS:
        results.append(estimate(model, quant, ctx))
    print_estimate_table(results)
    print()
    print(f"  All memory columns are in GB. 'Fits?' is Yes when Total <= {AVAILABLE_RAM_GB} GB.")
    print()

    print("STEP 3 - One model, different precisions, to see quantization's effect")
    print("  (Llama 3.2 3B at 8K context)")
    print()
    quant_results = []
    for quant in ["FP16", "Q8", "Q4"]:
        quant_results.append(estimate("Llama 3.2 3B", quant, 8192))
    print_estimate_table(quant_results)
    print()
    print("  Watch the Weights column drop as we quantize. This is the main")
    print("  reason a model that does not fit in FP16 can fit as Q4.")
    print()

    print("STEP 4 - One model, growing context, to see the KV cache grow")
    print("  (Qwen3 4B at Q4)")
    print()
    ctx_results = []
    for ctx in [2048, 4096, 8192, 16384, 32768]:
        ctx_results.append(estimate("Qwen3 4B", "Q4", ctx))
    print_estimate_table(ctx_results)
    print()
    print("  Watch the Weights column stay the same, but KV Cache and Total")
    print("  grow as the context window gets bigger.")
    print()

    print("STEP 5 - Compare our Q4 estimate to the published Ollama Q4 size")
    print()
    header = f"{'Model':<16}{'Our Q4 estimate':>18}{'Ollama Q4_K_M size':>22}{'Difference':>13}"
    print(header)
    print("-" * len(header))
    for name in MODELS:
        est = estimate(name, "Q4", 8192)
        ollama_size = MODELS[name]["ollama_q4_gb"]
        diff = est["weights_gb"] - ollama_size
        print(
            f"{name:<16}"
            f"{est['weights_gb']:>17.2f} GB"
            f"{ollama_size:>21.1f} GB"
            f"{diff:>12.2f} GB"
        )
    print()
    print("  Our estimate is a simple calculation; the Ollama number is the real")
    print("  downloaded file size. They should be close but not identical, because")
    print("  Q4_K_M stores a few extra scale values with the weights.")
    print()

    print("=" * 84)
    print("SUMMARY")
    print("=" * 84)
    print("  - Parameter count and precision together decide the Weights memory.")
    print("  - Quantization (FP16 -> Q8 -> Q4) lowers bytes per parameter, so it")
    print("    mainly reduces Weights memory.")
    print("  - A longer context window increases the KV cache, so Total memory")
    print("    grows even though the Weights stay the same.")
    print("  - On this ~15 GB laptop, small models (3B-4B) at Q4 fit comfortably,")
    print("    but the same models in FP16, or at very long contexts, may not.")
    print("  - No model was downloaded or run. This is an estimate only.")
    print()


if __name__ == "__main__":
    main()
