"""
Follow-up experiment: does the hex-fragmentation pattern hold
  (A) across newer tokenizers, and
  (B) on real design-system palettes, as they're actually written in CSS?

Tokenizers tested:
  - cl100k_base  (GPT-3.5 / GPT-4)            via tiktoken
  - o200k_base   (GPT-4o and later OpenAI)    via tiktoken
  - Qwen2.5      (open model)                 via Hugging Face `tokenizers` (optional)
  - DeepSeek-V3  (open model)                 via Hugging Face `tokenizers` (optional)

Palettes tested (downloaded at run time):
  - Tailwind CSS v3 default palette
  - Bootstrap 5.3 SCSS variables
  - Open Color

Also checks UPPERCASE vs lowercase hex, since real stylesheets
(Tailwind, Bootstrap) mostly use lowercase and the earlier runs used uppercase.

Run:
    pip3 install tiktoken tokenizers requests
    python3 ../experiments/03_tokenizer_and_palette_check.py

Writes tokenizer_and_palette_results.json in the current directory (run from results/).
"""

import json
import re
import statistics

import requests
import tiktoken

# ---------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------
tokenizers = {}

cl100k = tiktoken.get_encoding("cl100k_base")
assert cl100k.encode("hello world!") == [15339, 1917, 0], "cl100k mismatch"
tokenizers["cl100k_base (GPT-4)"] = lambda s: [cl100k.decode([i]) for i in cl100k.encode(s)]

try:
    o200k = tiktoken.get_encoding("o200k_base")
    tokenizers["o200k_base (GPT-4o+)"] = lambda s: [o200k.decode([i]) for i in o200k.encode(s)]
except Exception as e:
    print("Skipping o200k_base:", e)

try:
    from tokenizers import Tokenizer

    def hf_loader(repo):
        tok = Tokenizer.from_pretrained(repo)

        def pieces(s):
            enc = tok.encode(s, add_special_tokens=False)
            return [tok.decode([i]) for i in enc.ids]
        return pieces

    for label, repo in [("Qwen2.5", "Qwen/Qwen2.5-0.5B"),
                        ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3")]:
        try:
            tokenizers[label] = hf_loader(repo)
        except Exception as e:
            print("Skipping {}: {}".format(label, e))
except ImportError:
    print("`tokenizers` not installed - skipping open-model tokenizers "
          "(pip3 install tokenizers to include them)")

print("Tokenizers loaded:", ", ".join(tokenizers))
print()

CONTEXT_PREFIX = "color: "
CONTEXT_SUFFIX = ";"


def count(tok, s):
    return len(tok(s))


def in_context_cost(tok, hex_value):
    """Tokens the hex value adds when written inside a real CSS declaration."""
    with_value = count(tok, CONTEXT_PREFIX + hex_value + CONTEXT_SUFFIX)
    without = count(tok, CONTEXT_PREFIX.rstrip() + CONTEXT_SUFFIX)
    return with_value - without


# ---------------------------------------------------------------------
# (A) The 256 hex-pair sweep, per tokenizer, upper and lower case
# ---------------------------------------------------------------------
print("=" * 70)
print("(A) 256 hex pairs as #hhhhhh, per tokenizer")
print("=" * 70)
sweep = {}
for name, tok in tokenizers.items():
    sweep[name] = {}
    for case in ("upper", "lower"):
        same, diff = [], []
        max_frag = 0
        for i in range(256):
            h = format(i, "02X")
            if case == "lower":
                h = h.lower()
            c = count(tok, "#" + h * 3)
            (same if h[0] == h[1] else diff).append(c)
            if c >= 7:
                max_frag += 1
        sweep[name][case] = {
            "same_char_avg": round(statistics.mean(same), 2),
            "diff_char_avg": round(statistics.mean(diff), 2),
            "pairs_at_7plus_tokens": max_frag,
        }
        r = sweep[name][case]
        print("{:<22} {:<5}  same-char avg {:>5}   diff-char avg {:>5}   7+ tokens: {:>3}/256".format(
            name, case, r["same_char_avg"], r["diff_char_avg"], r["pairs_at_7plus_tokens"]))
print()

# Show a few concrete breakdowns so the pattern is visible, not just averaged
examples = ["#FFFFFF", "#C0C0C0", "#3A7BD5", "#c0c0c0", "#3a7bd5"]
print("Example breakdowns:")
examples_out = {}
for name, tok in tokenizers.items():
    examples_out[name] = {}
    for ex in examples:
        p = tok(ex)
        examples_out[name][ex] = p
        print("  {:<22} {:<9} {} tok  {}".format(name, ex, len(p), p))
print()

# ---------------------------------------------------------------------
# (B) Real design-system palettes
# ---------------------------------------------------------------------
print("=" * 70)
print("(B) Real design-system palettes (hex values as written in source)")
print("=" * 70)
SOURCES = {
    "Tailwind v3": "https://cdn.jsdelivr.net/npm/tailwindcss@3.4.17/lib/public/colors.js",
    "Bootstrap 5.3": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/scss/_variables.scss",
    "Open Color": "https://cdn.jsdelivr.net/npm/open-color@1.9.1/open-color.json",
}
HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")

palettes = {}
for label, url in SOURCES.items():
    try:
        text = requests.get(url, timeout=20).text
        found = sorted(set(HEX_RE.findall(text)))
        palettes[label] = found
        print("{:<14} {} unique hex values".format(label, len(found)))
    except Exception as e:
        print("Could not download {}: {}".format(label, e))
print()

palette_results = {}
for label, colors in palettes.items():
    palette_results[label] = {}
    for name, tok in tokenizers.items():
        palette_results[label][name] = {}
        for case in ("as_written", "upper"):
            vals = [c if case == "as_written" else c.upper() for c in colors]
            costs = [in_context_cost(tok, v) for v in vals]
            n = len(costs)
            res = {
                "n_colors": n,
                "avg_tokens_in_context": round(statistics.mean(costs), 2),
                "pct_5plus_tokens": round(100 * sum(c >= 5 for c in costs) / n, 1),
                "pct_2_or_fewer_tokens": round(100 * sum(c <= 2 for c in costs) / n, 1),
                "max_tokens": max(costs),
                "worst_5": [v for _, v in sorted(zip(costs, vals), reverse=True)[:5]],
            }
            palette_results[label][name][case] = res
            print("{:<14} {:<22} {:<10} avg {:>4}  >=5 tok: {:>5}%  <=2 tok: {:>5}%  max {}".format(
                label, name, case, res["avg_tokens_in_context"], res["pct_5plus_tokens"],
                res["pct_2_or_fewer_tokens"], res["max_tokens"]))
    print()

# ---------------------------------------------------------------------
# (C) Decimal rem check, per tokenizer
# ---------------------------------------------------------------------
print("=" * 70)
print("(C) rem values, per tokenizer (in 'padding: X;' context)")
print("=" * 70)
rem_values = ["0.25rem", "0.5rem", "0.75rem", "1rem", "1.25rem", "1.5rem", "2rem", "2.5rem", "3rem"]
rem_results = {}
for name, tok in tokenizers.items():
    base = count(tok, "padding:;")
    rem_results[name] = {v: count(tok, "padding: " + v + ";") - base for v in rem_values}
    print("{:<22} ".format(name) + "  ".join("{}={}".format(k, v) for k, v in rem_results[name].items()))
print()

output = {
    "tokenizers": list(tokenizers),
    "hex_pair_sweep": sweep,
    "examples": examples_out,
    "palette_sources": SOURCES,
    "palette_colors": palettes,
    "palette_results": palette_results,
    "rem_results": rem_results,
}
with open("tokenizer_and_palette_results.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("Saved to tokenizer_and_palette_results.json")
