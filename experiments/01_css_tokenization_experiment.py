"""
CSS Value Tokenization Experiment
==================================
Uses the real cl100k_base tokenizer (tiktoken, the actual GPT-3.5/GPT-4
tokenizer) to check whether common CSS spacing/sizing values get
represented as a single token, versus values a model might drift to
getting split into multiple digit tokens.

This tests the hypothesis: numbers that require multiple tokens to
represent are less "atomic" to the model than numbers that get a single
learned token, which may correlate with inconsistent generation.

Run:
    pip3 install tiktoken
    python3 ../experiments/01_css_tokenization_experiment.py

Writes results.json in the current directory (run from results/).
"""

import json
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# Sanity check against the known-correct test vector for this tokenizer.
assert enc.encode("hello world!") == [15339, 1917, 0], "Tokenizer mismatch — not real cl100k_base"
print("Tokenizer verified against known test vector.\n")


def token_count(s):
    return len(enc.encode(s))


def tokens_for(s):
    ids = enc.encode(s)
    pieces = [enc.decode([i]) for i in ids]
    return ids, pieces


# ---------------------------------------------------------------------
# 1. Exhaustive scan: every integer 0-999, how many tokens as a bare number?
# ---------------------------------------------------------------------
bare_number_counts = {}
for n in range(0, 1000):
    bare_number_counts[n] = token_count(str(n))

single_token_numbers = sorted([n for n, c in bare_number_counts.items() if c == 1])
multi_token_numbers = sorted([n for n, c in bare_number_counts.items() if c > 1])

print(f"0-999: {len(single_token_numbers)} numbers are a single token, "
      f"{len(multi_token_numbers)} need multiple tokens "
      f"({len(multi_token_numbers)/10:.1f}% split)\n")

# ---------------------------------------------------------------------
# 2. CSS spacing scale (common 4px/8px-based design system values)
#    vs a set of "arbitrary" values a model might drift to.
# ---------------------------------------------------------------------
common_spacing_scale = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40,
                         48, 56, 64, 72, 80, 96, 112, 128, 144, 160, 192, 224, 256]

arbitrary_nearby_values = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 27, 29, 31,
                            33, 37, 41, 43, 47, 53, 57, 61, 67, 71, 73, 77, 83,
                            87, 91, 97, 103, 109, 113, 117, 121, 127, 131, 137,
                            141]

with_px = lambda vals: [(n, f"{n}px", token_count(f"{n}px")) for n in vals]

scale_results = with_px(common_spacing_scale)
arbitrary_results = with_px(arbitrary_nearby_values)

scale_single = sum(1 for _, _, c in scale_results if c == 2)  # number+px = 2 tokens when number is 1 token
arbitrary_single = sum(1 for _, _, c in arbitrary_results if c == 2)

print("=== Common design-system spacing scale (as 'Npx') ===")
for n, s, c in scale_results:
    print(f"  {s:>8}  ->  {c} token(s)")
print(f"  {scale_single}/{len(scale_results)} are the minimal 2-token form (clean number + 'px')\n")

print("=== Arbitrary/adjacent values a model might drift to (as 'Npx') ===")
for n, s, c in arbitrary_results:
    print(f"  {s:>8}  ->  {c} token(s)")
print(f"  {arbitrary_single}/{len(arbitrary_results)} are the minimal 2-token form (clean number + 'px')\n")

# ---------------------------------------------------------------------
# 3. rem values (decimal point adds another dimension)
# ---------------------------------------------------------------------
rem_scale = ["0.25rem", "0.5rem", "0.75rem", "1rem", "1.25rem", "1.5rem",
             "1.75rem", "2rem", "2.5rem", "3rem", "4rem"]
rem_results = [(s, token_count(s)) for s in rem_scale]

print("=== rem scale ===")
for s, c in rem_results:
    print(f"  {s:>8}  ->  {c} token(s)")
print()

# ---------------------------------------------------------------------
# 4. Hex color pairs (00-FF) - do common ones (00, FF, 80) get fused
#    more cleanly than arbitrary ones (37, A9, etc.)?
# ---------------------------------------------------------------------
hex_pairs = ["00", "11", "22", "33", "44", "55", "66", "77", "88", "99",
             "AA", "BB", "CC", "DD", "EE", "FF", "80", "40", "C0", "1A", "3F", "A9"]
hex_results = [(h, token_count(f"#{h}{h}{h}")) for h in hex_pairs]

print("=== Hex color shorthand (#hhh repeated) ===")
for h, c in hex_results:
    print(f"  #{h}{h}{h}  ->  {c} token(s)")
print()

# ---------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------
output = {
    "tokenizer": "cl100k_base",
    "verified_against_test_vector": True,
    "bare_numbers_0_999": {
        "single_token_count": len(single_token_numbers),
        "multi_token_count": len(multi_token_numbers),
        "single_token_numbers": single_token_numbers,
        "multi_token_numbers": multi_token_numbers,
    },
    "css_spacing_scale_vs_arbitrary": {
        "common_spacing_scale": scale_results,
        "arbitrary_nearby_values": arbitrary_results,
        "scale_clean_ratio": scale_single / len(scale_results),
        "arbitrary_clean_ratio": arbitrary_single / len(arbitrary_results),
    },
    "rem_scale": rem_results,
    "hex_color_pairs": hex_results,
}

with open("results.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved full results to results.json")
