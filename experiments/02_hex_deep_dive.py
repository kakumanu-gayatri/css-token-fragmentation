"""
Hex Color Token Fragmentation - Deep Dive
==========================================
Follow-up to 01_css_tokenization_experiment.py. That script found that most
repeated-digit hex triplets (#000000, #111111, ... #EEEEEE) tokenize to
2-3 tokens, but four specific pairs (C0, 1A, 3F, A9) spike to 7 tokens
for the same "#hhh" repeated format. This script decodes WHY, by:

  1. Printing the actual token pieces (not just counts) for every pair
     in the original sample, so we can see exactly where the split
     boundaries fall.
  2. Running ALL 256 possible two-character hex pairs (00-FF) through
     the same "#hhh" format to see if the cliff is really about
     "digit == letter vs digit != letter" or something else.
  3. Testing whether the effect is specific to the "#hhh" repeated
     pattern, or shows up for the bare 6-char hex code too
     (e.g. "C0C0C0" without repeating each pair 3x - i.e. as it would
     really appear in a CSS file: #C0C0C0 is the same as #hhh repeated,
     so this also checks the ACTUAL 6-hex-digit form for a wider set
     of real, non-shorthand colors like #3A7BD5).

Run:
    python3 ../experiments/02_hex_deep_dive.py

Writes hex_deep_dive_results.json in the current directory (run from results/).
"""

import json
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
assert enc.encode("hello world!") == [15339, 1917, 0], "Tokenizer mismatch"
print("Tokenizer verified.\n")


def tokens_for(s):
    ids = enc.encode(s)
    pieces = [enc.decode([i]) for i in ids]
    return ids, pieces


# ---------------------------------------------------------------------
# 1. Original sample - print full token breakdown, not just counts
# ---------------------------------------------------------------------
original_sample = ["00", "11", "22", "33", "44", "55", "66", "77", "88", "99",
                    "AA", "BB", "CC", "DD", "EE", "FF", "80", "40", "C0", "1A", "3F", "A9"]

print("=== Full token breakdown for #hhh (h repeated 3x) ===")
breakdown = {}
for h in original_sample:
    color = f"#{h}{h}{h}"
    ids, pieces = tokens_for(color)
    breakdown[h] = {"color": color, "token_count": len(ids), "pieces": pieces}
    print(f"  {color:>10}  ({len(ids)} tok)  ->  {pieces}")
print()

# ---------------------------------------------------------------------
# 2. ALL 256 two-char hex pairs, same #hhh format
# ---------------------------------------------------------------------
print("=== Scanning all 256 hex pairs (00-FF) as #hhh ===")
all_pairs_results = []
for i in range(256):
    h = format(i, '02X')
    color = f"#{h}{h}{h}"
    ids, pieces = tokens_for(color)
    all_pairs_results.append({"hex": h, "color": color, "token_count": len(ids), "pieces": pieces})

counts_hist = {}
for r in all_pairs_results:
    counts_hist[r["token_count"]] = counts_hist.get(r["token_count"], 0) + 1

print("Token count distribution across all 256 pairs:")
for tc in sorted(counts_hist):
    print(f"  {tc} tokens: {counts_hist[tc]} pairs")

spikes = [r for r in all_pairs_results if r["token_count"] >= 6]
print(f"\n{len(spikes)} pairs spike to 6+ tokens:")
for r in spikes:
    print(f"  {r['color']}  ({r['token_count']} tok)  ->  {r['pieces']}")
print()

# Check: does same-character (00,11,...FF - "doubled" hex digit) always
# win, and different-character always lose?
same_char = [r for r in all_pairs_results if r["hex"][0] == r["hex"][1]]
diff_char = [r for r in all_pairs_results if r["hex"][0] != r["hex"][1]]
same_avg = sum(r["token_count"] for r in same_char) / len(same_char)
diff_avg = sum(r["token_count"] for r in diff_char) / len(diff_char)
print(f"Same-character pairs (00,11,22...FF): avg {same_avg:.2f} tokens (n={len(same_char)})")
print(f"Different-character pairs (all others): avg {diff_avg:.2f} tokens (n={len(diff_char)})")
print()

# ---------------------------------------------------------------------
# 3. Real (non-shorthand) 6-digit hex colors, as they'd actually
#    appear in CSS - not the #hhh-repeated shorthand pattern.
# ---------------------------------------------------------------------
real_colors = [
    "#3A7BD5", "#FF5733", "#1ABC9C", "#2ECC71", "#9B59B6",
    "#E74C3C", "#F39C12", "#34495E", "#16A085", "#27AE60",
    "#2980B9", "#8E44AD", "#C0392B", "#D35400", "#7F8C8D",
    "#A9A9A9", "#000000", "#FFFFFF", "#808080", "#C0C0C0",
]
print("=== Real (non-shorthand) 6-digit hex colors ===")
real_results = []
for c in real_colors:
    ids, pieces = tokens_for(c)
    real_results.append({"color": c, "token_count": len(ids), "pieces": pieces})
    print(f"  {c:>10}  ({len(ids)} tok)  ->  {pieces}")
print()

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
output = {
    "tokenizer": "cl100k_base",
    "verified_against_test_vector": True,
    "original_sample_breakdown": breakdown,
    "all_256_pairs": all_pairs_results,
    "token_count_distribution": counts_hist,
    "spikes_6plus_tokens": spikes,
    "same_char_avg_tokens": same_avg,
    "diff_char_avg_tokens": diff_avg,
    "real_hex_colors": real_results,
}

with open("hex_deep_dive_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved to hex_deep_dive_results.json")
