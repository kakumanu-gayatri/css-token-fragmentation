# CSS Token Fragmentation

How LLM tokenizers split CSS values, and why it matters for AI-generated UI.

When an AI model writes a CSS value like `#3b82f6` or `0.25rem`, it doesn't produce it in one step. It predicts it token by token. This repo measures how many tokens common CSS values take across current tokenizers, using real design-system palettes, and tests whether values that take more tokens come back wrong more often.

## Key findings

**1. Hex colors fragment based on whether their digits repeat.**
Across all 256 two-character hex pairs repeated into six-digit colors (`#C0C0C0`, `#1A1A1A`, ...), on GPT-4's `cl100k_base` tokenizer:

| Pair type | Example | Avg tokens | Pairs at 7-token max |
|---|---|---|---|
| Same character (`00`, `11`, ... `FF`) | `#FFFFFF` → `#`, `FFFFFF` | 2.94 | 0 |
| Different characters (240 pairs) | `#C0C0C0` → `#`,`C`,`0`,`C`,`0`,`C`,`0` | 5.15 | 120 |

**2. The pattern holds across current tokenizers and real palettes.**
Share of colors taking 5+ tokens, measured inside a real declaration (`color: #3b82f6;`), hex as written in each palette's source:

| Palette | Colors | GPT-4 (`cl100k_base`) | GPT-4o (`o200k_base`) | DeepSeek-V3 | Qwen2.5 |
|---|---|---|---|---|---|
| Tailwind CSS v3 | 243 | 57% | 58% | 61% | 91% |
| Open Color | 132 | 61% | 63% | 65% | 91% |
| Bootstrap 5.3 | 21 | 43% | 43% | 43% | 81% |

Fewer than 1% of Tailwind colors get the clean 2-token form. Common neutral greys such as `#f5f5f5` and `#f3f4f6` are among the most fragmented, because their two hex digits differ. Qwen2.5 splits every digit individually, so nearly all colors fragment there.

**3. Decimal rem values cost more than whole numbers.**
In `padding: X;` context, `1rem`, `2rem`, and `3rem` add 3 tokens; `0.25rem`, `0.5rem`, `1.25rem`, `1.5rem`, and `2.5rem` add 5 on GPT-4, GPT-4o, and DeepSeek-V3 (5–6 on Qwen2.5).

**4. What didn't hold.**
The starting hypothesis was that common spacing-scale values (4, 8, 16, 24 ...) tokenize more cleanly than arbitrary values (3, 17, 43 ...). They don't: every integer from 0 to 999 is a single token in `cl100k_base`, and `16px` and `17px` both take 2 tokens.

## Status

- Tokenizer measurements (experiments 01–03): complete, results in `results/`.
- Live-model drift test (experiment 04): in progress. It asks GPT-4o to write every Tailwind color under three conditions (copy the hex from the prompt, convert from `rgb()`, recall from the Tailwind name) and checks whether exact-match accuracy falls as token count rises.

## Reproduce

Requires Python 3.8+.

```bash
pip3 install -r requirements.txt
cd results
python3 ../experiments/01_css_tokenization_experiment.py   # integers, px scale, rem, hex sample
python3 ../experiments/02_hex_deep_dive.py                 # all 256 hex pairs, token breakdowns
python3 ../experiments/03_tokenizer_and_palette_check.py   # 4 tokenizers x 3 real palettes
```

`tokenizers` is only needed for the Qwen2.5 and DeepSeek-V3 comparisons; experiment 03 skips them if it isn't installed.

The drift test calls the OpenAI API (roughly $0.50–$1 on `gpt-4o`). Keep your key in the environment, never in a file:

```bash
cd results
export OPENAI_API_KEY="sk-..."
python3 ../experiments/04_drift_experiment.py            # or: ... 04_drift_experiment.py gpt-4o-mini
```

## Method notes

- `cl100k_base` is verified against its published test vector (`"hello world!"` → `[15339, 1917, 0]`) before any measurement.
- "In context" counts are `tokens("color: <hex>;") - tokens("color:;")`, so they reflect how a value is split inside a real CSS declaration rather than on its own.
- Palettes are downloaded at run time from jsDelivr: `tailwindcss@3.4.17`, `bootstrap@5.3.3`, `open-color@1.9.1`. Tailwind v4 uses OKLCH rather than hex and is not covered.
- Token count is a leading indicator. Experiments 01–03 measure fragmentation, not error rates; experiment 04 tests the link to actual output errors.

## Related work

- *Date Fragments: A Hidden Bottleneck of Tokenization for Temporal Reasoning* (arXiv:2505.16088) shows date fragmentation correlates with lower reasoning accuracy. This repo applies the same question to design values.
- Kakumanu, *Figma-to-Code at Scale: What Actually Drives Cost, Quota, and Quality* ([Dev.to](https://dev.to/gayatrikakumanu25/figma-to-code-at-scale-what-actually-drives-cost-quota-and-quality-2e4)): measurements of raw vs. tokenized values in Figma MCP output.

## License

MIT. See `LICENSE`.
