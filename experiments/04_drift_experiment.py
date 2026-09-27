"""
Drift experiment: do colors that split into more tokens come back wrong more often?

Uses every color in Tailwind CSS v3's default palette (real colors developers
actually write) and asks GPT-4o to produce each one under three conditions:

  1. COPY       - the exact hex is in the prompt; model writes a CSS rule using it.
                  (Like a Figma payload that already contains the value.)
  2. CONVERT    - the prompt gives rgb(r, g, b); model must write the hex.
                  (Like a pipeline that transforms a value instead of copying it.)
  3. RECALL     - the prompt gives only the Tailwind name, e.g. "blue-500";
                  model must produce the hex from memory.
                  (Like an AI writing brand colors it "knows" without a lookup.)

Each color is sampled several times at normal temperature. For each answer we
record whether it's exactly right, and if not, how far off it is.
Accuracy is then compared against how many tokens the correct value takes
in GPT-4o's own tokenizer (o200k_base).

SETUP (run in Terminal):
    cd results
    export OPENAI_API_KEY="sk-...your key..."      # stays in this terminal only
    python3 ../experiments/04_drift_experiment.py

Optional: python3 ../experiments/04_drift_experiment.py gpt-4o-mini    (cheaper model)

Safe to re-run: finished requests are saved to drift_raw_<model>.jsonl and
skipped next time, so an interruption doesn't waste money.
Writes drift_results_<model>.json with the summary.
"""

import concurrent.futures
import json
import os
import re
import statistics
import sys
import threading
import time

import requests
import tiktoken

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"
SAMPLES = 5            # answers per color per condition (returned in one request)
TEMPERATURE = 1.0      # normal/default sampling, where drift would show up
WORKERS = 6
API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not API_KEY:
    sys.exit("OPENAI_API_KEY is not set. Run:  export OPENAI_API_KEY=\"sk-...\"  then re-run.")

RAW_PATH = "drift_raw_{}.jsonl".format(MODEL)
OUT_PATH = "drift_results_{}.json".format(MODEL)

enc = tiktoken.get_encoding("o200k_base")


def token_cost(hex_value):
    """Tokens the hex adds inside a real CSS declaration, in GPT-4o's tokenizer."""
    return len(enc.encode("color: " + hex_value + ";")) - len(enc.encode("color:;"))


# ---------------------------------------------------------------------
# Load Tailwind v3 palette with names
# ---------------------------------------------------------------------
URL = "https://cdn.jsdelivr.net/npm/tailwindcss@3.4.17/lib/public/colors.js"
src = requests.get(URL, timeout=30).text
colors = []
family = None
for line in src.splitlines():
    m_fam = re.match(r'^\s*"?([a-zA-Z]+)"?\s*:\s*\{\s*$', line)
    if m_fam:
        family = m_fam.group(1)
        continue
    m_shade = re.match(r'^\s*"?(\d{2,3})"?\s*:\s*["\'](#[0-9a-fA-F]{6})["\']', line)
    if m_shade and family:
        colors.append({"name": "{}-{}".format(family, m_shade.group(1)),
                       "hex": m_shade.group(2).lower()})
# de-duplicate by name (deprecated aliases can repeat families)
seen, uniq = set(), []
for c in colors:
    if c["name"] not in seen:
        seen.add(c["name"])
        uniq.append(c)
colors = uniq
if len(colors) < 150:
    sys.exit("Parsed only {} Tailwind colors - palette format changed; tell Claude.".format(len(colors)))
for c in colors:
    c["tokens"] = token_cost(c["hex"])
    h = c["hex"][1:]
    c["rgb"] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
print("Loaded {} Tailwind colors. Model: {}. {} samples each x 3 conditions.".format(
    len(colors), MODEL, SAMPLES))


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
def prompt_for(cond, c):
    if cond == "copy":
        return ("Write one CSS rule for the selector .card that sets background-color to {}. "
                "Output only the CSS, nothing else.").format(c["hex"])
    if cond == "convert":
        r, g, b = c["rgb"]
        return ("Convert rgb({}, {}, {}) to a 6-digit hex color and write one CSS rule for "
                ".card that sets background-color to it. Output only the CSS, nothing else.").format(r, g, b)
    if cond == "recall":
        return ("Using the Tailwind CSS v3 default color palette, write one CSS rule for .card "
                "that sets background-color to the exact hex value of {}. "
                "Output only the CSS, nothing else.").format(c["name"])
    raise ValueError(cond)


HEX_RE = re.compile(r"#([0-9a-fA-F]{6})\b")


def parse_hex(text):
    m = HEX_RE.search(text or "")
    return ("#" + m.group(1).lower()) if m else None


# ---------------------------------------------------------------------
# API calls with resume + retry
# ---------------------------------------------------------------------
done = {}
if os.path.exists(RAW_PATH):
    with open(RAW_PATH) as f:
        for line in f:
            r = json.loads(line)
            done[(r["cond"], r["name"])] = r
print("Already completed: {} requests (will be skipped).".format(len(done)))

lock = threading.Lock()
cost_tokens = {"in": 0, "out": 0}


def call(cond, c):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_for(cond, c)}],
        "temperature": TEMPERATURE,
        "n": SAMPLES,
        "max_tokens": 40,
    }
    for attempt in range(6):
        try:
            resp = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers={"Authorization": "Bearer " + API_KEY},
                                 json=body, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                return {"error": "HTTP {}: {}".format(resp.status_code, resp.text[:300])}
            data = resp.json()
            return data
        except requests.RequestException as e:
            time.sleep(2 ** attempt)
            last = str(e)
    return {"error": "gave up after retries"}


stop = threading.Event()


def run_one(cond, c):
    if stop.is_set():
        return cond, c, None, "skipped (stopped early)"
    data = call(cond, c)
    if "error" in data:
        return cond, c, None, data["error"]
    answers = [ch["message"]["content"] for ch in data.get("choices", [])]
    usage = data.get("usage", {})
    rec = {"cond": cond, "name": c["name"], "hex": c["hex"], "tokens": c["tokens"],
           "answers": answers, "parsed": [parse_hex(a) for a in answers], "usage": usage}
    with lock:
        with open(RAW_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        cost_tokens["in"] += usage.get("prompt_tokens", 0)
        cost_tokens["out"] += usage.get("completion_tokens", 0)
    return cond, c, rec, None


jobs = [(cond, c) for cond in ("copy", "convert", "recall") for c in colors
        if (cond, c["name"]) not in done]
print("Requests to send now: {}".format(len(jobs)))

errors = []
finished = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = [pool.submit(run_one, cond, c) for cond, c in jobs]
    for fut in concurrent.futures.as_completed(futures):
        cond, c, rec, err = fut.result()
        finished += 1
        if err:
            errors.append((cond, c["name"], err))
            if len(errors) == 1:
                print("First error:", err)
            if len(errors) >= 10 and len(errors) > finished // 2:
                print("Too many errors - stopping. Check the API key / billing.")
                stop.set()
                break
        else:
            done[(cond, c["name"])] = rec
        if finished % 50 == 0:
            print("  {}/{} done".format(finished, len(jobs)))

if errors:
    print("{} requests failed; re-run the script to retry just those.".format(len(errors)))

# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------
def hex_digits_off(a, b):
    return sum(x != y for x, y in zip(a[1:], b[1:]))


def rgb_distance(a, b):
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5


def bucket(t):
    if t <= 4:
        return "3-4 tokens"
    if t == 5:
        return "5 tokens"
    return "6-7 tokens"


summary = {"model": MODEL, "samples_per_color": SAMPLES, "temperature": TEMPERATURE,
           "n_colors": len(colors), "conditions": {}}
print()
print("=" * 72)
print("Exact-match accuracy by how many tokens the correct color takes (GPT-4o tokenizer)")
print("=" * 72)
for cond in ("copy", "convert", "recall"):
    rows = [r for (cd, _), r in done.items() if cd == cond]
    by_bucket = {}
    per_color = []
    near_misses = 0
    wrong_total = 0
    for r in rows:
        correct = [p == r["hex"] for p in r["parsed"]]
        acc = sum(correct) / len(correct) if correct else 0
        per_color.append((r["tokens"], acc))
        by_bucket.setdefault(bucket(r["tokens"]), []).extend(correct)
        for p in r["parsed"]:
            if p and p != r["hex"]:
                wrong_total += 1
                if hex_digits_off(p, r["hex"]) <= 2:
                    near_misses += 1
    # correlation between token count and per-color accuracy
    corr = None
    if len(per_color) > 2:
        xs = [t for t, _ in per_color]
        ys = [a for _, a in per_color]
        if statistics.pstdev(xs) > 0 and statistics.pstdev(ys) > 0:
            mx, my = statistics.mean(xs), statistics.mean(ys)
            cov = sum((x - mx) * (y - my) for x, y in per_color) / len(per_color)
            corr = round(cov / (statistics.pstdev(xs) * statistics.pstdev(ys)), 3)
    buckets_out = {}
    for b in ("3-4 tokens", "5 tokens", "6-7 tokens"):
        vals = by_bucket.get(b, [])
        if vals:
            buckets_out[b] = {"answers": len(vals), "accuracy_pct": round(100 * sum(vals) / len(vals), 1)}
    summary["conditions"][cond] = {
        "colors_tested": len(rows),
        "overall_accuracy_pct": round(100 * sum(a for _, a in per_color) / max(len(per_color), 1), 1),
        "by_token_bucket": buckets_out,
        "corr_tokens_vs_accuracy": corr,
        "wrong_answers": wrong_total,
        "wrong_answers_off_by_1_or_2_hex_digits": near_misses,
    }
    s = summary["conditions"][cond]
    print("\n{} (overall {}% exact, {} colors)".format(cond.upper(), s["overall_accuracy_pct"], s["colors_tested"]))
    for b, v in buckets_out.items():
        print("   {:<12} {:>6}% exact   ({} answers)".format(b, v["accuracy_pct"], v["answers"]))
    print("   correlation(token count, accuracy): {}".format(corr))
    print("   wrong answers: {}   of which off by only 1-2 hex digits: {}".format(wrong_total, near_misses))

# a few concrete examples of wrong answers, for the article
examples = []
for (cd, _), r in done.items():
    for p in r["parsed"]:
        if p and p != r["hex"] and len(examples) < 40:
            examples.append({"cond": cd, "name": r["name"], "correct": r["hex"], "got": p,
                             "tokens": r["tokens"], "hex_digits_off": hex_digits_off(p, r["hex"]),
                             "rgb_distance": round(rgb_distance(p, r["hex"]), 1)})
summary["example_wrong_answers"] = examples
summary["api_tokens_used_this_run"] = cost_tokens
with open(OUT_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved summary to {} and raw answers to {}".format(OUT_PATH, RAW_PATH))
print("API tokens used this run: {} in / {} out".format(cost_tokens["in"], cost_tokens["out"]))
