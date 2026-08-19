import json
import os
import time
import datetime
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

# Consistency check: run every model this many times, cycling through all
# models once per pass (run1: llama, qwen, mistral -> run2: llama, qwen,
# mistral -> run3: ...) rather than exhausting one model's 3 runs before
# moving to the next. This spreads any time-of-day/load-related drift on
# the Ollama server evenly across models instead of concentrating it.
N_CONSISTENCY_RUNS = 3

OLLAMA_MODELS = [
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b",
    "mistral:7b",
]

MODEL_ALIASES = {
    "llama3.1:8b-instruct-q4_K_M": "llama",
    "qwen2.5:7b": "qwen",
    "mistral:7b": "mistral",
}

REQUEST_TIMEOUT = 300
BATCH_SIZE = 5

# tracks which batches are done, for resume
PROGRESS_MANIFEST = "scoring_progress.json"

# 2. Load the input posts dataset
with open('oil_posts_v3.json', 'r', encoding='utf-8') as f:
    posts_data = json.load(f)

# TEST MODE: limit to the first N posts for debugging. Set to None (or
# remove the slice below) to run the full dataset.
TEST_LIMIT = None
if TEST_LIMIT is not None:
    posts_data = posts_data[:TEST_LIMIT]
    print(f"TEST MODE: only processing the first {TEST_LIMIT} posts.")

# Load the consolidated v1.3 rubric (oil_price_direction is a regular
# scored feature, OIL_DIRECTION_01, inside "features" - not a separate
# top-level triple of fields). "eligibility" is its own section since it's
# a gate decided before feature scoring, not a feature itself.
with open('oil_rubric_v1_4.json', 'r', encoding='utf-8') as f:
    rubric_data = json.load(f)

FEATURE_IDS = [f["id"] for f in rubric_data["features"]]
ELIGIBILITY_CODES = [c["code"] for c in rubric_data["eligibility"]["codes"]]
EXCLUSION_REASON_CODES = [c["code"]
                          for c in rubric_data["eligibility"]["exclusion_reasons"]]

rubric_block = "\n\n".join(
    f"### {f['id']} — {f['feature']}\n{f['rubric']}"
    for f in rubric_data["features"]
)

eligibility_block = "\n".join(
    f"- {c['code']}: {c['description']}"
    for c in rubric_data["eligibility"]["codes"]
)

exclusion_reason_block = "\n".join(
    f"- {c['code']}: {c['description']}"
    for c in rubric_data["eligibility"]["exclusion_reasons"]
)

# Fully-enumerated scores block (every real feature id already present as
# valid JSON, no comments, no "repeat this" placeholder) - smaller local
# models were imitating only a single illustrative key in a comment-based
# template and dropping the rest, since a JS-style comment isn't valid JSON.
scores_template = ",\n".join(
    f'''        "{fid}": {{
          "score": <int, or null if not applicable>,
          "evidence_span": "<shortest supporting quote or null>",
          "justification": "<1-2 sentences tied to explicit wording, or null>"
        }}'''
    for fid in FEATURE_IDS
)

# num_ctx in Ollama is the TOTAL context window (input prompt + generated
# output combined), not an output-only budget. Give NUM_PREDICT its own
# generous but bounded budget, then size NUM_CTX comfortably above
# (prompt + NUM_PREDICT), with margin.
NUM_PREDICT = BATCH_SIZE * len(FEATURE_IDS) * 150 + 500  # ~8,750 tokens
PROMPT_TOKEN_ESTIMATE = 4000  # rough upper bound for rubric+rules+batch text
NUM_CTX = PROMPT_TOKEN_ESTIMATE + NUM_PREDICT + 2000  # margin
# NOTE: verify NUM_CTX against each model's actual max context (some
# quantized local builds cap lower than the base model card advertises) -
# if it's capped below this value, drop BATCH_SIZE instead of raising NUM_CTX.

SYSTEM_PROMPT = f"""You are a literal-minded coder scoring Trump Truth Social posts for an
oil/energy-market study, benchmarked against an adjudicated human standard.
Apply the rubric provided below exactly - do not use your own judgement about intent.

RULES:
- Code only the visible text. No hindsight (don't infer later market moves),
  no engagement signals, no assumptions about what Trump "generally" means.
- A keyword is not evidence ("barrels" of meth, "production" of cars/chips,
  generic "made in America" are not oil content unless context confirms it).
- Score each feature independently - e.g. pro-drilling does not imply
  anti-green or energy-nationalism without its own separate evidence, and
  a severe geopolitical threat (OIL_MARKET_02) does not by itself dictate
  OIL_DIRECTION_01 - price direction depends on the specific mechanism, not
  the threat's severity.
- Judge affect/blame/nationalism from the energy-relevant passage only, not
  unrelated capitalisation or insults elsewhere in a long post.
- null vs 0 vs 9: 0 = confirmed absent (or, for OIL_DIRECTION_01, no clear
  price implication). null = not applicable (post excluded/unresolved).
  9 = eligible post but this feature is genuinely ambiguous. Never use 0
  for uncertainty - use 9 instead, and never guess.
- Each feature's rubric entry below lists its own "scores" array - that is
  the complete set of legal values for that feature and nothing else. A
  score must come from that exact feature's own array: never reuse a value
  from a different feature's array, never output a value the array doesn't
  contain (e.g. a 2 for a feature whose array stops at 1), and never invent
  a value not listed anywhere (e.g. 3, -2, 99). If uncertain which value
  applies, use that feature's own ambiguity code (9) rather than guessing
  or borrowing the nearest value from another feature.
- A post and its re-truth (identical text) must get identical scores.
- Rate your own confidence in this post's overall coding: 1 (low), 2
  (medium), or 3 (high). Use 1 when eligibility or multiple features were
  genuinely hard calls, not just when a score happens to be 9.

STEP 1 - ELIGIBILITY (decide before scoring anything else):
Assign exactly one code:
{eligibility_block}

If EXCLUDE, also choose exactly one exclusion_reason:
{exclusion_reason_block}

Record energy_evidence_span (shortest phrase justifying inclusion; null if
EXCLUDE/UNRESOLVED). If EXCLUDE or UNRESOLVED: set every feature score
(including OIL_DIRECTION_01) to null and stop.

STEP 2 - FEATURE SCORING (only if INCLUDE_CORE/INCLUDE_CONTEXT):
Score every feature in the rubric provided, including OIL_DIRECTION_01
(implied oil-price direction - a distinct question from topic/severity,
scored the same way as every other feature). For any non-zero score, give
an evidence_span (shortest exact quote) and a 1-2 sentence justification;
for OIL_DIRECTION_01 specifically, name the mechanism in that justification
rather than just asserting a direction. Do not claim causal certainty for
OIL_DIRECTION_01 - you are coding implied direction, not forecasting the
market. Use only the "scores" values listed for that specific feature id
in the rubric.

RESOLVED LINK CONTENT:
A post may include "links" (headline/subheading/article_excerpt per URL).
Use it for eligibility and every feature except OIL_AFFE_01/OIL_PRAG_01,
which score Trump's own rhetoric from "text" alone. Multiple links: eligible
if ANY link is oil/energy-relevant; cite the specific link's headline in
evidence_span. If "text" is empty or URL-only, score OIL_AFFE_01/OIL_PRAG_01
as 0 (confirmed absent), not 9.

OUTPUT FORMAT:
Return a valid JSON object matching this exact structure for the batch.
Every feature id in the rubric MUST appear as its own key inside "scores" -
do not omit any, and do not collapse them into fewer keys. Each "score"
value must be null, or exactly one of that feature's own "scores" values
from the rubric - never a value from a different feature's array and never
a value not listed for that feature at all:

{{
  "results": [
    {{
      "post_id": "string_matching_input_id",
      "corpus_eligibility": "<one of {ELIGIBILITY_CODES}>",
      "exclusion_reason": "<one of {EXCLUSION_REASON_CODES} or null>",
      "energy_evidence_span": "<shortest supporting phrase or null>",
      "scores": {{
{scores_template}
      }},
    }}
  ]
}}

Every field must be present even when null. Use exactly the feature ids
from the rubric as "scores" keys - do not invent, rename, or omit any.
Output only the JSON object - no preamble, no ```json fences.
"""


def score_batch(model, batch_payload, retries=3):
    """Sends micro-batch of posts to a local Ollama model with backoff retry
    logic. Returns (parsed_response_or_None, attempts_used)."""
    prompt_content = f"Evaluate these {len(batch_payload)} posts:\n" + \
        json.dumps(batch_payload, indent=2)

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt_content}"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            raw_text = response.json()["response"]
            return json.loads(raw_text), attempt + 1
        except Exception as e:
            print(f"Error on attempt {attempt + 1} ({model}): {e}")
            time.sleep(2 ** attempt)

    return None, retries


# ---------------------------------------------------------------------
# Crash-safety helpers
# ---------------------------------------------------------------------

def load_progress():
    """Progress manifest maps 'run{run_id}_{alias}' -> list of completed
    batch start-indices. Missing/corrupt file just means start fresh -
    the .jsonl files are the actual source of truth for what's saved, this
    manifest just avoids re-reading every .jsonl to figure out where to
    resume."""
    if os.path.exists(PROGRESS_MANIFEST):
        try:
            with open(PROGRESS_MANIFEST, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"WARNING: {PROGRESS_MANIFEST} unreadable, starting fresh.")
    return {}


def save_progress(progress):
    """Atomic write: write to a temp file then rename over the real one, so
    a crash mid-write never leaves a half-written, unreadable manifest."""
    tmp_path = PROGRESS_MANIFEST + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, PROGRESS_MANIFEST)


def append_jsonl(path, records):
    """Append records to a .jsonl file (one JSON object per line) and force
    the write to disk immediately - flush() alone can leave data sitting in
    an OS buffer that a crash loses; fsync forces it onto physical storage."""
    with open(path, 'a', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ---------------------------------------------------------------------
# Main loop — outer: consistency run, inner: model, innermost: batch.
# Every batch is appended to a per-(run,model) .jsonl file and fsynced
# before moving on, and the progress manifest is updated after every
# batch too - so a crash at any point loses at most the one batch that
# was in flight, and re-running the script picks up exactly where it
# left off instead of restarting from post 0.
# ---------------------------------------------------------------------
total_posts = len(posts_data)
n_batches = (total_posts + BATCH_SIZE - 1) // BATCH_SIZE
progress = load_progress()

retry_summary = {
    (run_id, model): {"first_try": 0, "retried": 0, "exhausted": 0}
    for run_id in range(1, N_CONSISTENCY_RUNS + 1)
    for model in OLLAMA_MODELS
}

run_start_time = time.time()

for run_id in range(1, N_CONSISTENCY_RUNS + 1):
    RUN_TAG = str(run_id)

    print(f"\n{'#'*60}")
    print(f"CONSISTENCY RUN {run_id}/{N_CONSISTENCY_RUNS}")
    print(f"{'#'*60}")

    for model in OLLAMA_MODELS:
        alias = MODEL_ALIASES.get(model, model.split(":")[0])
        progress_key = f"run{run_id}_{alias}"
        jsonl_path = f"oil_posts_v2_scored_LLM_{alias}_run{RUN_TAG}.jsonl"
        completed_batches = set(progress.get(progress_key, []))

        if len(completed_batches) >= n_batches:
            log(f"[run {run_id}][{alias}] SKIP — already complete "
                f"({len(completed_batches)}/{n_batches} batches per manifest).")
            continue

        if completed_batches:
            log(f"[run {run_id}][{alias}] RESUMING — "
                f"{len(completed_batches)}/{n_batches} batches already done.")

        print(f"\n{'='*60}")
        print(
            f"[run {run_id}] Starting batch evaluation for {total_posts} posts — model: {model}")
        print(f"{'='*60}")

        batch_num = 0
        for i in range(0, total_posts, BATCH_SIZE):
            batch_num += 1
            if i in completed_batches:
                continue  # already scored and saved in an earlier attempt

            batch = posts_data[i: i + BATCH_SIZE]

            batch_payload = [
                {
                    "post_id": item["comment_metadata"]["id"],
                    "text": item["comment"],
                    **({"links": item["resolution"]["links"]}
                       if item.get("resolution") and item["resolution"].get("links")
                       else {})
                }
                for item in batch
            ]

            elapsed = time.time() - run_start_time
            pct = 100 * batch_num / n_batches
            log(f"[run {run_id}][{alias}] PROGRESS batch {batch_num}/{n_batches} "
                f"({pct:.1f}%) — posts {i + 1}-{min(i + BATCH_SIZE, total_posts)}/{total_posts} "
                f"— elapsed {elapsed/60:.1f} min")

            api_response, attempts_used = score_batch(model, batch_payload)

            if api_response is None:
                retry_summary[(run_id, model)]["exhausted"] += 1
            elif attempts_used == 1:
                retry_summary[(run_id, model)]["first_try"] += 1
            else:
                retry_summary[(run_id, model)]["retried"] += 1

            batch_records = []
            if api_response and "results" in api_response:
                results_lookup = {res["post_id"]                                  : res for res in api_response["results"]}

                for item in batch:
                    p_id = item["comment_metadata"]["id"]
                    result = results_lookup.get(p_id, {})

                    scores = result.get("scores", {})
                    missing_features = [
                        fid for fid in FEATURE_IDS if fid not in scores]
                    if missing_features:
                        log(f"  [run {run_id}][{alias}] WARNING: post {p_id} "
                            f"missing scores for {missing_features}")

                    batch_records.append({
                        "comment": item.get("comment", ""),
                        "resolution": item.get("resolution"),
                        "text_is_url_only": item.get("text_is_url_only"),
                        "eligibility": {
                            "corpus_eligibility": result.get("corpus_eligibility"),
                            "exclusion_reason": result.get("exclusion_reason"),
                            "energy_evidence_span": result.get("energy_evidence_span"),
                        },
                        "scores": scores,
                        "confidence": result.get("confidence"),
                        "batch_attempts_used": attempts_used,
                        "run_id": RUN_TAG,
                        "backend": "ollama",
                        "model": model,
                        "comment_metadata": item.get("comment_metadata", {})
                    })
            else:
                log(f"[run {run_id}][{alias}] WARNING: batch at index {i} "
                    f"failed evaluation after {attempts_used} attempt(s) — 0 records saved for it.")

            # --- Crash-safety: write + fsync this batch immediately, then
            # mark it done in the manifest. If the process dies on the very
            # next line, at worst you redo one already-saved batch (the
            # jsonl append is idempotent-ish here since batches are skipped
            # by index, not by content - see note below if you rerun after
            # a partial jsonl write).
            if batch_records:
                append_jsonl(jsonl_path, batch_records)
            completed_batches.add(i)
            progress[progress_key] = sorted(completed_batches)
            save_progress(progress)

        # Finalize: convert the accumulated .jsonl into the combined .json
        # array your comparison/analysis scripts expect, without holding
        # everything in memory during the run itself.
        final_formatted_output = read_jsonl(jsonl_path)
        output_filename = f"oil_posts_v2_scored_LLM_{alias}_run{RUN_TAG}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_formatted_output, f, indent=2, ensure_ascii=False)

        log(f"[run {run_id}][{alias}] COMPLETE — {len(final_formatted_output)} "
            f"records saved to '{output_filename}' (source: '{jsonl_path}').")
        rs = retry_summary[(run_id, model)]
        total_batches_done = rs["first_try"] + rs["retried"] + rs["exhausted"]
        if total_batches_done:
            print(f"[run {run_id}][{alias}] Batch outcomes this session: "
                  f"{rs['first_try']} first try, {rs['retried']} retried, "
                  f"{rs['exhausted']} exhausted all attempts.")

print(f"\nAll {N_CONSISTENCY_RUNS} runs across all models complete.")

print("\n" + "=" * 60)
print("RETRY SUMMARY (this session only — resumed batches from a prior")
print("session aren't recounted here; check the .jsonl files for full history)")
print("=" * 60)
for (run_id, model), rs in retry_summary.items():
    total_batches_done = rs["first_try"] + rs["retried"] + rs["exhausted"]
    if total_batches_done:
        print(f"run {run_id} | {model:30s}: {total_batches_done} batches | "
              f"first_try={rs['first_try']} retried={rs['retried']} exhausted={rs['exhausted']}")
