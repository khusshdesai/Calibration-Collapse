"""
Synthetic Data Generator for AG News (Optimised for Speed)
--------------------------------------------------------------
Features:
- Batch Generation (requests 10 articles per LLM call)
- Parallelism (Thread Pool handles multiple requests to Ollama)
"""

import os
import time
import json
import argparse
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_MAP  = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}

OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """You are generating synthetic training data for an NLP classification task.

Dataset: AG News
Task: Classify news into 4 categories: World, Sports, Business, Sci/Tech

Instructions:
Generate exactly 10 realistic, diverse, and completely unique news examples for the category: {CATEGORY}

Requirements:
- Write a short news headline + 1–2 sentence description for each.
- Do NOT include explanations or intro text.
- Output your response STRICTLY as a valid JSON array of objects.
- Each object must have "text" (headline + description) and "label" ("{CATEGORY}").

Example format:
[
  {{"text": "Global leaders meet to discuss climate change policies amid rising environmental concerns.", "label": "{CATEGORY}"}},
  {{"text": "Major breakthrough in renewable energy promises cheaper solar power.", "label": "{CATEGORY}"}}
]
"""

# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic AG News data via Ollama (Fast)")
    parser.add_argument("--total-per-class", type=int, default=2500,
                        help="Total examples to generate per category (default: 2500)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="How many articles the LLM should write per prompt (default: 10)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel request threads (default: 4)")
    parser.add_argument("--output", type=str, default="synthetic_agnews_fast.csv",
                        help="Output CSV file path")
    parser.add_argument("--model", type=str, default="llama3.2",
                        help="Ollama model name to use")
    return parser.parse_args()

# ── Parse LLM response ────────────────────────────────────────────────────────
def parse_response(response_text: str, expected_category: str):
    """Extract valid JSON from the model response."""
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        results = []
        for item in data:
            # Ensure text exists
            text = item.get("text", "").strip()
            if text:
                results.append({
                    "text": text,
                    "label": LABEL_MAP[expected_category],
                    "label_name": expected_category
                })
        return results
    except Exception as e:
        return []

# ── Generate a single batch (Worker function) ────────────────────────────────
def generate_batch(model: str, category: str):
    prompt = PROMPT_TEMPLATE.format(CATEGORY=category)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8  # Slight randomness for diversity
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        generated_text = response.json().get("response", "")
        return parse_response(generated_text, category)
    except Exception as e:
        return []

# ── Generate all examples for a category ─────────────────────────────────────
def generate_for_category(model: str, category: str, total_n: int, batch_size: int, workers: int):
    results = []
    print(f"\n[{category}] Generating {total_n} examples (running parallel batches until reached) …")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_batch, model, category) for _ in range(workers)}
        
        batches_done = 0
        while futures and len(results) < total_n:
            for future in as_completed(list(futures)):
                futures.remove(future)
                batch_results = future.result()
                if batch_results:
                    results.extend(batch_results)
                
                batches_done += 1
                if batches_done % 5 == 0 or batch_results:
                     print(f"  ✓ Batch {batches_done} processed -> Total valid saved so far: {len(results)}/{total_n}")
                
                # Check if we need more
                if len(results) < total_n:
                    futures.add(executor.submit(generate_batch, model, category))
                break # Restart as_completed on updated futures set

    return results[:total_n] # Trim exactly to requested total

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    start_time = time.time()

    print(f"Model       : {args.model} (Local Ollama)")
    print(f"Target      : {args.total_per_class} per class ({args.total_per_class * 4} total)")
    print(f"Batch Size  : {args.batch_size} articles per prompt")
    print(f"Concurrency : {args.workers} parallel threads\n")

    all_data = []
    for category in CATEGORIES:
        examples = generate_for_category(args.model, category, args.total_per_class, args.batch_size, args.workers)
        all_data.extend(examples)

    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle
        df.to_csv(args.output, index=False)

        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"  Done in {elapsed/60:.1f} minutes!")
        print(f"  {len(df)} examples saved to: {args.output}")
        print(f"{'='*50}")
        print(df["label_name"].value_counts().to_string())
    else:
        print("\nFailed to generate any examples. Make sure Ollama is running.")

if __name__ == "__main__":
    main()
