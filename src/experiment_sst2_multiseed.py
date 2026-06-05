"""
Multi-Seed SST-2 Experiment
============================
Mirrors experiment_multiseed.py exactly but for SST-2 / Gemma-2 synthetic data.
Runs DistilBERT across 3 seeds (42, 123, 456) at 5 synthetic ratios.

Outputs:
  - results/multiseed_sst2_all.csv
  - results/multiseed_sst2_summary.csv  (mean ± std aggregation)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import compute_ece

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}
NUM_LABELS    = 2
MAX_LENGTH    = 128
BATCH_SIZE    = 16
EPOCHS        = 3
LR            = 2e-5
TOTAL_TRAIN_N = 10_000
VAL_N         = 200

SEEDS = [42, 123, 456]

SYNTH_FILE   = r"C:\Users\Lenovo\Downloads\gen ai\data\synthetic\synthetic_sst2_10k.csv"
RESULTS_DIR  = r"C:\Users\Lenovo\Downloads\gen ai\results"
LOGITS_DIR   = os.path.join(RESULTS_DIR, "logits")

ID2LABEL = {0: "Negative", 1: "Positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

RATIOS = [
    (0.00, "0%"),
    (0.25, "25%"),
    (0.50, "50%"),
    (0.75, "75%"),
    (1.00, "100%"),
]

os.makedirs(LOGITS_DIR, exist_ok=True)

# ── GPU Info ──────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*60}")
print(f"  DEVICE: {device.upper()}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  SEEDS: {SEEDS}")
print(f"  MODELS: {list(MODELS.keys())}")
print(f"{'='*60}\n")


# ── 1. Load raw data ─────────────────────────────────────────────────────────
print("Loading SST-2 dataset from HuggingFace ...")
real_raw = load_dataset("glue", "sst2")

# SST-2 uses "validation" split as test (no ground-truth test labels available)
real_test_df = real_raw["validation"].to_pandas()[["sentence", "label"]]
real_test_df = real_test_df.rename(columns={"sentence": "text"})

try:
    synth_df = pd.read_csv(SYNTH_FILE)[["text", "label"]].dropna()
    print(f"  Synthetic pool loaded: {len(synth_df):,} samples")
except FileNotFoundError:
    print(f"\nERROR: Could not find {SYNTH_FILE}")
    print("Run generate_sst2.py first!\n")
    exit(1)

accuracy_metric = evaluate.load("accuracy")


# ── 2. Global Temperature Scaling ────────────────────────────────────────────
def find_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    logits_t = torch.FloatTensor(val_logits)
    labels_t = torch.LongTensor(val_labels)
    T = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([T], lr=0.01, max_iter=100)

    def eval_step():
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(logits_t / T.clamp(min=0.05), labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    return float(T.clamp(min=0.05).item())


# ── 3. Helper functions ──────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    return {"accuracy": acc}


def make_hf_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
    ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


# ── 4. Main experiment loop ──────────────────────────────────────────────────
all_results = []

for model_key, model_name in MODELS.items():
    print(f"\n{'#'*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'#'*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_ds   = make_hf_dataset(real_test_df, tokenizer)

    for seed in SEEDS:
        print(f"\n  -- SEED: {seed} --")

        # Shuffle training pool with this seed
        real_shuffled   = real_raw["train"].shuffle(seed=seed).to_pandas()[["sentence", "label"]]
        real_shuffled   = real_shuffled.rename(columns={"sentence": "text"})
        real_val_df     = real_shuffled.iloc[:VAL_N].reset_index(drop=True)
        real_train_pool = real_shuffled.iloc[VAL_N : VAL_N + TOTAL_TRAIN_N].reset_index(drop=True)
        synth_shuffled  = synth_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        val_ds = make_hf_dataset(real_val_df, tokenizer)

        for synth_frac, ratio_label in RATIOS:
            n_synth = int(TOTAL_TRAIN_N * synth_frac)
            n_real  = TOTAL_TRAIN_N - n_synth

            print(f"\n    {model_key} | seed={seed} | {ratio_label} synthetic "
                  f"({n_real:,} real + {n_synth:,} synthetic)")

            frames = []
            if n_real > 0:
                frames.append(real_train_pool.iloc[:n_real])
            if n_synth > 0:
                frames.append(synth_shuffled.iloc[:n_synth])
    
            train_df = pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)
            train_ds = make_hf_dataset(train_df, tokenizer)
    
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
            )
    
            ratio_tag    = str(int(synth_frac * 100))
            output_dir   = f"./checkpoints/sst2_{model_key}/seed{seed}/ratio_{ratio_tag}"
            warmup_steps = int(0.06 * (len(train_ds) / BATCH_SIZE) * EPOCHS)
    
            args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE * 2,
                learning_rate=LR,
                weight_decay=0.01,
                warmup_steps=warmup_steps,
                fp16=torch.cuda.is_available(),
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=200,
                seed=seed,
                report_to="none",
            )
    
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics,
            )
    
            trainer.train()
    
            metrics     = trainer.evaluate(test_ds)
            acc         = metrics["eval_accuracy"]
    
            test_output = trainer.predict(test_ds)
            test_logits = test_output.predictions
            test_labels = test_output.label_ids
    
            val_output  = trainer.predict(val_ds)
            val_logits  = val_output.predictions
            val_labels  = val_output.label_ids
    
            ece_before = compute_ece(test_logits, test_labels, n_bins=10, verbose=False)
            T_global   = find_temperature(val_logits, val_labels)
            ece_global = compute_ece(test_logits / T_global, test_labels, n_bins=10, verbose=False)
    
            # Save logits for reproducibility
            np.save(os.path.join(LOGITS_DIR, f"sst2_{model_key}_s{seed}_{ratio_tag}_test_logits.npy"), test_logits)
            np.save(os.path.join(LOGITS_DIR, f"sst2_{model_key}_s{seed}_{ratio_tag}_test_labels.npy"), test_labels)
    
            print(f"    -> Acc={acc:.4f} | ECE_before={ece_before:.4f} | "
                  f"T_global={T_global:.3f} | ECE_global={ece_global:.4f}")
    
            all_results.append({
                "model":       model_key,
                "seed":        seed,
                "synth_ratio": ratio_label,
                "n_real":      n_real,
                "n_synthetic": n_synth,
                "accuracy":    round(acc,             4),
                "ece_before":  round(float(ece_before), 4),
                "T_global":    round(float(T_global),   4),
                "ece_global":  round(float(ece_global), 4),
            })
    
            del model, trainer
            torch.cuda.empty_cache()


# ── 5. Save all results ──────────────────────────────────────────────────────
all_df = pd.DataFrame(all_results)
all_df.to_csv(os.path.join(RESULTS_DIR, "multiseed_sst2_all.csv"), index=False)

# ── 6. Compute mean ± std summary ────────────────────────────────────────────
summary_rows = []
for model_key in MODELS:
    for ratio_label in [r[1] for r in RATIOS]:
        subset = all_df[(all_df["model"] == model_key) & (all_df["synth_ratio"] == ratio_label)]
        summary_rows.append({
            "model":             model_key,
            "synth_ratio":       ratio_label,
            "accuracy_mean":     round(subset["accuracy"].mean(),    4),
            "accuracy_std":      round(subset["accuracy"].std(),     4),
            "ece_before_mean":   round(subset["ece_before"].mean(),  4),
            "ece_before_std":    round(subset["ece_before"].std(),   4),
            "T_global_mean":     round(subset["T_global"].mean(),    4),
            "T_global_std":      round(subset["T_global"].std(),     4),
            "ece_global_mean":   round(subset["ece_global"].mean(),  4),
            "ece_global_std":    round(subset["ece_global"].std(),   4),
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(RESULTS_DIR, "multiseed_sst2_summary.csv"), index=False)

print(f"\n{'='*60}")
print("  ALL SST-2 MULTI-SEED EXPERIMENTS COMPLETE")
print(f"{'='*60}\n")
print(summary_df.to_string(index=False))
print(f"\nFull results : {os.path.join(RESULTS_DIR, 'multiseed_sst2_all.csv')}")
print(f"Summary      : {os.path.join(RESULTS_DIR, 'multiseed_sst2_summary.csv')}")
