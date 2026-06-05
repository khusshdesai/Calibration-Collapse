"""
Label Smoothing Baseline Experiment
====================================
Trains a DistilBERT model on 100% synthetic AG News data
using Label Smoothing (factor=0.1) instead of post-hoc Temperature Scaling.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
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
MODEL_NAME    = "distilbert-base-uncased"
NUM_LABELS    = 4
MAX_LENGTH    = 128
BATCH_SIZE    = 16
EPOCHS        = 3
LR            = 2e-5
TOTAL_TRAIN_N = 10_000
VAL_N         = 200
SEED          = 42

SYNTH_FILE    = r"C:\Users\Lenovo\Downloads\gen ai\data\synthetic\synthetic_agnews_10k_fast.csv"
RESULTS_DIR   = r"C:\Users\Lenovo\Downloads\gen ai\results"

ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── GPU Info ──────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*60}")
print(f"  DEVICE: {device.upper()}")
print(f"  BASELINE: LABEL SMOOTHING (0.1)")
print(f"{'='*60}\n")

# ── 1. Load raw data ─────────────────────────────────────────────────────────
real_raw = load_dataset("ag_news")
real_test_df = real_raw["test"].to_pandas()[["text", "label"]]

try:
    synth_df = pd.read_csv(SYNTH_FILE)[["text", "label"]].dropna()
except FileNotFoundError:
    print(f"ERROR: Could not find {SYNTH_FILE}")
    exit(1)

accuracy_metric = evaluate.load("accuracy")

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_ds   = make_hf_dataset(real_test_df, tokenizer)

# We use 100% synthetic data for training
print("Preparing 100% Synthetic Data...")
synth_shuffled = synth_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
train_df = synth_shuffled.iloc[:TOTAL_TRAIN_N]
train_ds = make_hf_dataset(train_df, tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
)

warmup_steps = int(0.06 * (len(train_ds) / BATCH_SIZE) * EPOCHS)

# KEY ADDITION: label_smoothing_factor=0.1
args = TrainingArguments(
    output_dir="./checkpoints/ls_baseline",
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
    seed=SEED,
    report_to="none",
    label_smoothing_factor=0.1,  # <--- This is what reviewers want to see
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

print("\nStarting Label Smoothing Training...")
trainer.train()

print("\nEvaluating...")
metrics     = trainer.evaluate(test_ds)
acc         = metrics["eval_accuracy"]

test_output = trainer.predict(test_ds)
test_logits = test_output.predictions
test_labels = test_output.label_ids

ece = compute_ece(test_logits, test_labels, n_bins=10, verbose=False)

print(f"\n{'='*60}")
print(f"  LABEL SMOOTHING RESULTS (100% Synthetic)")
print(f"  Accuracy : {acc:.4f}")
print(f"  ECE      : {ece:.4f}")
print(f"{'='*60}\n")
