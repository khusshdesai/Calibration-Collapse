import os
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
from calibration import compute_ece, plot_reliability_diagram

MODEL_NAME      = "distilbert-base-uncased"
NUM_LABELS      = 2
MAX_LENGTH      = 128
BATCH_SIZE      = 16
EPOCHS          = 3
LR              = 2e-5
SEED            = 42
TOTAL_TRAIN_N   = 10000
VAL_N           = 200       

SYNTH_FILE      = r"C:\Users\Lenovo\Downloads\gen ai\data\synthetic\synthetic_sst2_10k.csv"
RESULTS_FILE    = r"C:\Users\Lenovo\Downloads\gen ai\results\experiment_sst2_results.csv"

ID2LABEL = {0: "Negative", 1: "Positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

RATIOS = [
    (0.00, "0%  synthetic"),
    (0.25, "25% synthetic"),
    (0.50, "50% synthetic"),
    (0.75, "75% synthetic"),
    (1.00, "100% synthetic"),
]

print("Downloading and Loading Real SST-2 Dataset from HuggingFace …")
real_raw = load_dataset("glue", "sst2")

real_shuffled = real_raw["train"].shuffle(seed=SEED).to_pandas()[["sentence", "label"]]
real_shuffled = real_shuffled.rename(columns={"sentence": "text"})

real_val_df   = real_shuffled.iloc[:VAL_N].reset_index(drop=True)
real_train_pool = real_shuffled.iloc[VAL_N : VAL_N + TOTAL_TRAIN_N].reset_index(drop=True)

try:
    synth_df = pd.read_csv(SYNTH_FILE)[["text", "label"]].dropna()
    synth_df = synth_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
except FileNotFoundError:
    print(f"\nERROR: Could not find {SYNTH_FILE}")
    print("Execution explicitly halted. Please run generate_sst2.py first!\n")
    exit(1)

real_test_df = real_raw["validation"].to_pandas()[["sentence", "label"]]
real_test_df = real_test_df.rename(columns={"sentence": "text"})


print(f"  Real train pool : {len(real_train_pool):,} samples")
print(f"  Synthetic pool  : {len(synth_df):,} samples")
print(f"  Val set (for T) : {len(real_val_df):,} real samples")
print(f"  Test set        : {len(real_test_df):,} samples\n")

tokenizer       = AutoTokenizer.from_pretrained(MODEL_NAME)
accuracy_metric = evaluate.load("accuracy")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    return {"accuracy": acc}

def make_hf_dataset(df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds

print("Tokenising test and validation sets …")
test_ds = make_hf_dataset(real_test_df)
val_ds  = make_hf_dataset(real_val_df)

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

results_log = []

for synth_frac, label in RATIOS:
    n_synth = int(TOTAL_TRAIN_N * synth_frac)
    n_real  = TOTAL_TRAIN_N - n_synth

    print(f"\n{'='*60}")
    print(f"  Experiment: {label}")
    print(f"  Real : {n_real:,}  |  Synthetic : {n_synth:,}")
    print(f"{'='*60}")

    frames = []
    if n_real > 0:
        frames.append(real_train_pool.iloc[:n_real])
    if n_synth > 0:
        frames.append(synth_df.iloc[:n_synth])

    train_df = pd.concat(frames).sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_ds = make_hf_dataset(train_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    ratio_tag    = str(int(synth_frac * 100))
    output_dir   = f"./checkpoints/experiments_sst2/ratio_{ratio_tag}"
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
        logging_steps=50,
        seed=SEED,
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

    metrics = trainer.evaluate(test_ds)
    acc = metrics["eval_accuracy"]

    test_output  = trainer.predict(test_ds)
    test_logits  = test_output.predictions
    test_labels  = test_output.label_ids

    val_output   = trainer.predict(val_ds)
    val_logits   = val_output.predictions
    val_labels   = val_output.label_ids

    print(f"\n  --- ECE BEFORE Temperature Scaling ---")
    ece_before = compute_ece(test_logits, test_labels, n_bins=10, verbose=False)

    T = find_temperature(val_logits, val_labels)
    
    scaled_logits = test_logits / T
    ece_after = compute_ece(scaled_logits, test_labels, n_bins=10, verbose=False)

    print(f"\n  → SST-2 Acc: {acc:.4f} | ECE before: {ece_before:.4f} | ECE after: {ece_after:.4f} | T: {T:.3f}")
    results_log.append({
        "synth_ratio":  f"{int(synth_frac * 100)}%",
        "n_real":       n_real,
        "n_synthetic":  n_synth,
        "accuracy":     round(acc,        4),
        "ece_before":   round(ece_before, 4),
        "ece_after":    round(float(ece_after),  4),
        "temperature":  round(float(T),          4),
    })

    del model, trainer
    torch.cuda.empty_cache()

results_df = pd.DataFrame(results_log)
results_df.to_csv(RESULTS_FILE, index=False)

print(f"\n{'='*60}")
print("  ALL SST-2 EXPERIMENTS COMPLETE")
print(f"{'='*60}")
print(results_df.to_string(index=False))
