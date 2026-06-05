<div align="center">
  <h1>📉 Calibration Collapse</h1>
  <p><b>Diagnosing and Correcting Synthetic-Induced Overconfidence in Transformer Classifiers</b></p>
</div>

> **Disclaimer:** This repository, the code, and the accompanying synthetic datasets are intended strictly for **academic and research purposes**.

An empirical machine learning research project demonstrating how LLM-generated synthetic training data degrades the probabilistic calibration of downstream transformer classifiers, and introducing a novel validation-free mitigation strategy.

---

## 📖 Overview
**What problem the project solves:** 
When organizations use Large Language Models (LLMs) to generate synthetic training data, downstream classifiers successfully learn to classify but lose the ability to accurately estimate their own confidence. They become systematically overconfident—a phenomenon termed **Calibration Collapse**.

**Why it exists:** 
To prove that traditional calibration techniques (like Temperature Scaling) fail at 100% synthetic data extremes, and to provide a new heuristic method (SATS) that calibrates mixed-data models without requiring costly human-labeled validation data.

**Who it is intended for:** 
AI researchers, MLOps engineers, and data scientists utilizing synthetic datasets for training classification models in production environments where probabilistic reliability is critical.

**Key business value:** 
Prevents "confidently incorrect" predictions in safety-critical applications (e.g., medical triage, automated moderation) by restoring accurate confidence scores to synthetic-trained models.

## ✨ Features
### 1. Calibration Collapse Diagnosis Pipeline
- **What it does:** Measures the Expected Calibration Error (ECE) of models trained on varying ratios of synthetic-to-real data.
- **How it works internally:** Trains DistilBERT and RoBERTa classifiers across 5 discrete synthetic ratios (0%, 25%, 50%, 75%, 100%) and evaluates logit distributions.
- **Relevant files:** `src/experiment.py`, `src/experiment_roberta.py`

### 2. Synthetic-Aware Temperature Scaling (SATS)
- **What it does:** Automatically calculates the optimal Temperature Scaling parameter ($T$) without needing a labeled validation set.
- **How it works internally:** Computes the mean magnitude of unscaled test logits and maps it through a fitted linear regression model ($T = \alpha \times X + \beta$) to reverse the logit inflation caused by synthetic data.
- **Relevant files:** `src/sats.py`, `src/run_sats_analysis.py`

### 3. Cross-Domain Generative Pipeline
- **What it does:** Generates synthetic datasets to avoid generator monoculture bias.
- **How it works internally:** Utilizes LLaMA 3.2 (for AG News topic classification) and Gemma 2 (for SST-2 sentiment analysis) to synthesize training samples matching real-world class distributions.
- **Relevant files:** `src/generate_data.py`, `src/generate_sst2.py`

## 📊 The Phenomenon: Calibration Collapse
As the proportion of synthetic data increases, models learn to inflate logit magnitudes rather than robust decision boundaries, largely due to the lack of long-tail linguistic diversity in LLM-generated text. This destroys probabilistic calibration.

### Reliability Diagrams (DistilBERT)
*Ideal calibration follows the diagonal line. Predictions below the line indicate overconfidence.*

| 0% Synthetic (Real Baseline) | 100% Synthetic (Uncalibrated) | 100% Synthetic (Post-Scaling) |
|:---:|:---:|:---:|
| <img src="results/figures/before_0pct.png" width="300" /> | <img src="results/figures/before_100pct.png" width="300" /> | <img src="results/figures/after_100pct.png" width="300" /> |
| **ECE: 4.96%** | **ECE: 17.10%** | **ECE: 8.42%** |

### Key Findings (DistilBERT)

| Synthetic Ratio | Accuracy | Uncalibrated ECE | Global Temp ($T$) | Post-Scaling ECE |
|:---:|:---:|:---:|:---:|:---:|
| 0% (Real) | 91.9% | 4.96% | 1.43 | 2.04% |
| 25% | 91.4% | 5.31% | 1.46 | 2.01% |
| 50% | 90.7% | 6.00% | 1.49 | 2.08% |
| 75% | 89.7% | 6.82% | 1.54 | **1.80%** |
| 100% | 78.4% | **17.10%** | 1.86 | 8.42% |

*Notice the massive spike in overconfidence at 100% synthetic training. While Temperature Scaling improves calibration beautifully in mixed regimes (achieving an excellent 1.80% ECE at a 75% synthetic mix), a residual 8.42% error persists at the 100% extreme. This proves that fully synthetic regimes inflict irreversible structural damage to the logit space.*

## 🚀 Synthetic-Aware Temperature Scaling (SATS)
We propose SATS to achieve validation-free calibration. By mathematically modeling the linear logit inflation caused by synthetic data variance reduction, SATS predicts the optimal Temperature Scaling parameter ($T$) directly from unscaled test-time logit statistics. 

<div align="center">
  <img src="results/figures/sats_scatter.png" width="450" />
</div>

## 🏗 Architecture & Data Flow
Because this is an ML research pipeline, the architecture is designed around reproducibility and sequential data transformation.

1. **Generation Layer:** `generate_data.py` interfaces with LLMs to output synthetic `.csv` files into the `data/synthetic/` directory.
2. **Training Layer:** `experiment_multiseed.py` orchestrates training loops across multiple random seeds, fine-tuning HuggingFace models on dynamic mixes of real/synthetic data.
3. **Inference & Logit Extraction Layer:** Raw logits (before softmax) are saved as `.npy` arrays into `results/logits/` to decouple slow neural network inference from fast calibration analysis.
4. **Calibration Layer:** `sats.py` ingests the raw `.npy` logits, computes statistics (mean magnitude, variance, entropy), fits the SATS regression, and outputs scaled ECE metrics.

## 🔬 Technical Deep Dive
### SATS Algorithm
Traditional Temperature Scaling requires a labeled validation dataset to minimize Negative Log-Likelihood (NLL) via L-BFGS optimization. 

Our core abstraction, `SATSCalibrator`, bypasses this:
1. `compute_logit_stats()` extracts the absolute mean magnitude of raw test logits.
2. `fit()` establishes the linear relationship between this magnitude and the optimal $T$ across known mixed regimes.
3. `predict_temperature()` applies this scalar division mathematically, restoring the underlying entropic shape of the probability distribution zero-shot.

### Error Handling & Statistical Rigor
To ensure results are not statistical anomalies, experiments are wrapped in multi-seed runners (`experiment_multiseed.py`) that average findings across 3 distinct random initializations. 

## 💻 Technology Stack

| Category | Technology | Why it was chosen |
|---|---|---|
| **Deep Learning** | PyTorch | Core tensor operations and gradient calculations. |
| **NLP Models** | HuggingFace Transformers | Provides pre-trained DistilBERT and RoBERTa architectures. |
| **Data Processing** | NumPy, Pandas | Efficient handling of logit arrays (`.npy`) and dataset structures (`.csv`, `.parquet`). |
| **Evaluation** | Scikit-Learn | Accuracy metrics and standard statistical functions. |
| **Generators** | LLaMA 3.2, Gemma 2 | Open-weights models ensuring the collapse is not isolated to a single LLM family. |

## 📂 Folder Structure
```text
Calibration-Collapse/
├── src/
│   ├── sats.py                      # Core SATS calibration heuristic logic
│   ├── experiment_*.py              # Pipeline runners for different models
│   ├── generate_*.py                # Synthetic dataset generation scripts
│   └── calibration.py               # ECE calculation utilities
├── data/
│   ├── real/                        # Baseline benchmark datasets (AG News, SST-2)
│   └── synthetic/                   # LLM-generated training data
├── results/
│   ├── figures/                     # Generated reliability diagrams (.png)
│   ├── logits/                      # Raw unscaled model outputs (.npy)
│   └── *_results.csv                # Aggregated experimental metrics
└── README.md                        # Project documentation
```

## 🔒 Security Measures
As a local machine learning pipeline, standard network security controls (CORS, CSRF, JWT) do not apply. 
- **Secrets Management:** The repository relies on local `.env` files (excluded via `.gitignore`) for any API keys needed during the data generation phase.
- **Model Checkpoints:** Saved model weights (`checkpoints/`) are explicitly `.gitignore`d to prevent accidental leakage of proprietary fine-tunes and to respect GitHub's storage limits.

## 🚀 Deployment & Local Setup
### Environment Requirements
- Python 3.10+
- CUDA-compatible GPU (Highly recommended for fine-tuning)

### Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/khusshdesai/Calibration-Collapse.git
cd Calibration-Collapse

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the multi-seed experiment suite
python src/experiment_multiseed.py
```

## 🚧 Challenges & Engineering Decisions
**Decision:** Decoupling Inference from Calibration
* *Alternative:* Calculating ECE directly during the PyTorch training loop.
* *Tradeoff:* We chose to save raw logits as massive `.npy` arrays instead. While this consumes disk space, it allows the `SATSCalibrator` to run iterative experiments in milliseconds on CPU, rather than requiring the GPU to re-run inference for every calibration test.

**Decision:** Rejecting Vector Scaling
* *Alternative:* Using per-class temperature vectors instead of a global scalar.
* *Tradeoff:* We implemented Vector Scaling but discovered it offered 0% improvement over Global Scaling at the 100% synthetic extreme. This led to the fundamental engineering conclusion that purely synthetic datasets inflict complex, non-linear structural damage that linear scaling cannot fix.

## 🔮 Future Improvements
- **Short-term:** Expand the `sats.py` pipeline to support Generative LLMs (e.g., evaluating calibration on token generation rather than sequence classification).
- **Medium-term:** Create a PyPI package for `sats-calibrator` to allow developers to drop it into existing PyTorch pipelines.
- **Long-term:** Design novel loss constraints (e.g., Synthetic Penalty Loss) applied during training to mathematically prevent logit inflation before it happens.

## 🌟 Resume-Worthy Highlights
- **Architectural Depth:** Designed a completely decoupled ML evaluation pipeline separating heavy tensor inference from fast heuristic calibration analysis.
- **Mathematical Innovation:** Formulated and implemented a novel validation-free calibration heuristic (SATS) that predicts Temperature Scaling parameters with $\Delta T < 0.06$ error.
- **Empirical Rigor:** Managed thousands of model training epochs across 5 data ratios, 2 architectures, 2 dataset domains, and 3 random seeds to prove a systemic flaw in modern AI data practices.

## 🤝 Credits
- **Khussh Desai** - Lead Researcher & Engineer
- **Joel Kallarakal** - Co-Researcher

*(Note: Synthetic datasets provided were generated using LLaMA 3.2 and Gemma 2. Their usage is subject to the respective Meta and Google Acceptable Use Policies.)*
<br>
&copy; 2026 Khussh Desai. All Rights Reserved.
