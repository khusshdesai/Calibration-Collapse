# Calibration Collapse: Diagnosing and Correcting Synthetic-Induced Overconfidence

Models trained on LLM-generated synthetic data suffer from systematic overconfidence, a phenomenon we term **Calibration Collapse**. This repository contains the code and experiments demonstrating this effect across DistilBERT and RoBERTa classifiers. 

It also includes the implementation of **Synthetic-Aware Temperature Scaling (SATS)**, a heuristic method that corrects this logit inflation without requiring human-labeled validation data.

## Highlights
- **Diagnosed the limits of synthetic training:** Proved that while mixed regimes (up to 75% synthetic data) can be calibrated, 100% synthetic training induces irreversible structural logit imbalances.
- **SATS Heuristic:** A novel approach to dynamically predict optimal Temperature Scaling parameters directly from unscaled logit variance, achieving $\Delta T < 0.06$ prediction error.
- **Reproducible Pipeline:** Complete code for data generation (LLaMA 3.2 / Gemma 2), model fine-tuning, and evaluating Expected Calibration Error (ECE).

## Repository Structure
- `src/`: Core Python modules for training, calibration (SATS), and experimentation.
- `data/`: Sample synthetic datasets used for training and evaluation.
- `results/`: Output CSVs, logs, and reliability diagrams.
- `test.py`: Entry point for testing components.

## Setup
```bash
pip install -r requirements.txt
```

## License
MIT License. 
*(Note: Synthetic datasets provided were generated using LLaMA 3.2 and Gemma 2, and their use is subject to the respective Meta and Google terms of service.)*
