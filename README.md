# table-tennis-prediction

## 1. Project Overview

This project implements a multi-task sequence modeling framework based on a Transformer encoder in PyTorch.

The model learns from sequential rally-level data and predicts:
- actionId (multi-class classification)
- pointId (multi-class classification)
- serverGetPoint (binary classification)

The system is designed for reproducibility, debugging, and model retraining.

---

## 2. Environment

- Operating System: Ubuntu 24.04.4 LTS
- Python: 3.8+

---

## 3. Dependencies

Required packages:
- argparse
- random
- numpy
- pandas
- torch
- sklearn

Installation:
pip install numpy pandas torch scikit-learn

---

## 4. Dataset Files

Place the following files in the same directory:
- train.csv
- test.csv
- sample_submission.csv
- baseline code.py

---

## 5. How to Run

python3 baseline code.py

Output:
submission_lstm_baseline_epochs350.csv

---

## 6. Output Format

rally_uid, actionId, pointId, serverGetPoint

---

## 7. Reproducibility

- Seed fixed at 42 (random, numpy, torch)
- Match-based split
- Deterministic encoding
