# IMDB Sentiment Analysis — BiRNN (PyTorch)

A deep learning project that implements a **Bidirectional LSTM (BiRNN)** for binary sentiment classification on the IMDB movie review dataset, compared against a Logistic Regression + TF-IDF baseline.

## Overview

| | |
|---|---|
| **Dataset** | IMDB (25,000 train / 25,000 test reviews) |
| **Model** | Bidirectional LSTM (PyTorch) |
| **Baseline** | Logistic Regression + TF-IDF (sklearn) |
| **Sequence Length** | 200 tokens (padded / truncated) |
| **Vocabulary Size** | 30,123 (min_freq=5) |
| **Epochs** | 5 |

## Results

|  | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **BiRNN (from scratch)** | 84.05% | 83.03% | 85.59% | 84.29% |
| **Logistic Regression + TF-IDF** | 88.36% | 88.42% | 88.29% | 88.36% |

## Model Architecture

```
Embedding(vocab_size, 100)
  → Bidirectional LSTM(100 hidden, 1 layer)
  → Concat(first_output, last_output)  # 4 × 100 = 400
  → Linear(400, 1)
  → BCEWithLogitsLoss
```

## Output Graphs

- `training_result.png` — Loss curve, Train/Test accuracy per epoch, Metrics comparison
- `confusion_matrices.png` — Confusion matrix for BiRNN and baseline

## Requirements

```bash
pip install torch torchtext scikit-learn matplotlib
```

## Dataset Setup

Download [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and place it at:
```
data/
└── aclImdb/
    ├── train/
    │   ├── pos/
    │   └── neg/
    └── test/
        ├── pos/
        └── neg/
```

## Run

```bash
python main.py
```
