# Banking Intent Classifier

An end-to-end AI engineering project that classifies customer banking queries into 77 intent categories using transformer-based deep learning. Built as a capstone project while learning AI and ML engineering fundamentals for codeacademy.

---

## Overview

When a customer contacts their bank and types something like "my card hasn't arrived yet" or "I was charged twice for the same transaction", a bank's support system needs to figure out what the customer actually wants so it can route them to the right place automatically. That's intent classification.

This project builds that system by training and comparing two models on the Banking77 dataset and wrapping the best model in a production ready inference pipeline with PII protection.

---

## Results

| Model | Accuracy | Macro F1 | Trainable Params | Train Time |
| ------- | ---------- | ---------- | ----------------- | ------------ |
| MLP Baseline | 88.9% | 0.889 | 5.27M | ~2 min |
| LoRA-RoBERTa | 92.5% | 0.925 | 1.24M | ~5 min |

RoBERTa improved on 61 of 77 intent classes. The biggest gains were on semantically similar intents that share surface vocabulary but differ in meaning.

---

## Project Structure

```text
banking-intent-classifier/
│
├── banking_classification_BERT.ipynb   # main project notebook
├── requirements.txt                    # project dependencies
│
├── datasets/
    ├── banking77_train.csv             # 10,003 training queries
    └── banking77_test.csv             # 3,080 test queries

```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- Google Colab with T4 GPU (recommended) or a local GPU machine

### Clone the repository

```bash
git clone https://github.com/ovesa/banking-intent-classifier.git
cd banking-intent-classifier
```

### Install dependencies

```bash
pip install -r requirements.txt
pip install peft
```

### Run the notebook

Open `banking_classification_BERT.ipynb` in Google Colab or Jupyter and run cells from top to bottom. The first cell handles cloning and dependency installation automatically.

---

## Quick Inference Demo

```python
result = predict_intent(
    "My card was charged twice for the same transaction",
    model=lora_model,
    tokenizer=tokenizer,
    label_encoder=label_encoder,
    device=device
)

print(result['predicted_intent'])  # transaction_charged_twice
print(result['confidence'])        # 0.994
```

---

## Tech Stack

| Tool | Purpose |
| ------ | --------- |
| PyTorch | Model training and inference |
| Hugging Face Transformers | RoBERTa model and tokenizer |
| PEFT | LoRA configuration and adapter training |
| scikit-learn | TF-IDF vectorization, label encoding, evaluation metrics |
| pandas and numpy | Data loading and manipulation |
| matplotlib | EDA visualizations |

---

## Dataset

Banking77 dataset obtained from codeacademy for the `Classifying Banking Intent From Customer Queries Capstone Project`
