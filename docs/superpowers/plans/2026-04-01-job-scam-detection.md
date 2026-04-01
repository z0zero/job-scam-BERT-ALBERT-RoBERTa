# Job Scam Detection Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a job scam detection system with a research notebook comparing BERT/ALBERT/RoBERTa and a Streamlit app serving the best model.

**Architecture:** Monolithic notebook handles data loading, preprocessing, training 3 transformer models, evaluation, and export. Standalone Streamlit app loads the exported model and provides text/image input with OCR.

**Tech Stack:** PyTorch, HuggingFace Transformers, scikit-learn, pandas, matplotlib, seaborn, Streamlit, pytesseract, Pillow

**Spec:** `docs/superpowers/specs/2026-04-01-job-scam-detection-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | All pip dependencies with pinned versions |
| `research_pipeline.ipynb` | End-to-end research: data, EDA, preprocessing, train 3 models, evaluate, export best |
| `app.py` | Streamlit UI: load best model, accept text/image input, classify, display results |

---

### Task 1: Create `requirements.txt`

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Write requirements.txt**

```txt
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
pytesseract>=0.3.10
Pillow>=10.0.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements.txt with all project dependencies"
```

---

### Task 2: Create Research Notebook — Setup, Data Loading & EDA

**Files:**
- Create: `research_pipeline.ipynb`

This task creates the notebook with the first 3 sections: Setup & Config, Data Loading, and EDA.

- [ ] **Step 1: Create notebook with Cell 1 — Setup & Config**

Markdown cell with title, then code cell:

```python
# ============================================================
# Cell 1: Setup & Configuration
# ============================================================

import os
import re
import json
import html
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ----- Hyperparameters -----
CONFIG = {
    "seed": 42,
    "max_len": 256,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "test_size": 0.15,
    "val_size": 0.15,
}

# ----- Reproducibility -----
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

- [ ] **Step 2: Add Cell 2 — Data Loading**

```python
# ============================================================
# Cell 2: Data Loading
# ============================================================

# Load the EMSCAD dataset
# Download from: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
df = pd.read_csv("fake_job_postings.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()
```

- [ ] **Step 3: Add Cell 3 — Combine Text Fields**

```python
# ============================================================
# Cell 3: Combine Text Fields
# ============================================================

text_columns = ["title", "company_profile", "description", "requirements", "benefits"]

for col in text_columns:
    df[col] = df[col].fillna("")

df["text"] = df[text_columns].apply(lambda row: " ".join(row.values), axis=1)

print(f"Combined text column created.")
print(f"Sample text (first 300 chars): {df['text'].iloc[0][:300]}")
```

- [ ] **Step 4: Add Cell 4 — EDA**

```python
# ============================================================
# Cell 4: Exploratory Data Analysis
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Class Distribution
class_counts = df["fraudulent"].value_counts()
axes[0].bar(
    ["Legitimate (0)", "Fraudulent (1)"],
    class_counts.values,
    color=["#2ecc71", "#e74c3c"],
)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")

# 2. Text Length Distribution
df["text_length"] = df["text"].apply(len)
axes[1].hist(df["text_length"], bins=50, color="#3498db", edgecolor="black", alpha=0.7)
axes[1].set_title("Text Length Distribution (characters)")
axes[1].set_xlabel("Length")
axes[1].set_ylabel("Frequency")

# 3. Missing Values (from raw data)
df_raw = pd.read_csv("fake_job_postings.csv")
missing = df_raw[text_columns].isnull().sum()
axes[2].barh(text_columns, missing.values, color="#9b59b6")
axes[2].set_title("Missing Values per Text Column")
axes[2].set_xlabel("Count")

plt.tight_layout()
plt.show()

print(f"\nClass distribution:\n{df['fraudulent'].value_counts()}")
print(f"\nFraud percentage: {df['fraudulent'].mean() * 100:.2f}%")
print(f"\nText length stats:\n{df['text_length'].describe()}")
```

- [ ] **Step 5: Commit**

```bash
git add research_pipeline.ipynb
git commit -m "feat: add research notebook with setup, data loading, and EDA"
```

---

### Task 3: Research Notebook — Preprocessing & Dataset Class

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add Cell 5 — Text Cleaning**

```python
# ============================================================
# Cell 5: Text Preprocessing
# ============================================================

def clean_text(text):
    """Clean and normalize text for model input."""
    text = html.unescape(text)                          # Decode HTML entities
    text = re.sub(r"<[^>]+>", " ", text)                # Strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # Strip URLs
    text = text.lower()                                  # Lowercase
    text = re.sub(r"\s+", " ", text).strip()            # Collapse whitespace
    return text

df["text"] = df["text"].apply(clean_text)

print("Text cleaning complete.")
print(f"Sample cleaned text: {df['text'].iloc[0][:300]}")
```

- [ ] **Step 2: Add Cell 6 — Train/Val/Test Split**

```python
# ============================================================
# Cell 6: Stratified Train / Validation / Test Split
# ============================================================

train_val_df, test_df = train_test_split(
    df,
    test_size=CONFIG["test_size"],
    stratify=df["fraudulent"],
    random_state=CONFIG["seed"],
)

val_fraction = CONFIG["val_size"] / (1 - CONFIG["test_size"])
train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_fraction,
    stratify=train_val_df["fraudulent"],
    random_state=CONFIG["seed"],
)

print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

print(f"\nFraud ratio - Train: {train_df['fraudulent'].mean():.4f}")
print(f"Fraud ratio - Val:   {val_df['fraudulent'].mean():.4f}")
print(f"Fraud ratio - Test:  {test_df['fraudulent'].mean():.4f}")
```

- [ ] **Step 3: Add Cell 7 — Compute Class Weights**

```python
# ============================================================
# Cell 7: Compute Class Weights for Imbalanced Data
# ============================================================

class_counts = train_df["fraudulent"].value_counts().sort_index()
total = len(train_df)
class_weights = torch.tensor(
    [total / (2 * class_counts[0]), total / (2 * class_counts[1])],
    dtype=torch.float32,
).to(device)

print(f"Class weights: {class_weights}")
print(f"  Legitimate (0): {class_weights[0]:.4f}")
print(f"  Fraudulent (1): {class_weights[1]:.4f}")
```

- [ ] **Step 4: Add Cell 8 — PyTorch Dataset Class**

```python
# ============================================================
# Cell 8: Custom PyTorch Dataset
# ============================================================

class JobPostingDataset(Dataset):
    """PyTorch Dataset for job posting text classification."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

print("JobPostingDataset class defined.")
```

- [ ] **Step 5: Commit**

```bash
git add research_pipeline.ipynb
git commit -m "feat: add preprocessing, stratified split, and dataset class"
```

---

### Task 4: Research Notebook — Training Infrastructure & Model Training

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add Cell 9 — Metrics & Custom Trainer**

```python
# ============================================================
# Cell 9: Metrics Function & Custom Weighted-Loss Trainer
# ============================================================

def compute_metrics(pred):
    """Compute accuracy, precision, recall, and F1 for HF Trainer."""
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


class WeightedTrainer(Trainer):
    """Custom Trainer that uses weighted cross-entropy loss for class imbalance."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

print("Metrics function and WeightedTrainer defined.")
```

- [ ] **Step 2: Add Cell 10 — Training Loop for All 3 Models**

```python
# ============================================================
# Cell 10: Train BERT, ALBERT, and RoBERTa
# ============================================================

MODEL_NAMES = {
    "BERT": "bert-base-uncased",
    "ALBERT": "albert-base-v2",
    "RoBERTa": "roberta-base",
}

results = {}
trained_models = {}

for model_label, model_name in MODEL_NAMES.items():
    print(f"\n{'='*60}")
    print(f"Training: {model_label} ({model_name})")
    print(f"{'='*60}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # Create datasets
    train_dataset = JobPostingDataset(
        train_df["text"], train_df["fraudulent"], tokenizer, CONFIG["max_len"]
    )
    val_dataset = JobPostingDataset(
        val_df["text"], val_df["fraudulent"], tokenizer, CONFIG["max_len"]
    )
    test_dataset = JobPostingDataset(
        test_df["text"], test_df["fraudulent"], tokenizer, CONFIG["max_len"]
    )

    # Training arguments
    output_dir = f"./training_output/{model_label.lower()}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_ratio=CONFIG["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=CONFIG["seed"],
    )

    # Create trainer with weighted loss
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate on test set
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_df["fraudulent"].values

    metrics = {
        "accuracy": accuracy_score(test_labels, test_preds),
        "precision": precision_score(test_labels, test_preds, zero_division=0),
        "recall": recall_score(test_labels, test_preds, zero_division=0),
        "f1": f1_score(test_labels, test_preds, zero_division=0),
    }

    results[model_label] = metrics
    trained_models[model_label] = {
        "model": trainer.model,
        "tokenizer": tokenizer,
        "predictions": test_preds,
    }

    print(f"\n{model_label} Test Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
```

- [ ] **Step 3: Commit**

```bash
git add research_pipeline.ipynb
git commit -m "feat: add training infrastructure and model training loop"
```

---

### Task 5: Research Notebook — Comparison & Model Export

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add Cell 11 — Comparison Matrix & Bar Chart**

```python
# ============================================================
# Cell 11: Comparative Evaluation Matrix
# ============================================================

comparison_df = pd.DataFrame(results).T
comparison_df.index.name = "Model"
print("=" * 60)
print("COMPARATIVE EVALUATION MATRIX")
print("=" * 60)
print(comparison_df.round(4).to_string())
print()

# Grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison_df.columns))
width = 0.25
models = list(comparison_df.index)
colors = ["#3498db", "#e74c3c", "#2ecc71"]

for i, model_name in enumerate(models):
    values = comparison_df.loc[model_name].values
    ax.bar(x + i * width, values, width, label=model_name, color=colors[i])

ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_title("Model Comparison - Test Set Metrics")
ax.set_xticks(x + width)
ax.set_xticklabels(comparison_df.columns)
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Add Cell 12 — Confusion Matrices**

```python
# ============================================================
# Cell 12: Confusion Matrices
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
test_labels = test_df["fraudulent"].values

for idx, (model_label, model_data) in enumerate(trained_models.items()):
    cm = confusion_matrix(test_labels, model_data["predictions"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraudulent"],
        yticklabels=["Legitimate", "Fraudulent"],
        ax=axes[idx],
    )
    axes[idx].set_title(f"{model_label} - Confusion Matrix")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Add Cell 13 — Export Best Model**

```python
# ============================================================
# Cell 13: Export Best Model
# ============================================================

best_model_name = max(results, key=lambda k: results[k]["f1"])
best_metrics = results[best_model_name]

print(f"Best model: {best_model_name} (F1: {best_metrics['f1']:.4f})")

# Save model and tokenizer
save_dir = "./best_model"
os.makedirs(save_dir, exist_ok=True)

best_model_obj = trained_models[best_model_name]["model"]
best_tokenizer = trained_models[best_model_name]["tokenizer"]

best_model_obj.save_pretrained(save_dir)
best_tokenizer.save_pretrained(save_dir)

# Save metadata
meta = {
    "model_name": best_model_name,
    "hf_model_id": MODEL_NAMES[best_model_name],
    "metrics": best_metrics,
    "config": CONFIG,
    "exported_at": datetime.now().isoformat(),
}

with open(os.path.join(save_dir, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nModel saved to: {save_dir}/")
print(f"Metadata saved to: {save_dir}/model_meta.json")

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"Best Model: {best_model_name} ({MODEL_NAMES[best_model_name]})")
for metric, value in best_metrics.items():
    print(f"  {metric}: {value:.4f}")
```

- [ ] **Step 4: Commit**

```bash
git add research_pipeline.ipynb
git commit -m "feat: add model comparison, confusion matrices, and model export"
```

---

### Task 6: Create Streamlit App

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write the complete `app.py`**

```python
"""
Job Scam Detection - Streamlit Web Application

Loads the best-performing NLP model exported from the research notebook
and provides a UI for classifying job postings as legitimate or fraudulent.
Supports text input and image upload (with OCR via pytesseract).
"""

import json
import os

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pytesseract


# ----- Configuration -----
MODEL_DIR = "./best_model"
MAX_LEN = 256


@st.cache_resource
def load_model():
    """Load the saved model, tokenizer, and metadata."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, tokenizer, device, meta


def classify_text(text, model, tokenizer, device):
    """Classify a job posting text and return label + confidence."""
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()

    label = "Potential Scam" if predicted_class == 1 else "Legitimate Job"
    return label, confidence


def extract_text_from_image(image):
    """Extract text from an uploaded image using OCR."""
    return pytesseract.image_to_string(image)


def main():
    st.set_page_config(
        page_title="Job Scam Detector",
        page_icon="shield",
        layout="centered",
    )

    st.title("Job Scam Detector")
    st.markdown(
        "Analyze job postings to detect potential scams using NLP. "
        "Paste a job description or upload a screenshot."
    )

    # Load model
    try:
        model, tokenizer, device, meta = load_model()
        st.info(
            f"**Model:** {meta['model_name']} ({meta['hf_model_id']}) | "
            f"F1: {meta['metrics']['f1']:.4f} | "
            f"Accuracy: {meta['metrics']['accuracy']:.4f}"
        )
    except Exception as e:
        st.error(
            f"Failed to load model from `{MODEL_DIR}/`. "
            f"Run the research notebook first to export a model.\n\nError: {e}"
        )
        return

    # Input mode
    st.subheader("Input")
    input_mode = st.radio(
        "Choose input method:",
        ["Paste Text", "Upload Image"],
        horizontal=True,
    )

    text = ""

    if input_mode == "Paste Text":
        text = st.text_area(
            "Paste the job description below:",
            height=250,
            placeholder="Enter or paste the full job posting text here...",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a screenshot of the job posting:",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Extracting text with OCR..."):
                extracted = extract_text_from_image(image)

            text = st.text_area(
                "Extracted text (edit if needed):",
                value=extracted,
                height=250,
            )

    # Classification
    st.subheader("Result")

    if st.button("Analyze", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please provide a job description to analyze.")
        else:
            with st.spinner("Analyzing..."):
                label, confidence = classify_text(text, model, tokenizer, device)

            if label == "Legitimate Job":
                st.success(f"**{label}**")
            else:
                st.error(f"**{label}**")

            st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit web app with OCR support"
```

---

### Task 7: Final Integration Commit

- [ ] **Step 1: Verify all files exist**

```bash
ls -la requirements.txt research_pipeline.ipynb app.py
```

Expected: all 3 files present.

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "feat: complete job scam detection project"
```
