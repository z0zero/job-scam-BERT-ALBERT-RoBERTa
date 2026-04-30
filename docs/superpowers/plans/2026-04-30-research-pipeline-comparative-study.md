# Research Pipeline Comparative Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `research_pipeline.ipynb` into a resumable, paper-oriented comparative study pipeline for BERT, ALBERT, and RoBERTa on EMSCAD.

**Architecture:** Keep one primary research notebook, but organize it around reusable functions for configuration, data preparation, splitting, training, artifact persistence, aggregation, statistical testing, visualization, error analysis, and model export. Per-run artifacts are saved immediately so free Google Colab T4 sessions can resume after disconnects. Local verification checks notebook JSON and required cell content without running full transformer training.

**Tech Stack:** Jupyter Notebook, Python, PyTorch, HuggingFace Transformers Trainer, scikit-learn, pandas, NumPy, matplotlib, seaborn, Google Colab, Streamlit-compatible HuggingFace model export.

---

## Spec Reference

Approved spec:

```text
docs/superpowers/specs/2026-04-30-research-pipeline-comparative-study-design.md
```

## File Map

| File | Action | Responsibility |
| --- | --- | --- |
| `research_pipeline.ipynb` | Modify | Main research notebook with multi-seed training, evaluation, artifacts, plots, statistics, and best model export. |
| `.gitignore` | Modify | Ignore generated `artifacts/` outputs and exported zip archives. |
| `scripts/validate_research_notebook.py` | Create | Lightweight local validator that checks notebook JSON and required research pipeline sections without executing training. |

Generated but git-ignored during notebook execution:

```text
artifacts/
best_model/
training_output/
fake_job_postings.csv
*.zip
```

## Implementation Notes

- Keep `src/models/preprocessor.py` unchanged. The notebook's `clean_text()` must match it.
- Keep `best_model/model_meta.json` compatible with `src/models/classifier.py` and `src/views/main_view.py`.
- Do not add new Python package dependencies unless implementation proves one is unavoidable.
- Development verification is structural. Full training verification is expected in Google Colab with GPU T4.

---

### Task 1: Ignore Generated Experiment Artifacts

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add generated artifact ignore rules**

Append these lines to `.gitignore`:

```gitignore
artifacts/
*.zip
```

- [ ] **Step 2: Verify ignore rules are present**

Run:

```powershell
rg -n "artifacts/|\*\.zip" .gitignore
```

Expected output includes:

```text
artifacts/
*.zip
```

- [ ] **Step 3: Commit**

Run:

```powershell
git add .gitignore
git commit -m "chore: ignore research experiment artifacts" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

---

### Task 2: Add Notebook Structure Validator

**Files:**
- Create: `scripts/validate_research_notebook.py`

- [ ] **Step 1: Create `scripts/validate_research_notebook.py`**

Create the file with this content:

```python
"""Validate required sections and code markers in research_pipeline.ipynb.

This script intentionally does not execute notebook cells. It checks that the
paper-oriented comparative study workflow is present after refactoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


NOTEBOOK_PATH = Path("research_pipeline.ipynb")

REQUIRED_MARKERS = [
    "EXPERIMENT_SEEDS",
    "MODEL_REGISTRY",
    "RUN_MODE",
    "FORCE_RETRAIN",
    "EVALUATE_TRAIN_EACH_EPOCH",
    "get_environment_info",
    "write_json",
    "load_emscad_dataset",
    "prepare_text_columns",
    "clean_text",
    "create_stratified_split",
    "JobPostingDataset",
    "compute_binary_metrics",
    "find_best_threshold",
    "WeightedTrainer",
    "train_and_evaluate",
    "save_run_artifacts",
    "aggregate_completed_runs",
    "paired_bootstrap_test",
    "plot_roc_pr_curves",
    "plot_learning_curves",
    "build_error_analysis",
    "build_subgroup_metrics",
    "export_best_model",
]

REQUIRED_HEADINGS = [
    "## 1. Setup and Configuration",
    "## 2. Environment and Reproducibility Log",
    "## 3. Data Loading and Provenance",
    "## 4. EDA and Data Profile Export",
    "## 5. Text Preprocessing",
    "## 6. Repeated Stratified Split Generation",
    "## 7. Dataset Class",
    "## 8. Metrics Helpers",
    "## 9. Weighted Trainer and Training Runner",
    "## 10. Execute Resumable Experiments",
    "## 11. Aggregate Evaluation",
    "## 12. Statistical Testing",
    "## 13. Learning Curve Visualization",
    "## 14. ROC and PR Curve Visualization",
    "## 15. Error Analysis and Subgroup Analysis",
    "## 16. Paper Tables and Figures",
    "## 17. Export Best Model",
    "## 18. Colab Drive Backup",
]


def load_notebook(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    notebook = load_notebook(NOTEBOOK_PATH)
    cells = notebook.get("cells", [])
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)

    missing_markers = [marker for marker in REQUIRED_MARKERS if marker not in source]
    missing_headings = [heading for heading in REQUIRED_HEADINGS if heading not in source]

    if missing_markers or missing_headings:
        if missing_markers:
            print("Missing required code markers:")
            for marker in missing_markers:
                print(f"- {marker}")
        if missing_headings:
            print("Missing required headings:")
            for heading in missing_headings:
                print(f"- {heading}")
        return 1

    print(f"Notebook validation passed: {NOTEBOOK_PATH} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run validator before notebook refactor to confirm it fails**

Run:

```powershell
python scripts\validate_research_notebook.py
```

Expected: FAIL with missing markers and headings.

- [ ] **Step 3: Commit**

Run:

```powershell
git add scripts\validate_research_notebook.py
git commit -m "test: add research notebook structure validator" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

---

### Task 3: Rebuild Notebook Skeleton and Global Configuration

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Replace the opening notebook sections**

Replace the current title/setup cells with markdown and code cells containing this structure.

Markdown cell:

```markdown
# Job Scam Detection - Comparative Research Pipeline

This notebook compares BERT, ALBERT, and RoBERTa on the EMSCAD fake job posting dataset using repeated stratified holdout, multi-metric evaluation, statistical testing, learning curves, runtime logging, and paper-ready artifacts.
```

Markdown cell:

```markdown
## 1. Setup and Configuration
```

Code cell:

```python
import html
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    import transformers
except ImportError as exc:
    raise ImportError("Install transformers before running this notebook.") from exc

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 80)
```

Code cell:

```python
PROJECT_ROOT = Path(".").resolve()
DATASET_PATH = PROJECT_ROOT / "fake_job_postings.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACT_DIR / "runs"
SUMMARY_DIR = ARTIFACT_DIR / "summary"
FIGURES_DIR = ARTIFACT_DIR / "figures"
BEST_MODEL_DIR = PROJECT_ROOT / "best_model"

MODEL_REGISTRY = {
    "BERT": "bert-base-uncased",
    "ALBERT": "albert-base-v2",
    "RoBERTa": "roberta-base",
}

EXPERIMENT_SEEDS = [42, 123, 2024]
RUN_MODE = "single_seed"  # options: "single_seed", "full_multi_seed"
SINGLE_SEED = 42
MODELS_TO_RUN = ["BERT", "ALBERT", "RoBERTa"]
FORCE_RETRAIN = False
EVALUATE_TRAIN_EACH_EPOCH = True

CONFIG = {
    "max_len": 256,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "test_size": 0.15,
    "val_size": 0.15,
    "threshold_grid": np.round(np.arange(0.10, 0.91, 0.01), 2).tolist(),
    "bootstrap_iterations": 1000,
    "bootstrap_seed": 777,
}

for directory in [ARTIFACT_DIR, RUNS_DIR, SUMMARY_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Code cell:

```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_active_seeds() -> list[int]:
    if RUN_MODE == "single_seed":
        return [SINGLE_SEED]
    if RUN_MODE == "full_multi_seed":
        return EXPERIMENT_SEEDS
    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


def model_slug(model_label: str) -> str:
    return model_label.lower().replace(" ", "_")


def run_dir_for(seed: int, model_label: str) -> Path:
    return RUNS_DIR / f"seed_{seed}" / model_slug(model_label)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
```

- [ ] **Step 2: Validate notebook JSON**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
```

Expected: command exits with code `0`.

- [ ] **Step 3: Commit**

Run:

```powershell
git add research_pipeline.ipynb
git commit -m "refactor: add comparative study notebook configuration" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

---

### Task 4: Add Environment Logging and Data Profile Sections

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add environment logging cells**

Add markdown cell:

```markdown
## 2. Environment and Reproducibility Log
```

Add code cell:

```python
def get_environment_info() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "created_at": datetime.now().isoformat(),
        "config": CONFIG,
        "experiment_seeds": EXPERIMENT_SEEDS,
        "models": MODEL_REGISTRY,
        "run_mode": RUN_MODE,
        "evaluate_train_each_epoch": EVALUATE_TRAIN_EACH_EPOCH,
    }


environment_info = get_environment_info()
write_json(ARTIFACT_DIR / "environment.json", environment_info)
environment_info
```

- [ ] **Step 2: Add data loading and provenance cells**

Add markdown cell:

```markdown
## 3. Data Loading and Provenance
```

Add code cell:

```python
def load_emscad_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Upload fake_job_postings.csv or run the Colab download cell."
        )
    df = pd.read_csv(path)
    expected_columns = {"job_id", "fraudulent", "title", "company_profile", "description", "requirements", "benefits"}
    missing_columns = sorted(expected_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return df


df_raw = load_emscad_dataset(DATASET_PATH)
print(f"Dataset shape: {df_raw.shape}")
print(f"Fraud rate: {df_raw['fraudulent'].mean() * 100:.2f}%")
df_raw.head()
```

Add optional Colab download code cell before `load_emscad_dataset()` if the dataset is not already uploaded:

```python
if "google.colab" in sys.modules and not DATASET_PATH.exists():
    !gdown "https://drive.google.com/uc?id=1-Bn_Ey676EijYC3zdzqMIrYgA2FAHaOh" -O fake_job_postings.csv
```

- [ ] **Step 3: Add EDA and data profile export cells**

Add markdown cell:

```markdown
## 4. EDA and Data Profile Export
```

Add code cell:

```python
TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
SUBGROUP_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "industry",
    "function",
]


def build_data_profile(df: pd.DataFrame) -> dict:
    label_counts = df["fraudulent"].value_counts().sort_index()
    missing_values = df[TEXT_COLUMNS + SUBGROUP_COLUMNS].isna().sum().to_dict()
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "fraud_rate": float(df["fraudulent"].mean()),
        "missing_values": {key: int(value) for key, value in missing_values.items()},
    }


data_profile = build_data_profile(df_raw)
write_json(ARTIFACT_DIR / "data_profile.json", data_profile)
data_profile
```

Add code cell:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

label_counts = df_raw["fraudulent"].value_counts().sort_index()
axes[0].bar(["Legitimate", "Fraudulent"], label_counts.values, color=["#4c78a8", "#e45756"])
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")

missing_text = df_raw[TEXT_COLUMNS].isna().sum()
axes[1].barh(missing_text.index, missing_text.values, color="#72b7b2")
axes[1].set_title("Missing Text Fields")
axes[1].set_xlabel("Missing Count")

plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Validate JSON and commit**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
git add research_pipeline.ipynb
git commit -m "feat: add environment and data provenance logging" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

Expected: JSON validation passes and commit succeeds.

---

### Task 5: Add Preprocessing, Split Generation, and Dataset Class

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add preprocessing cells**

Add markdown cell:

```markdown
## 5. Text Preprocessing
```

Add code cell:

```python
def clean_text(text: str) -> str:
    """Clean and normalize text. This must match src/models/preprocessor.py."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in TEXT_COLUMNS:
        prepared[column] = prepared[column].fillna("")
    prepared["text_raw"] = prepared[TEXT_COLUMNS].agg(" ".join, axis=1)
    prepared["text"] = prepared["text_raw"].apply(clean_text)
    prepared["text_length"] = prepared["text"].str.len()
    return prepared


df = prepare_text_columns(df_raw)
print(df[["job_id", "fraudulent", "text_length", "text"]].head())
```

- [ ] **Step 2: Add split generation cells**

Add markdown cell:

```markdown
## 6. Repeated Stratified Split Generation
```

Add code cell:

```python
def create_stratified_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        df,
        test_size=CONFIG["test_size"],
        stratify=df["fraudulent"],
        random_state=seed,
    )
    val_fraction = CONFIG["val_size"] / (1 - CONFIG["test_size"])
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        stratify=train_val_df["fraudulent"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def summarize_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        rows.append(
            {
                "split": split_name,
                "rows": len(split_df),
                "fraud_count": int(split_df["fraudulent"].sum()),
                "fraud_rate": float(split_df["fraudulent"].mean()),
            }
        )
    return pd.DataFrame(rows)


preview_train_df, preview_val_df, preview_test_df = create_stratified_split(df, SINGLE_SEED)
summarize_split(preview_train_df, preview_val_df, preview_test_df)
```

- [ ] **Step 3: Add dataset class cells**

Add markdown cell:

```markdown
## 7. Dataset Class
```

Add code cell:

```python
class JobPostingDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer, max_len: int):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoding = self.tokenizer(
            str(self.texts[index]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }
```

- [ ] **Step 4: Validate JSON and commit**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
git add research_pipeline.ipynb
git commit -m "feat: add reusable preprocessing and split helpers" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

Expected: JSON validation passes and commit succeeds.

---

### Task 6: Add Metrics, Threshold Tuning, and Bootstrap Helpers

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add metrics helper cells**

Add markdown cell:

```markdown
## 8. Metrics Helpers
```

Add code cell:

```python
def safe_auc(metric_func, labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(metric_func(labels, scores))
    except ValueError:
        return float("nan")


def compute_binary_metrics(labels: np.ndarray, fraud_probs: np.ndarray, threshold: float = 0.50) -> dict:
    labels = np.asarray(labels).astype(int)
    fraud_probs = np.asarray(fraud_probs).astype(float)
    predictions = (fraud_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "fraud_precision": float(precision_score(labels, predictions, zero_division=0)),
        "fraud_recall": float(recall_score(labels, predictions, zero_division=0)),
        "fraud_f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "specificity": float(specificity),
        "roc_auc": safe_auc(roc_auc_score, labels, fraud_probs),
        "pr_auc": safe_auc(average_precision_score, labels, fraud_probs),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(labels: np.ndarray, fraud_probs: np.ndarray) -> tuple[float, dict]:
    best_threshold = 0.50
    best_metrics = compute_binary_metrics(labels, fraud_probs, best_threshold)
    for threshold in CONFIG["threshold_grid"]:
        metrics = compute_binary_metrics(labels, fraud_probs, threshold)
        if metrics["fraud_f1"] > best_metrics["fraud_f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics
```

Add code cell:

```python
def compute_trainer_metrics(eval_prediction) -> dict:
    logits, labels = eval_prediction
    fraud_probs = F.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    metrics = compute_binary_metrics(labels, fraud_probs, threshold=0.50)
    return {
        "accuracy": metrics["accuracy"],
        "fraud_precision": metrics["fraud_precision"],
        "fraud_recall": metrics["fraud_recall"],
        "fraud_f1": metrics["fraud_f1"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }
```

- [ ] **Step 2: Add bootstrap helper cell**

Add markdown cell:

```markdown
## 12. Statistical Testing
```

Add code cell:

```python
def paired_bootstrap_test(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    metric_name: str,
    iterations: int = 1000,
    seed: int = 777,
) -> dict:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(int)
    probs_a = np.asarray(probs_a).astype(float)
    probs_b = np.asarray(probs_b).astype(float)
    n = len(labels)
    differences = []

    def metric_value(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> float:
        if metric_name == "fraud_f1":
            preds = (probs >= threshold).astype(int)
            return float(f1_score(y_true, preds, zero_division=0))
        if metric_name == "pr_auc":
            return safe_auc(average_precision_score, y_true, probs)
        raise ValueError(f"Unsupported bootstrap metric: {metric_name}")

    observed = metric_value(labels, probs_a, threshold_a) - metric_value(labels, probs_b, threshold_b)
    for _ in range(iterations):
        sample_idx = rng.integers(0, n, size=n)
        diff = metric_value(labels[sample_idx], probs_a[sample_idx], threshold_a) - metric_value(
            labels[sample_idx], probs_b[sample_idx], threshold_b
        )
        differences.append(diff)

    differences = np.asarray(differences)
    p_value = float(2 * min(np.mean(differences <= 0), np.mean(differences >= 0)))
    return {
        "metric": metric_name,
        "observed_difference": float(observed),
        "ci_lower": float(np.percentile(differences, 2.5)),
        "ci_upper": float(np.percentile(differences, 97.5)),
        "p_value_approx": min(p_value, 1.0),
    }
```

- [ ] **Step 3: Validate JSON and commit**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
git add research_pipeline.ipynb
git commit -m "feat: add comparative metrics and bootstrap helpers" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

Expected: JSON validation passes and commit succeeds.

---

### Task 7: Add Weighted Trainer and Resumable Training Runner

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add weighted trainer cells**

Add markdown cell:

```markdown
## 9. Weighted Trainer and Training Runner
```

Add code cell:

```python
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    class_counts = train_df["fraudulent"].value_counts().sort_index()
    total = len(train_df)
    weights = torch.tensor(
        [total / (2 * class_counts[0]), total / (2 * class_counts[1])],
        dtype=torch.float32,
    )
    return weights
```

- [ ] **Step 2: Add artifact writer helper cells**

Add code cell:

```python
def save_run_artifacts(
    run_dir: Path,
    config_payload: dict,
    runtime_payload: dict,
    default_metrics: dict,
    tuned_metrics: dict,
    predictions_df: pd.DataFrame,
    training_history_df: pd.DataFrame,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config_payload)
    write_json(run_dir / "runtime.json", runtime_payload)
    write_json(run_dir / "metrics_default_threshold.json", default_metrics)
    write_json(run_dir / "metrics_tuned_threshold.json", tuned_metrics)
    confusion_payload = {
        "tn": default_metrics["tn"],
        "fp": default_metrics["fp"],
        "fn": default_metrics["fn"],
        "tp": default_metrics["tp"],
    }
    write_json(run_dir / "confusion_matrix.json", confusion_payload)
    predictions_df.to_csv(run_dir / "predictions.csv", index=False)
    training_history_df.to_csv(run_dir / "training_history.csv", index=False)


def trainer_history_to_frame(log_history: list[dict]) -> pd.DataFrame:
    rows = []
    for row in log_history:
        if "epoch" not in row:
            continue
        cleaned = {"epoch": row.get("epoch")}
        for key, value in row.items():
            if isinstance(value, (int, float)) and key != "epoch":
                cleaned[key] = value
        rows.append(cleaned)
    return pd.DataFrame(rows)
```

- [ ] **Step 3: Add prediction frame helper cell**

Add code cell:

```python
def build_predictions_frame(test_df: pd.DataFrame, fraud_probs: np.ndarray, threshold: float) -> pd.DataFrame:
    predictions = (fraud_probs >= threshold).astype(int)
    output = pd.DataFrame(
        {
            "job_id": test_df["job_id"].values,
            "true_label": test_df["fraudulent"].astype(int).values,
            "predicted_label": predictions,
            "fraud_probability": fraud_probs,
            "text_length": test_df["text_length"].values,
            "title": test_df["title"].fillna("").values,
            "location": test_df["location"].fillna("").values if "location" in test_df.columns else "",
            "cleaned_text_excerpt": test_df["text"].str.slice(0, 500).values,
        }
    )
    output["error_type"] = np.select(
        [
            (output["true_label"] == 1) & (output["predicted_label"] == 1),
            (output["true_label"] == 0) & (output["predicted_label"] == 0),
            (output["true_label"] == 0) & (output["predicted_label"] == 1),
            (output["true_label"] == 1) & (output["predicted_label"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )
    return output
```

- [ ] **Step 4: Add `train_and_evaluate` runner cell**

Add code cell:

```python
def train_and_evaluate(model_label: str, seed: int, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    set_seed(seed)
    hf_model_id = MODEL_REGISTRY[model_label]
    current_run_dir = run_dir_for(seed, model_label)
    completed_marker = current_run_dir / "metrics_tuned_threshold.json"

    if completed_marker.exists() and not FORCE_RETRAIN:
        print(f"Skipping completed run: seed={seed}, model={model_label}")
        return {
            "seed": seed,
            "model_label": model_label,
            "run_dir": str(current_run_dir),
            "status": "skipped",
        }

    start_time = time.time()
    started_at = datetime.now().isoformat()
    current_run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_id, num_labels=2)

    train_dataset = JobPostingDataset(train_df["text"], train_df["fraudulent"], tokenizer, CONFIG["max_len"])
    val_dataset = JobPostingDataset(val_df["text"], val_df["fraudulent"], tokenizer, CONFIG["max_len"])
    test_dataset = JobPostingDataset(test_df["text"], test_df["fraudulent"], tokenizer, CONFIG["max_len"])

    class_weights = compute_class_weights(train_df)
    output_dir = current_run_dir / "trainer_output"
    model_dir = current_run_dir / "model"

    eval_dataset = {"validation": val_dataset}
    metric_for_best_model = "eval_validation_fraud_f1"
    if EVALUATE_TRAIN_EACH_EPOCH:
        eval_dataset["train"] = train_dataset

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_ratio=CONFIG["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        logging_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_trainer_metrics,
    )

    trainer.train()

    val_output = trainer.predict(val_dataset)
    test_output = trainer.predict(test_dataset)
    val_probs = F.softmax(torch.tensor(val_output.predictions), dim=-1).numpy()[:, 1]
    test_probs = F.softmax(torch.tensor(test_output.predictions), dim=-1).numpy()[:, 1]
    val_labels = val_df["fraudulent"].astype(int).values
    test_labels = test_df["fraudulent"].astype(int).values

    tuned_threshold, validation_tuned_metrics = find_best_threshold(val_labels, val_probs)
    default_metrics = compute_binary_metrics(test_labels, test_probs, threshold=0.50)
    tuned_metrics = compute_binary_metrics(test_labels, test_probs, threshold=tuned_threshold)
    tuned_metrics["validation_selected_threshold"] = float(tuned_threshold)
    tuned_metrics["validation_fraud_f1_at_threshold"] = float(validation_tuned_metrics["fraud_f1"])

    predictions_df = build_predictions_frame(test_df, test_probs, tuned_threshold)
    training_history_df = trainer_history_to_frame(trainer.state.log_history)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    ended_at = datetime.now().isoformat()
    total_seconds = time.time() - start_time
    effective_batch_size = CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"]
    runtime_payload = {
        "seed": seed,
        "model_label": model_label,
        "hf_model_id": hf_model_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "total_seconds": float(total_seconds),
        "total_minutes": float(total_seconds / 60),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "batch_size": CONFIG["batch_size"],
        "gradient_accumulation_steps": CONFIG["gradient_accumulation_steps"],
        "effective_batch_size": effective_batch_size,
        "fp16": torch.cuda.is_available(),
        "epochs": CONFIG["epochs"],
        "max_len": CONFIG["max_len"],
    }
    config_payload = {
        "seed": seed,
        "model_label": model_label,
        "hf_model_id": hf_model_id,
        "config": CONFIG,
        "evaluate_train_each_epoch": EVALUATE_TRAIN_EACH_EPOCH,
    }

    save_run_artifacts(
        current_run_dir,
        config_payload,
        runtime_payload,
        default_metrics,
        tuned_metrics,
        predictions_df,
        training_history_df,
    )

    return {
        "seed": seed,
        "model_label": model_label,
        "run_dir": str(current_run_dir),
        "status": "completed",
        "fraud_f1": tuned_metrics["fraud_f1"],
        "pr_auc": tuned_metrics["pr_auc"],
        "runtime_minutes": runtime_payload["total_minutes"],
    }
```

- [ ] **Step 5: Validate JSON and commit**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
git add research_pipeline.ipynb
git commit -m "feat: add resumable transformer training runner" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

Expected: JSON validation passes and commit succeeds.

---

### Task 8: Add Resumable Experiment Execution and Aggregation

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add experiment execution cells**

Add markdown cell:

```markdown
## 10. Execute Resumable Experiments
```

Add code cell:

```python
experiment_results = []

for seed in get_active_seeds():
    train_df, val_df, test_df = create_stratified_split(df, seed)
    split_summary = summarize_split(train_df, val_df, test_df)
    split_summary.to_csv(RUNS_DIR / f"seed_{seed}" / "split_summary.csv", index=False)
    print(f"\nSeed {seed} split summary")
    display(split_summary)

    for model_label in MODELS_TO_RUN:
        print(f"\nRunning seed={seed}, model={model_label}")
        result = train_and_evaluate(model_label, seed, train_df, val_df, test_df)
        experiment_results.append(result)

pd.DataFrame(experiment_results)
```

- [ ] **Step 2: Add aggregation cells**

Add markdown cell:

```markdown
## 11. Aggregate Evaluation
```

Add code cell:

```python
def load_completed_run(seed: int, model_label: str) -> dict | None:
    current_run_dir = run_dir_for(seed, model_label)
    default_path = current_run_dir / "metrics_default_threshold.json"
    tuned_path = current_run_dir / "metrics_tuned_threshold.json"
    runtime_path = current_run_dir / "runtime.json"
    if not (default_path.exists() and tuned_path.exists() and runtime_path.exists()):
        return None
    default_metrics = read_json(default_path)
    tuned_metrics = read_json(tuned_path)
    runtime_metrics = read_json(runtime_path)
    row = {
        "seed": seed,
        "model_label": model_label,
        "run_dir": str(current_run_dir),
        "default_threshold": default_metrics["threshold"],
        "tuned_threshold": tuned_metrics["threshold"],
        "runtime_minutes": runtime_metrics["total_minutes"],
    }
    for key, value in default_metrics.items():
        row[f"default_{key}"] = value
    for key, value in tuned_metrics.items():
        row[f"tuned_{key}"] = value
    return row


def aggregate_completed_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in EXPERIMENT_SEEDS:
        for model_label in MODEL_REGISTRY:
            loaded = load_completed_run(seed, model_label)
            if loaded is not None:
                rows.append(loaded)
    all_runs_df = pd.DataFrame(rows)
    all_runs_df.to_csv(SUMMARY_DIR / "all_runs.csv", index=False)

    metric_columns = [
        "tuned_accuracy",
        "tuned_fraud_precision",
        "tuned_fraud_recall",
        "tuned_fraud_f1",
        "tuned_macro_f1",
        "tuned_balanced_accuracy",
        "tuned_specificity",
        "tuned_roc_auc",
        "tuned_pr_auc",
        "runtime_minutes",
    ]
    summary_df = (
        all_runs_df.groupby("model_label")[metric_columns]
        .agg(["mean", "std"])
        .sort_values(("tuned_fraud_f1", "mean"), ascending=False)
    )
    summary_df.to_csv(SUMMARY_DIR / "mean_std_by_model.csv")
    all_runs_df[["seed", "model_label", "runtime_minutes"]].to_csv(SUMMARY_DIR / "runtime_by_model.csv", index=False)
    return all_runs_df, summary_df


all_runs_df, mean_std_by_model = aggregate_completed_runs()
display(all_runs_df)
display(mean_std_by_model)
```

- [ ] **Step 3: Validate JSON and commit**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
git add research_pipeline.ipynb
git commit -m "feat: add experiment aggregation workflow" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

Expected: JSON validation passes and commit succeeds.

---

### Task 9: Add Statistical Comparison Workflow

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add statistical comparison execution cell**

Add this code cell under the `## 12. Statistical Testing` section after `paired_bootstrap_test()`:

```python
def run_statistical_tests() -> pd.DataFrame:
    comparisons = [("ALBERT", "BERT"), ("ALBERT", "RoBERTa"), ("BERT", "RoBERTa")]
    rows = []

    for seed in EXPERIMENT_SEEDS:
        prediction_frames = {}
        thresholds = {}
        for model_label in MODEL_REGISTRY:
            current_run_dir = run_dir_for(seed, model_label)
            prediction_path = current_run_dir / "predictions.csv"
            metrics_path = current_run_dir / "metrics_tuned_threshold.json"
            if prediction_path.exists() and metrics_path.exists():
                prediction_frames[model_label] = pd.read_csv(prediction_path).sort_values("job_id").reset_index(drop=True)
                thresholds[model_label] = read_json(metrics_path)["threshold"]

        for model_a, model_b in comparisons:
            if model_a not in prediction_frames or model_b not in prediction_frames:
                continue
            frame_a = prediction_frames[model_a]
            frame_b = prediction_frames[model_b]
            labels = frame_a["true_label"].values
            for metric_name in ["fraud_f1", "pr_auc"]:
                result = paired_bootstrap_test(
                    labels=labels,
                    probs_a=frame_a["fraud_probability"].values,
                    probs_b=frame_b["fraud_probability"].values,
                    threshold_a=thresholds[model_a],
                    threshold_b=thresholds[model_b],
                    metric_name=metric_name,
                    iterations=CONFIG["bootstrap_iterations"],
                    seed=CONFIG["bootstrap_seed"] + seed,
                )
                result.update({"seed": seed, "model_a": model_a, "model_b": model_b})
                rows.append(result)

    statistical_tests_df = pd.DataFrame(rows)
    statistical_tests_df.to_csv(SUMMARY_DIR / "significance_tests.csv", index=False)
    return statistical_tests_df


significance_tests_df = run_statistical_tests()
display(significance_tests_df)
```

- [ ] **Step 2: Validate JSON and commit**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
git add research_pipeline.ipynb
git commit -m "feat: add paired bootstrap statistical tests" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

Expected: JSON validation passes and commit succeeds.

---

### Task 10: Add Learning Curve, ROC, PR, and Comparison Figures

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add learning curve plotting cells**

Add markdown cell:

```markdown
## 13. Learning Curve Visualization
```

Add code cell:

```python
def normalize_history_frame(history_df: pd.DataFrame, model_label: str, seed: int) -> pd.DataFrame:
    normalized = history_df.copy()
    normalized["model_label"] = model_label
    normalized["seed"] = seed
    return normalized


def load_training_histories() -> pd.DataFrame:
    frames = []
    for seed in EXPERIMENT_SEEDS:
        for model_label in MODEL_REGISTRY:
            history_path = run_dir_for(seed, model_label) / "training_history.csv"
            if history_path.exists():
                frames.append(normalize_history_frame(pd.read_csv(history_path), model_label, seed))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_learning_curves() -> None:
    history_df = load_training_histories()
    if history_df.empty:
        print("No training history files found.")
        return

    curve_specs = [
        ("loss", ["loss", "eval_validation_loss", "eval_train_loss"], "learning_curves_loss_mean_std.png"),
        ("accuracy", ["eval_validation_accuracy", "eval_train_accuracy"], "learning_curves_accuracy_mean_std.png"),
        ("f1", ["eval_validation_fraud_f1", "eval_train_fraud_f1"], "learning_curves_f1_mean_std.png"),
    ]

    for title, candidate_columns, filename in curve_specs:
        available_columns = [column for column in candidate_columns if column in history_df.columns]
        if not available_columns:
            print(f"No columns found for {title}: {candidate_columns}")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for model_label in MODEL_REGISTRY:
            model_history = history_df[history_df["model_label"] == model_label]
            for column in available_columns:
                curve = model_history.groupby("epoch")[column].agg(["mean", "std"]).dropna()
                if curve.empty:
                    continue
                label = f"{model_label} {column}"
                ax.plot(curve.index, curve["mean"], marker="o", label=label)
                ax.fill_between(
                    curve.index,
                    curve["mean"] - curve["std"].fillna(0),
                    curve["mean"] + curve["std"].fillna(0),
                    alpha=0.12,
                )
        ax.set_title(f"Learning Curves - {title.title()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title.title())
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=200)
        plt.show()


plot_learning_curves()
```

- [ ] **Step 2: Add ROC and PR plotting cells**

Add markdown cell:

```markdown
## 14. ROC and PR Curve Visualization
```

Add code cell:

```python
def load_prediction_frames() -> pd.DataFrame:
    frames = []
    for seed in EXPERIMENT_SEEDS:
        for model_label in MODEL_REGISTRY:
            prediction_path = run_dir_for(seed, model_label) / "predictions.csv"
            if prediction_path.exists():
                frame = pd.read_csv(prediction_path)
                frame["seed"] = seed
                frame["model_label"] = model_label
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_roc_pr_curves() -> None:
    predictions_df = load_prediction_frames()
    if predictions_df.empty:
        print("No prediction files found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_label in MODEL_REGISTRY:
        model_predictions = predictions_df[predictions_df["model_label"] == model_label]
        if model_predictions.empty:
            continue
        labels = model_predictions["true_label"].values
        probs = model_predictions["fraud_probability"].values

        fpr, tpr, _ = roc_curve(labels, probs)
        precision, recall, _ = precision_recall_curve(labels, probs)
        roc_score = roc_auc_score(labels, probs)
        pr_score = average_precision_score(labels, probs)

        axes[0].plot(fpr, tpr, label=f"{model_label} AUC={roc_score:.3f}")
        axes[1].plot(recall, precision, label=f"{model_label} AP={pr_score:.3f}")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=200)
    fig.savefig(FIGURES_DIR / "pr_curves.png", dpi=200)
    plt.show()


plot_roc_pr_curves()
```

- [ ] **Step 3: Add model comparison figure cell**

Add code cell:

```python
def plot_model_comparison(mean_std_df: pd.DataFrame) -> None:
    if mean_std_df.empty:
        print("No aggregate metrics found.")
        return
    metrics = ["tuned_fraud_precision", "tuned_fraud_recall", "tuned_fraud_f1", "tuned_pr_auc"]
    means = mean_std_df.loc[:, [(metric, "mean") for metric in metrics]]
    stds = mean_std_df.loc[:, [(metric, "std") for metric in metrics]]
    means.columns = metrics
    stds.columns = metrics

    ax = means.plot(kind="bar", yerr=stds, figsize=(10, 6), capsize=4)
    ax.set_title("Model Comparison - Mean +/- Std Across Seeds")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_mean_std.png", dpi=200)
    plt.show()


plot_model_comparison(mean_std_by_model)
```

- [ ] **Step 4: Validate JSON and commit**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
git add research_pipeline.ipynb
git commit -m "feat: add paper-ready research figures" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

Expected: JSON validation passes and commit succeeds.

---

### Task 11: Add Error Analysis and Subgroup Metrics

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add error analysis cells**

Add markdown cell:

```markdown
## 15. Error Analysis and Subgroup Analysis
```

Add code cell:

```python
def build_error_analysis() -> pd.DataFrame:
    predictions_df = load_prediction_frames()
    if predictions_df.empty:
        return pd.DataFrame()

    rows = []
    for model_label in MODEL_REGISTRY:
        model_predictions = predictions_df[predictions_df["model_label"] == model_label]
        high_conf_fp = model_predictions[model_predictions["error_type"] == "FP"].nlargest(10, "fraud_probability")
        high_conf_fn = model_predictions[model_predictions["error_type"] == "FN"].nsmallest(10, "fraud_probability")
        for case_type, frame in [("high_confidence_fp", high_conf_fp), ("high_confidence_fn", high_conf_fn)]:
            case_frame = frame.copy()
            case_frame["case_type"] = case_type
            rows.append(case_frame)

    error_cases_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    error_cases_df.to_csv(SUMMARY_DIR / "error_cases.csv", index=False)
    return error_cases_df


error_cases_df = build_error_analysis()
display(error_cases_df.head(20))
```

- [ ] **Step 2: Add subgroup metrics cells**

Add code cell:

```python
def attach_subgroup_columns(predictions_df: pd.DataFrame) -> pd.DataFrame:
    subgroup_source = df[["job_id"] + SUBGROUP_COLUMNS].copy()
    return predictions_df.merge(subgroup_source, on="job_id", how="left")


def build_subgroup_metrics() -> pd.DataFrame:
    predictions_df = load_prediction_frames()
    if predictions_df.empty:
        return pd.DataFrame()
    predictions_df = attach_subgroup_columns(predictions_df)
    rows = []

    for model_label in MODEL_REGISTRY:
        model_predictions = predictions_df[predictions_df["model_label"] == model_label]
        for subgroup_column in SUBGROUP_COLUMNS:
            for subgroup_value, subgroup_df in model_predictions.groupby(subgroup_column, dropna=False):
                if len(subgroup_df) == 0:
                    continue
                labels = subgroup_df["true_label"].astype(int).values
                preds = subgroup_df["predicted_label"].astype(int).values
                rows.append(
                    {
                        "model_label": model_label,
                        "subgroup_column": subgroup_column,
                        "subgroup_value": str(subgroup_value),
                        "sample_count": int(len(subgroup_df)),
                        "fraud_count": int(labels.sum()),
                        "fraud_precision": float(precision_score(labels, preds, zero_division=0)),
                        "fraud_recall": float(recall_score(labels, preds, zero_division=0)),
                        "fraud_f1": float(f1_score(labels, preds, zero_division=0)),
                    }
                )

    subgroup_metrics_df = pd.DataFrame(rows)
    subgroup_metrics_df.to_csv(SUMMARY_DIR / "subgroup_metrics.csv", index=False)
    return subgroup_metrics_df


subgroup_metrics_df = build_subgroup_metrics()
display(subgroup_metrics_df.head(20))
```

- [ ] **Step 3: Validate JSON and commit**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
git add research_pipeline.ipynb
git commit -m "feat: add error and subgroup analysis outputs" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

Expected: JSON validation passes and commit succeeds.

---

### Task 12: Add Paper Tables, Best Model Export, and Colab Backup

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add paper output cells**

Add markdown cell:

```markdown
## 16. Paper Tables and Figures
```

Add code cell:

```python
table_1_dataset_statistics = pd.DataFrame(
    [
        {
            "total_samples": data_profile["rows"],
            "legitimate_samples": data_profile["label_counts"].get("0", 0),
            "fraudulent_samples": data_profile["label_counts"].get("1", 0),
            "fraud_rate": data_profile["fraud_rate"],
        }
    ]
)
table_1_dataset_statistics.to_csv(SUMMARY_DIR / "table_1_dataset_statistics.csv", index=False)

table_2_model_performance = mean_std_by_model.copy()
table_2_model_performance.to_csv(SUMMARY_DIR / "table_2_model_performance.csv")

table_3_statistical_comparison = significance_tests_df.copy() if "significance_tests_df" in globals() else pd.DataFrame()
table_3_statistical_comparison.to_csv(SUMMARY_DIR / "table_3_statistical_comparison.csv", index=False)

table_4_runtime = all_runs_df[["seed", "model_label", "runtime_minutes"]].copy() if "all_runs_df" in globals() else pd.DataFrame()
table_4_runtime.to_csv(SUMMARY_DIR / "table_4_runtime.csv", index=False)

display(table_1_dataset_statistics)
display(table_2_model_performance)
display(table_3_statistical_comparison.head())
display(table_4_runtime.head())
```

- [ ] **Step 2: Add best model export cells**

Add markdown cell:

```markdown
## 17. Export Best Model
```

Add code cell:

```python
def select_best_completed_run(all_runs_df: pd.DataFrame) -> pd.Series:
    if all_runs_df.empty:
        raise ValueError("No completed runs available for best model export.")
    sorted_runs = all_runs_df.sort_values(
        ["tuned_fraud_f1", "tuned_pr_auc", "tuned_fraud_recall"],
        ascending=False,
    )
    return sorted_runs.iloc[0]


def export_best_model() -> dict:
    best_run = select_best_completed_run(all_runs_df)
    source_model_dir = Path(best_run["run_dir"]) / "model"
    if not source_model_dir.exists():
        raise FileNotFoundError(f"Best run model directory not found: {source_model_dir}")

    if BEST_MODEL_DIR.exists():
        shutil.rmtree(BEST_MODEL_DIR)
    shutil.copytree(source_model_dir, BEST_MODEL_DIR)

    model_label = best_run["model_label"]
    seed = int(best_run["seed"])
    meta = {
        "model_name": model_label,
        "hf_model_id": MODEL_REGISTRY[model_label],
        "metrics": {
            "accuracy": float(best_run["tuned_accuracy"]),
            "precision": float(best_run["tuned_fraud_precision"]),
            "recall": float(best_run["tuned_fraud_recall"]),
            "f1": float(best_run["tuned_fraud_f1"]),
            "pr_auc": float(best_run["tuned_pr_auc"]),
            "roc_auc": float(best_run["tuned_roc_auc"]),
        },
        "config": CONFIG,
        "exported_at": datetime.now().isoformat(),
        "selection_metric": "tuned_fraud_f1",
        "selected_seed": seed,
        "selected_threshold": float(best_run["tuned_threshold"]),
        "artifact_path": str(best_run["run_dir"]),
        "multi_seed_summary_path": str(SUMMARY_DIR / "mean_std_by_model.csv"),
    }
    write_json(BEST_MODEL_DIR / "model_meta.json", meta)
    return meta


best_model_meta = export_best_model()
best_model_meta
```

- [ ] **Step 3: Add Colab backup cells**

Add markdown cell:

```markdown
## 18. Colab Drive Backup
```

Add code cell:

```python
def backup_artifacts_to_drive() -> None:
    if "google.colab" not in sys.modules:
        print("Not running in Google Colab; skipping Drive backup.")
        return

    from google.colab import drive

    drive.mount("/content/drive")
    backup_root = Path("/content/drive/MyDrive/job_scam_research_artifacts")
    backup_root.mkdir(parents=True, exist_ok=True)

    artifacts_zip = shutil.make_archive("job_scam_artifacts", "zip", str(ARTIFACT_DIR))
    best_model_zip = shutil.make_archive("best_model", "zip", str(BEST_MODEL_DIR))
    shutil.copy(artifacts_zip, backup_root / "job_scam_artifacts.zip")
    shutil.copy(best_model_zip, backup_root / "best_model.zip")
    print(f"Backed up artifacts to {backup_root}")


backup_artifacts_to_drive()
```

- [ ] **Step 4: Validate JSON and commit**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
git add research_pipeline.ipynb
git commit -m "feat: add paper tables and best model export" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

Expected: JSON validation passes and commit succeeds.

---

### Task 13: Run Final Local Validation

**Files:**
- Modify: `research_pipeline.ipynb` only if validation finds missing required markers.

- [ ] **Step 1: Run notebook JSON validation**

Run:

```powershell
python -m json.tool research_pipeline.ipynb > $null
```

Expected: command exits with code `0`.

- [ ] **Step 2: Run structure validator**

Run:

```powershell
python scripts\validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb
```

The cell count may appear after the filename. That is acceptable.

- [ ] **Step 3: Run git status**

Run:

```powershell
git status --short
```

Expected: no unstaged changes after the final commit. If validation required notebook fixes, commit them with:

```powershell
git add research_pipeline.ipynb scripts\validate_research_notebook.py .gitignore
git commit -m "test: validate comparative study notebook structure" -m "Co-authored-by: z0zero <32585806+z0zero@users.noreply.github.com>"
```

---

### Task 14: Colab Verification Checklist

**Files:**
- No repository file changes required unless a Colab-only error exposes a notebook bug.

- [ ] **Step 1: Upload or download dataset**

In Colab, confirm:

```python
DATASET_PATH.exists()
```

Expected:

```text
True
```

- [ ] **Step 2: Run one smoke experiment**

Use these settings:

```python
RUN_MODE = "single_seed"
SINGLE_SEED = 42
MODELS_TO_RUN = ["ALBERT"]
FORCE_RETRAIN = False
EVALUATE_TRAIN_EACH_EPOCH = True
```

Run all notebook cells through `## 10. Execute Resumable Experiments`.

Expected artifacts:

```text
artifacts/runs/seed_42/albert/config.json
artifacts/runs/seed_42/albert/metrics_default_threshold.json
artifacts/runs/seed_42/albert/metrics_tuned_threshold.json
artifacts/runs/seed_42/albert/predictions.csv
artifacts/runs/seed_42/albert/training_history.csv
artifacts/runs/seed_42/albert/runtime.json
artifacts/runs/seed_42/albert/model/
```

- [ ] **Step 3: Run full multi-seed experiment**

Use these settings:

```python
RUN_MODE = "full_multi_seed"
MODELS_TO_RUN = ["BERT", "ALBERT", "RoBERTa"]
FORCE_RETRAIN = False
```

Run experiment cells. Completed runs should skip on repeated execution.

- [ ] **Step 4: Run aggregate, statistics, figures, and export cells**

Expected summary artifacts:

```text
artifacts/summary/all_runs.csv
artifacts/summary/mean_std_by_model.csv
artifacts/summary/significance_tests.csv
artifacts/summary/runtime_by_model.csv
artifacts/summary/subgroup_metrics.csv
artifacts/summary/error_cases.csv
artifacts/figures/model_comparison_mean_std.png
artifacts/figures/roc_curves.png
artifacts/figures/pr_curves.png
artifacts/figures/learning_curves_loss_mean_std.png
artifacts/figures/learning_curves_accuracy_mean_std.png
artifacts/figures/learning_curves_f1_mean_std.png
best_model/model_meta.json
```

- [ ] **Step 5: Confirm app-compatible metadata**

In Colab or local after downloading `best_model/`, run:

```python
import json
with open("best_model/model_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
for key in ["model_name", "hf_model_id", "metrics", "config", "exported_at"]:
    assert key in meta, key
meta
```

Expected: no assertion error.

---

## Self-Review Checklist

- Spec coverage: The plan covers multi-seed repeated holdout, Colab resumability, ROC/PR curves, train/validation loss, train/validation accuracy, train/validation F1, runtime logging, threshold tuning, paired bootstrap, error analysis, subgroup analysis, paper tables, and best model export.
- Local verification: The plan adds notebook JSON validation and a structural validator.
- Runtime risk: Full model training is deferred to Colab because local execution may lack GPU and dataset.
- Compatibility: Existing app-compatible `best_model/model_meta.json` fields are preserved.
- Scope: No new model families, XAI methods, external datasets, or new package dependencies are introduced.
