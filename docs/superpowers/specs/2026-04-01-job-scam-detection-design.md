# Job Scam Detection — Design Spec

## Overview

Research project for detecting fraudulent job postings using NLP. Two deliverables: a Jupyter Notebook comparing three transformer models (BERT, ALBERT, RoBERTa), and a Streamlit web app that serves the best model with OCR support for image-based input.

## Project Structure

```
job-scam/
├── research_pipeline.ipynb   # Full research notebook
├── app.py                    # Streamlit web app
├── requirements.txt          # All dependencies
└── best_model/               # Created by notebook, consumed by app
    ├── model files
    ├── tokenizer files
    └── model_meta.json
```

## Dataset

**Employment Scam Aegean Dataset (EMSCAD)** — `fake_job_postings.csv` from Kaggle. ~17,880 job postings, ~800 fraudulent (~5%). Binary classification: `fraudulent` column (0 = legitimate, 1 = scam).

## Phase 1: Research Notebook (`research_pipeline.ipynb`)

### Sections

1. **Setup & Config**
   - Install/import dependencies
   - Hyperparameters: epochs=3, batch_size=16, learning_rate=2e-5, max_len=256, weight_decay=0.01
   - Set random seed for reproducibility
   - Device detection (CUDA)

2. **Data Loading**
   - Load `fake_job_postings.csv`
   - Combine text columns: `title`, `company_profile`, `description`, `requirements`, `benefits` into single `text` field (space-separated, nulls replaced with empty string)
   - Target: `fraudulent` column

3. **Exploratory Data Analysis**
   - Class distribution (bar chart)
   - Text length distribution
   - Missing value summary

4. **Preprocessing**
   - Fill NaN with empty string before concatenation
   - Clean text: lowercase, strip HTML tags, strip URLs, collapse whitespace
   - Stratified train/val/test split: 70/15/15
   - Preserve class ratio across splits

5. **Custom PyTorch Dataset**
   - Wraps HuggingFace tokenizer
   - Tokenizes on-the-fly with truncation and padding to `max_len=256`
   - Returns `input_ids`, `attention_mask`, `labels`

6. **Model Training (x3)**
   - Models: `bert-base-uncased`, `albert-base-v2`, `roberta-base`
   - Load via `AutoModelForSequenceClassification` (num_labels=2)
   - Load tokenizer via `AutoTokenizer`
   - Custom `Trainer` subclass with weighted cross-entropy loss to handle class imbalance (weight derived from class frequencies)
   - `TrainingArguments`: 3 epochs, batch 16, lr 2e-5, weight_decay 0.01, warmup_ratio 0.1, evaluation each epoch, save best by F1
   - Custom `compute_metrics` function: accuracy, precision, recall, F1

7. **Evaluation & Comparison**
   - Per-model metrics on test set: accuracy, precision, recall, F1
   - Confusion matrix heatmap per model (matplotlib/seaborn)
   - Combined comparison DataFrame (tabular)
   - Grouped bar chart comparing all metrics across models

8. **Export Best Model**
   - Select best model by F1-score on test set
   - Save model + tokenizer to `./best_model/`
   - Write `model_meta.json`: model name, all metrics, timestamp

## Phase 2: Streamlit App (`app.py`)

### UI

- Header: title, description, model info badge (reads `model_meta.json`)
- Input mode selector (radio): "Paste Text" / "Upload Image"
  - Text mode: `st.text_area` for pasting job description
  - Image mode: `st.file_uploader` (png, jpg, jpeg) -> `pytesseract.image_to_string` -> editable `st.text_area` showing extracted text
- "Analyze" button
- Results section:
  - Classification: "Legitimate Job" (green) or "Potential Scam" (red)
  - Confidence score as percentage
  - Styled with `st.success` / `st.error`

### Backend

- `@st.cache_resource` to load model + tokenizer from `./best_model/` at startup
- Tokenize input text (same max_len=256, truncation, padding)
- Forward pass -> softmax -> class prediction + confidence
- Display results

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| max_len=256 | Balances text coverage vs. GPU memory; most signal is in first 256 tokens |
| Weighted loss | EMSCAD is ~95/5 split — unweighted loss would bias toward majority class |
| Combined text fields | Richer signal than any single column |
| Stratified splits | Preserves fraud ratio across train/val/test |
| HF Trainer API | Handles training loop, gradient accumulation, device management cleanly |
| GPU-first | Full fine-tuning of 3 transformers requires CUDA for practical runtimes |

## Dependencies

- torch, transformers (model training)
- pandas, numpy, scikit-learn (data processing, metrics, splitting)
- matplotlib, seaborn (visualization)
- streamlit (web app)
- pytesseract, Pillow (OCR)
- re, html (text cleaning — stdlib)

## Out of Scope

- Hyperparameter search / Optuna
- MLflow or experiment tracking
- Deployment / containerization
- Data augmentation for class imbalance (using weighted loss instead)
