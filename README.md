# Job Scam Detection using BERT, ALBERT & RoBERTa

[EN](README.md) | [ID](README.id.md)

A machine learning system that detects fraudulent job postings using transformer-based NLP models. Includes a research notebook for model training/comparison and a Streamlit web app for real-time inference.

## Results

Trained on the [EMSCAD dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) — 17,880 job postings, 4.84% labeled as fraudulent (highly imbalanced).

### Data Split

Stratified on the `fraudulent` label (random seed `42`) so the fraud ratio (~4.84%) is preserved across all splits.

| Split      | Samples | Percentage |
|------------|---------|------------|
| Train      | 12,516  | 70%        |
| Validation | 2,682   | 15%        |
| Test       | 2,682   | 15%        |

### Training Setup

- **Max sequence length:** 256 tokens
- **Batch size:** 16
- **Epochs:** 3
- **Optimizer:** AdamW — learning rate `2e-5`, weight decay `0.01`, warmup ratio `0.1`
- **Loss:** Weighted cross-entropy — class weights `[0.5254, 10.3267]` to counter imbalance
- **Model selection:** best checkpoint by validation F1 (`load_best_model_at_end=True`)
- **Hardware:** single Tesla T4 GPU (Google Colab)

### Test Set Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| BERT | 0.9870 | 0.9130 | 0.8077 | 0.8571 |
| **ALBERT** | **0.9892** | **0.9469** | **0.8231** | **0.8807** |
| RoBERTa | 0.9847 | 0.8678 | 0.8077 | 0.8367 |

**Best model: ALBERT** (albert-base-v2) — selected by highest F1 score.  
Download best_model: https://drive.google.com/file/d/1YZoeuoWHS_oLGF4bu6cW3ePcbaBGEdB2/view?usp=sharing

## Project Structure

```
├── app.py                       # Entry point (delegates to AppController)
├── src/                         # MVC application code
│   ├── controllers/
│   │   └── app_controller.py    # Orchestrates input → preprocess → predict → render
│   ├── models/
│   │   ├── classifier.py        # Loads ALBERT, runs inference → (label, confidence)
│   │   ├── ocr_engine.py        # Tesseract OCR wrapper
│   │   └── preprocessor.py      # Text cleaning (mirrors the notebook)
│   └── views/
│       └── main_view.py         # Streamlit UI rendering
├── research_pipeline.ipynb      # Model training & comparison notebook
├── requirements.txt             # Python dependencies
├── test_data.txt                # Two sample postings (legit + scam) for manual testing
├── best_model/                  # Exported ALBERT model + model_meta.json (git-ignored)
└── docs/                        # Design notes and planning docs
```

## Architecture

The app follows a Model–View–Controller split so the Streamlit UI, business logic, and ML concerns stay separate.

- **View** (`src/views/main_view.py`) — pure Streamlit rendering (header, input form, result display). Stateless; it receives data to render and callbacks to invoke.
- **Model** (`src/models/`) — domain logic, not ML weights. `classifier.py` loads the fine-tuned ALBERT and returns `(label, confidence)`; `preprocessor.py` replicates the notebook's text cleaning so training/inference stay in sync; `ocr_engine.py` wraps Tesseract for image uploads.
- **Controller** (`src/controllers/app_controller.py`) — wires everything: boots the classifier (cached via `@st.cache_resource`), feeds user input through the preprocessor and classifier, and hands results back to the view.

`app.py` is a thin entry point that instantiates `AppController`, so `streamlit run app.py` still works as before.

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for image upload)

The image upload feature requires Tesseract OCR to extract text from screenshots.

- **Windows**: Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). The default install path (`C:\Program Files\Tesseract-OCR`) is auto-detected by the app.
- **Linux**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`

> If you only use the text input feature, Tesseract is not required.

### 3. Get the model weights

The app loads `./best_model/` on startup and fails early if it's missing. Pick one path:

**Option A — Download the pre-trained ALBERT (fastest)**

[Download `best_model.zip`](https://drive.google.com/file/d/1YZoeuoWHS_oLGF4bu6cW3ePcbaBGEdB2/view?usp=sharing) and extract it at the project root so `./best_model/` contains the model weights, tokenizer, and `model_meta.json`.

**Option B — Train from scratch**

Open `research_pipeline.ipynb` in Jupyter or Google Colab. Download the [EMSCAD dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) as `fake_job_postings.csv` and place it in the project root. Run all cells — the best model is exported to `best_model/`.

A GPU is recommended (training takes ~2 hours on a Tesla T4).

### 4. Run the web app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. To sanity-check the setup, copy either sample from `test_data.txt` (one legitimate posting, one obvious scam) into the text input and click **Analyze**.

## Features

- **Text input** — paste a job description directly
- **Image upload** — upload a screenshot; text is extracted via OCR (pytesseract)
- **Confidence score** — shows prediction probability
- **Weighted loss** — handles class imbalance (4.84% fraud rate) using weighted cross-entropy

## Limitations & Responsible Use

Treat the model's output as a **screening signal, not a verdict.** Pair it with your own judgment and external checks (company registration, recruiter identity, domain age, up-front payment requests, etc.).

- **~18% of scams are missed.** Test recall is 0.8231 — roughly one in five fraudulent postings in our test set was labeled "Legitimate Job". Never treat a legitimate prediction as proof of safety.
- **~5% of flags are false alarms.** Test precision is 0.9469. A "Potential Scam" flag warrants a closer look, not automatic rejection.
- **English only.** EMSCAD is an entirely English-language dataset. Postings in Indonesian, Spanish, or other languages are out of distribution and results are unreliable.
- **Dataset drift.** EMSCAD was collected in 2014–2015. Newer scam patterns (crypto bait, AI-generated postings, sophisticated MLM funnels) may slip past a model trained on older data.
- **256-token truncation.** Postings longer than ~256 tokens are cut off — signals that appear only near the end of a long description are invisible to the model.
- **OCR reliability.** Image uploads depend on Tesseract. Low-resolution, rotated, or noisy screenshots produce bad text extraction, which degrades predictions.
- **Not legal or financial advice.** Do not use this tool as the sole basis for accepting, rejecting, or reporting a job offer.

## Tech Stack

- **Models**: BERT, ALBERT, RoBERTa (HuggingFace Transformers)
- **Training**: PyTorch, HuggingFace Trainer with custom weighted loss
- **App**: Streamlit
- **OCR**: pytesseract (requires [Tesseract](https://github.com/tesseract-ocr/tesseract) installed)
- **Metrics**: scikit-learn (accuracy, precision, recall, F1, confusion matrix)
