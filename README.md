# Job Scam Detection using BERT, ALBERT & RoBERTa

A machine learning system that detects fraudulent job postings using transformer-based NLP models. Includes a research notebook for model training/comparison and a Streamlit web app for real-time inference.

## Results

Trained on the [EMSCAD dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (17,880 job postings, 4.84% fraudulent).

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| BERT | 0.9870 | 0.9130 | 0.8077 | 0.8571 |
| **ALBERT** | **0.9892** | **0.9469** | **0.8231** | **0.8807** |
| RoBERTa | 0.9847 | 0.8678 | 0.8077 | 0.8367 |

**Best model: ALBERT** (albert-base-v2) — selected by highest F1 score.

## Project Structure

```
├── research_pipeline.ipynb   # Model training & comparison notebook
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── test_data.txt             # Sample test data for manual testing
├── best_model/               # Exported ALBERT model (git-ignored)
└── docs/                     # Design specs and implementation plan
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Train the model

Open `research_pipeline.ipynb` in Jupyter or Google Colab. Download the [EMSCAD dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) as `fake_job_postings.csv` and place it in the project root. Run all cells — the best model is exported to `./best_model/`.

A GPU is recommended (training takes ~2 hours on a Tesla T4).

### 3. Run the web app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Features

- **Text input** — paste a job description directly
- **Image upload** — upload a screenshot; text is extracted via OCR (pytesseract)
- **Confidence score** — shows prediction probability
- **Weighted loss** — handles class imbalance (4.84% fraud rate) using weighted cross-entropy

## Tech Stack

- **Models**: BERT, ALBERT, RoBERTa (HuggingFace Transformers)
- **Training**: PyTorch, HuggingFace Trainer with custom weighted loss
- **App**: Streamlit
- **OCR**: pytesseract (requires [Tesseract](https://github.com/tesseract-ocr/tesseract) installed)
- **Metrics**: scikit-learn (accuracy, precision, recall, F1, confusion matrix)
