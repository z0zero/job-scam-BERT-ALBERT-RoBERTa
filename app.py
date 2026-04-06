"""
Job Scam Detection - Streamlit Web Application

Loads the best-performing NLP model exported from the research notebook
and provides a UI for classifying job postings as legitimate or fraudulent.
Supports text input and image upload (with OCR via pytesseract).
"""

import json
import html
import os
import re

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pytesseract

# Configure Tesseract OCR path for Windows
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

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


def clean_text(text):
    """Clean and normalize text — must match the notebook's preprocessing."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_text(text, model, tokenizer, device):
    """Classify a job posting text and return label + confidence."""
    text = clean_text(text)
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
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError:
        st.error(
            "Tesseract OCR is not installed. "
            "Please install it from https://github.com/UB-Mannheim/tesseract/wiki "
            "and add it to your PATH."
        )
        return ""


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
