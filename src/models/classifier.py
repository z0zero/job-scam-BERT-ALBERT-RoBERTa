import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ScamClassifier:
    def __init__(self, model_dir="./best_model", max_len=256):
        self.model_dir = model_dir
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.meta = None

    def load_model(self):
        """Load the saved model, tokenizer, and metadata."""
        if self.model is not None:
             return
             
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.model.to(self.device)

        meta_path = os.path.join(self.model_dir, "model_meta.json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

    def classify_text(self, text):
        """Classify a preprocessed job posting text and return label + confidence."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs.logits, dim=-1)

        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        label = "Potential Scam" if predicted_class == 1 else "Legitimate Job"
        return label, confidence
