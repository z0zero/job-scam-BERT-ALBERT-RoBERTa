import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class ScamClassifier:
    def __init__(self, model_dir: str = "./best_model", max_len: int = 256):
        self.local_model_dir = Path(model_dir)

        # Digunakan saat deployment.
        self.model_id = os.getenv("HF_MODEL_ID", "").strip()
        self.hf_token = os.getenv("HF_TOKEN", "").strip() or None

        # Jika HF_MODEL_ID tidak ada, tetap memakai folder lokal.
        self.model_source = self.model_id or str(self.local_model_dir)

        self.max_len = max_len
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.tokenizer = None
        self.meta = None

    def load_model(self) -> None:
        """Load model, tokenizer, and deployment metadata."""
        if self.model is not None:
            return

        auth_options = {}

        if self.hf_token:
            auth_options["token"] = self.hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            **auth_options,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_source,
            **auth_options,
        )

        self.model.eval()
        self.model.to(self.device)

        if self.model_id:
            meta_path = hf_hub_download(
                repo_id=self.model_id,
                filename="model_meta.json",
                token=self.hf_token,
            )
        else:
            meta_path = self.local_model_dir / "model_meta.json"

        with open(meta_path, "r", encoding="utf-8") as file:
            self.meta = json.load(file)

    def classify_text(self, text: str) -> tuple[str, float]:
        """Classify a job posting and return its label and confidence."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("The classifier model has not been loaded.")

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
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probabilities = F.softmax(outputs.logits, dim=-1)

        predicted_class = torch.argmax(
            probabilities,
            dim=-1,
        ).item()

        confidence = probabilities[0][predicted_class].item()

        label = (
            "Potential Scam"
            if predicted_class == 1
            else "Legitimate Job"
        )

        return label, confidence