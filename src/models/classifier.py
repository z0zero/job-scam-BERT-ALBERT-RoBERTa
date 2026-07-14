import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
if os.name == "nt":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_SNAPSHOT_FILES = [
    "config.json",
    "model.safetensors",
    "model_meta.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


class ScamClassifier:
    def __init__(self, model_dir: str = "./best_model", max_len: int = 256):
        self.local_model_dir = Path(model_dir)

        # Digunakan saat deployment.
        self.model_id = os.getenv("HF_MODEL_ID", "").strip()

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

        model_source = self.local_model_dir
        if self.model_id:
            started_at = time.perf_counter()
            print(
                f"[model] snapshot download started: {self.model_id}",
                flush=True,
            )
            model_source = Path(
                snapshot_download(
                    repo_id=self.model_id,
                    token=False,
                    allow_patterns=MODEL_SNAPSHOT_FILES,
                )
            )
            print(
                "[model] snapshot download completed in "
                f"{time.perf_counter() - started_at:.1f}s",
                flush=True,
            )

        started_at = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_source),
            local_files_only=True,
        )
        print(
            "[model] tokenizer loaded in "
            f"{time.perf_counter() - started_at:.1f}s",
            flush=True,
        )

        started_at = time.perf_counter()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_source),
            local_files_only=True,
        )
        print(
            "[model] weights loaded in "
            f"{time.perf_counter() - started_at:.1f}s",
            flush=True,
        )

        self.model.eval()
        self.model.to(self.device)

        meta_path = model_source / "model_meta.json"
        with open(meta_path, "r", encoding="utf-8") as file:
            self.meta = json.load(file)
        print(f"[model] classifier ready on {self.device}", flush=True)

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
