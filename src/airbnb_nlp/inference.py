"""Load the fine-tuned model once and score batches of review text on CPU or GPU."""
from __future__ import annotations

import logging

from airbnb_nlp import DEFAULT_MODEL_ID

log = logging.getLogger(__name__)


class Predictor:
    """Wraps tokenizer + model. `predict` returns P(top_tier) for each text."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, max_length: int = 256, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading %s on %s", model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        self.model_version = f"{model_id}@{_revision(model_id)}"

    def predict(self, texts: list[str]) -> list[float]:
        torch = self._torch
        probs: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(texts[i:i + self.batch_size], truncation=True, padding=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
        return probs


def _revision(model_id: str) -> str:
    """Short commit id of the model on the Hugging Face Hub, so every prediction is traceable to a version."""
    try:
        from huggingface_hub import model_info

        return model_info(model_id).sha[:7]
    except Exception:  # local folder or offline
        return "local"
