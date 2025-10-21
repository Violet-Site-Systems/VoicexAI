"""Hugging Face client wrapper.

Supports calling the HF Inference API (requires `HUGGINGFACE_API_KEY`) and a
local transformers pipeline fallback when no API key is present.
"""
import os
import requests
from typing import Optional, Dict, Any

try:
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - optional local fallback
    hf_pipeline = None


class HuggingFaceClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "sshleifer/distilbart-cnn-12-6"):
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        self.model = model
        self.local = hf_pipeline is not None

    def summarize(self, text: str, max_length: int = 120, min_length: int = 30) -> Dict[str, Any]:
        """Return a summary using the configured provider.

        If HUGGINGFACE_API_KEY is provided, call the Inference API; otherwise
        attempt a local pipeline.
        """
        if self.api_key:
            url = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"inputs": text, "parameters": {"max_length": max_length, "min_length": min_length}}
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            # HF Inference API returns list of outputs for text-generation/summarization
            out = resp.json()
            if isinstance(out, list) and out:
                return {"summary_text": out[0].get("summary_text") or out[0].get("generated_text")}
            if isinstance(out, dict) and out.get("error"):
                raise RuntimeError(f"HF inference error: {out.get('error')}")
            return {"summary_text": str(out)}

        # Local fallback
        if self.local:
            summarizer = hf_pipeline("summarization", model=self.model)
            out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return {"summary_text": out[0]["summary_text"] if out else ""}

        raise RuntimeError("No Hugging Face provider available: set HUGGINGFACE_API_KEY or install transformers")


def get_client(api_key: Optional[str] = None, model: Optional[str] = None) -> HuggingFaceClient:
    return HuggingFaceClient(api_key=api_key, model=model or os.environ.get("HUGGINGFACE_MODEL_SUMMARY", "sshleifer/distilbart-cnn-12-6"))
