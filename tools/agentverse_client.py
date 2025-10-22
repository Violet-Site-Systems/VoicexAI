"""
Minimal AgentVerse/ASI client wrapper used by the Cognitive Core.

Reads AGENTVERSE_API_KEY and AGENTVERSE_API_URL from env.
Provides call_model(model, prompt, params...) which returns parsed JSON/text.

This is intentionally small and tolerant: if no API key is present the client
reports itself as disabled and callers should fallback to local logic.
"""
import os
import requests
import json
from typing import Any, Dict, Optional


class AgentVerseClient:
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or os.environ.get("AGENTVERSE_API_URL", "https://api.agentverse.ai")
        self.api_key = api_key or os.environ.get("AGENTVERSE_API_KEY")
        self.enabled = bool(self.api_key)

    def call_model(self, model: str, prompt: str, params: Optional[Dict[str, Any]] = None, timeout: int = 60) -> Dict[str, Any]:
        """Call a hosted model and return a parsed response.

        This uses a generic POST to {api_url}/v1/models/{model}/invoke with
        JSON {"prompt": ..., "params": ...} and Authorization header.
        The exact AgentVerse API may differ; this wrapper can be adapted.
        """
        if not self.enabled:
            raise RuntimeError("AgentVerse client disabled: AGENTVERSE_API_KEY not set in environment")

        url = f"{self.api_url.rstrip('/')}/v1/models/{model}/invoke"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {"prompt": prompt, "params": params or {}}
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        # try parse JSON
        try:
            return resp.json()
        except Exception:
            return {"text": resp.text}


def get_client() -> AgentVerseClient:
    return AgentVerseClient()
