import os
import json
import requests
from unittest.mock import patch

from tools.huggingface_client import HuggingFaceClient


def test_hf_client_calls_api(monkeypatch):
    # Use a fake token
    os.environ["HF_API_TOKEN"] = "fake-token"
    client = HuggingFaceClient()

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"generated_text": "ok"}
        result = client.call_model("some-model", "hello world", params={"max_length": 10})
        assert result == {"generated_text": "ok"}
        # Ensure the token and url were used
        args, kwargs = mock_post.call_args
        assert "api-inference.huggingface.co/models/some-model" in args[0]
        assert kwargs["headers"]["Authorization"] == "Bearer fake-token"