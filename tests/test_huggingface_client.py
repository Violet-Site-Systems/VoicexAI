import json
import requests
from unittest.mock import patch

from tools.huggingface_client import HuggingFaceClient


def test_hf_client_inference_api_success():
    sample_text = "This is a test document."
    fake_resp = [{"summary_text": "Short summary."}]

    class FakeResponse:
        def __init__(self, j):
            self._j = j

        def raise_for_status(self):
            return

        def json(self):
            return self._j

    def fake_post(url, headers=None, json=None, timeout=None):
        assert "api-inference.huggingface.co/models" in url
        assert 'Authorization' in headers
        return FakeResponse(fake_resp)

    client = HuggingFaceClient(api_key="hf_test_key", model="sshleifer/distilbart-cnn-12-6")

    with patch('requests.post', fake_post):
        res = client.summarize(sample_text)
        assert isinstance(res, dict)
        assert res.get('summary_text') == 'Short summary.'


if __name__ == '__main__':
    test_hf_client_inference_api_success()
    print("HF client test passed")
