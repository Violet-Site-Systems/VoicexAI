"""Quick helper to call ASI1.ai 'asi1-mini' chat completions.

Usage:
  export ASI_ONE_API_KEY=sk_xxx...
  python scripts/asi1_chat.py --prompt "Hello"

This script intentionally reads the API key from the environment variable
`ASI_ONE_API_KEY`. Do NOT commit your API key to source control.
"""

import os
import argparse
import requests
import sys
from typing import Any, Dict

API_URL = "https://api.asi1.ai/v1/chat/completions"
ENV_KEY = "ASI_ONE_API_KEY"


def chat(prompt: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": "asi1-mini",
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(API_URL, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    # Safe extraction with fallback
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        return f"Unexpected response format: {j}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Call ASI1.ai mini chat")
    parser.add_argument(
        "--prompt", "-p",
        default="Hello! How can you help me today?",
        help="Prompt to send"
    )
    args = parser.parse_args(argv)

    api_key = os.getenv(ENV_KEY)
    if not api_key:
        print(f"Please set the environment variable {ENV_KEY} with your "
              f"ASI1.ai API key.")
        sys.exit(1)

    try:
        out = chat(args.prompt, api_key)
        print(out)
    except requests.HTTPError as e:
        print(f"HTTP error: {e} - {e.response.text}")
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
