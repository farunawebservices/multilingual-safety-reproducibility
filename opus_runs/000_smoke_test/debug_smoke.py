#!/usr/bin/env python3

import os
import sys
import requests
import json

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    raise SystemExit("OPENROUTER_API_KEY is not set")

model = os.environ.get(
    "OPENROUTER_MODEL",
    "anthropic/claude-opus-5",
)

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/farunawebservices/multilingual-multiagent-safety",
        "X-Title": "Multilingual Multiagent Safety Paper",
    },
    json={
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: OPUS_SMOKE_TEST_OK",
            }
        ],
        "temperature": 0,
        "max_tokens": 20,
    },
    timeout=120,
)

print("HTTP status:", response.status_code, file=sys.stderr)
print("Headers:", dict(response.headers), file=sys.stderr)
print("Body:", response.text[:4000], file=sys.stderr)

if response.status_code == 200:
    payload = response.json()
    print("\nParsed JSON keys:", list(payload.keys()), file=sys.stderr)
    print("choices:", payload.get("choices"), file=sys.stderr)
