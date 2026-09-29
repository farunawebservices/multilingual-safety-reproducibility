#!/usr/bin/env python3

import os
import sys
import requests

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    raise SystemExit("OPENROUTER_API_KEY is not set")

model = os.environ.get(
    "OPENROUTER_MODEL",
    "~anthropic/claude-opus-latest",
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
        "max_tokens": 256,
    },
    timeout=120,
)

if response.status_code != 200:
    print(f"HTTP {response.status_code}", file=sys.stderr)
    print(response.text[:2000], file=sys.stderr)
    raise SystemExit(1)

payload = response.json()
message = payload["choices"][0]["message"]
text = (message.get("content") or "").strip()
print(text)
