#!/usr/bin/env python3

import os, sys, requests

key = os.environ["OPENROUTER_API_KEY"]
model = os.environ.get("OPENROUTER_MODEL", "~anthropic/claude-opus-latest")

claims = open("/workspace/multilingual-multiagent-safety/opus_runs/002_claim_register/claim_register.csv", encoding="utf-8").read()
prompt = open("/workspace/multilingual-multiagent-safety/opus_runs/003_outline/outline_prompt.txt", encoding="utf-8").read()

user_content = f"{prompt}\n\n=== CLAIM REGISTER CSV ===\n{claims}"

resp = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/farunawebservices/multilingual-multiagent-safety",
        "X-Title": "Multilingual Multiagent Safety Paper",
    },
    json={
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0,
        "max_tokens": 12288,
    },
    timeout=600,
)

if resp.status_code != 200:
    print(f"HTTP {resp.status_code}", file=sys.stderr)
    print(resp.text[:2000], file=sys.stderr)
    sys.exit(1)

payload = resp.json()
text = (payload["choices"][0]["message"].get("content") or "")
print(text)
