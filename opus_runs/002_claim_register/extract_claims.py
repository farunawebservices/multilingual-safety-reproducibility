#!/usr/bin/env python3

import os, sys, requests, textwrap

key = os.environ["OPENROUTER_API_KEY"]
model = os.environ.get("OPENROUTER_MODEL", "~anthropic/claude-opus-latest")

audit_text = open("/workspace/multilingual-multiagent-safety/opus_runs/001_evidence_audit/response.md", encoding="utf-8").read()
prompt = open("/workspace/multilingual-multiagent-safety/opus_runs/002_claim_register/prompt.txt", encoding="utf-8").read()

user_content = f"{prompt}\n\n=== PRIOR AUDIT ===\n{audit_text}"

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
        "max_tokens": 8192,
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
