#!/usr/bin/env python3

import os
import sys
import requests

key = os.environ["OPENROUTER_API_KEY"]
model = os.environ.get("OPENROUTER_MODEL", "~anthropic/claude-opus-latest")

audit = open("/workspace/multilingual-multiagent-safety/opus_runs/001_evidence_audit/response.md", encoding="utf-8").read()
claims = open("/workspace/multilingual-multiagent-safety/opus_runs/002_claim_register/claim_register.csv", encoding="utf-8").read()
outline = open("/workspace/multilingual-multiagent-safety/opus_runs/003_outline/outline.md", encoding="utf-8").read()
results = open("/workspace/multilingual-multiagent-safety/paper_sections/assembled/results_chapter_inlined.tex", encoding="utf-8").read()
prompt = open("/workspace/multilingual-multiagent-safety/opus_runs/020_discussion_opening/prompt.txt", encoding="utf-8").read()

user_content = (
    f"{prompt}\n\n=== AUDIT ===\n{audit}"
    f"\n\n=== CLAIMS ===\n{claims}"
    f"\n\n=== OUTLINE ===\n{outline}"
    f"\n\n=== RESULTS CHAPTER ===\n{results}"
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
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0,
        "max_tokens": 8192,
    },
    timeout=900,
)

if response.status_code != 200:
    print(f"HTTP {response.status_code}", file=sys.stderr)
    print(response.text[:2000], file=sys.stderr)
    raise SystemExit(1)

payload = response.json()
print((payload["choices"][0]["message"].get("content") or ""))
