#!/usr/bin/env python3

import os, sys, requests

key = os.environ["OPENROUTER_API_KEY"]
model = os.environ.get("OPENROUTER_MODEL", "~anthropic/claude-opus-latest")

related = open("/workspace/multilingual-multiagent-safety/paper_sections/02_related_work.tex", encoding="utf-8").read()
bib = open("/workspace/multilingual-multiagent-safety/paper.bib", encoding="utf-8").read()
intro = open("/workspace/multilingual-multiagent-safety/paper_sections/01_introduction.tex", encoding="utf-8").read()
prompt = open("/workspace/multilingual-multiagent-safety/opus_runs/026_cited_related_work/prompt.txt", encoding="utf-8").read()

user_content = f"{prompt}\n\n=== CURRENT RELATED WORK ===\n{related}\n\n=== BIBTEX ===\n{bib}\n\n=== INTRODUCTION (context) ===\n{intro}"

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
        "max_tokens": 16384,
    },
    timeout=900,
)

if resp.status_code != 200:
    print(f"HTTP {resp.status_code}", file=sys.stderr)
    print(resp.text[:2000], file=sys.stderr)
    sys.exit(1)

payload = resp.json()
print((payload["choices"][0]["message"].get("content") or ""))
