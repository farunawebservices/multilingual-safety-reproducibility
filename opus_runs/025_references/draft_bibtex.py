#!/usr/bin/env python3

import os, sys, requests

key = os.environ["OPENROUTER_API_KEY"]
model = os.environ.get("OPENROUTER_MODEL", "~anthropic/claude-opus-latest")

intro = open("/workspace/multilingual-multiagent-safety/paper_sections/01_introduction.tex", encoding="utf-8").read()
related = open("/workspace/multilingual-multiagent-safety/paper_sections/02_related_work.tex", encoding="utf-8").read()
methods = open("/workspace/multilingual-multiagent-safety/paper_sections/03_experimental_setup.tex", encoding="utf-8").read()
results = open("/workspace/multilingual-multiagent-safety/paper_sections/assembled/results_chapter_inlined.tex", encoding="utf-8").read()
discussion = open("/workspace/multilingual-multiagent-safety/paper_sections/07_discussion_opening.tex", encoding="utf-8").read()
conclusion = open("/workspace/multilingual-multiagent-safety/paper_sections/08_conclusion.tex", encoding="utf-8").read()
prompt = open("/workspace/multilingual-multiagent-safety/opus_runs/025_references/prompt.txt", encoding="utf-8").read()

user_content = f"{prompt}\n\n=== INTRODUCTION ===\n{intro}\n\n=== RELATED WORK ===\n{related}\n\n=== METHODS ===\n{methods}\n\n=== RESULTS ===\n{results}\n\n=== DISCUSSION ===\n{discussion}\n\n=== CONCLUSION ===\n{conclusion}"

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
        "max_tokens": 32768,
    },
    timeout=900,
)

if resp.status_code != 200:
    print(f"HTTP {resp.status_code}", file=sys.stderr)
    print(resp.text[:2000], file=sys.stderr)
    sys.exit(1)

payload = resp.json()
print((payload["choices"][0]["message"].get("content") or ""))
