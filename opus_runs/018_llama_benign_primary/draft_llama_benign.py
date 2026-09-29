#!/usr/bin/env python3

import os
import sys
import requests

key = os.environ["OPENROUTER_API_KEY"]
model = os.environ.get("OPENROUTER_MODEL", "~anthropic/claude-opus-latest")

audit = open("/workspace/multilingual-multiagent-safety/opus_runs/001_evidence_audit/response.md", encoding="utf-8").read()
claims = open("/workspace/multilingual-multiagent-safety/opus_runs/002_claim_register/claim_register.csv", encoding="utf-8").read()
outline = open("/workspace/multilingual-multiagent-safety/opus_runs/003_outline/outline.md", encoding="utf-8").read()
setup = open("/workspace/multilingual-multiagent-safety/paper_sections/03_experimental_setup.tex", encoding="utf-8").read()
llama_harm = open("/workspace/multilingual-multiagent-safety/paper_sections/04_1_llama_primary.tex", encoding="utf-8").read()
qwen = open("/workspace/multilingual-multiagent-safety/paper_sections/04_2_qwen_primary.tex", encoding="utf-8").read()
gemma = open("/workspace/multilingual-multiagent-safety/paper_sections/04_3_gemma_primary.tex", encoding="utf-8").read()
gptoss = open("/workspace/multilingual-multiagent-safety/paper_sections/04_4_gptoss_primary.tex", encoding="utf-8").read()
gpt5 = open("/workspace/multilingual-multiagent-safety/paper_sections/05_1_gpt5_baseline.tex", encoding="utf-8").read()
claude = open("/workspace/multilingual-multiagent-safety/paper_sections/05_2_claude_baseline.tex", encoding="utf-8").read()
prompt = open("/workspace/multilingual-multiagent-safety/opus_runs/018_llama_benign_primary/prompt.txt", encoding="utf-8").read()

user_content = (
    f"{prompt}\n\n=== AUDIT ===\n{audit}"
    f"\n\n=== CLAIMS ===\n{claims}"
    f"\n\n=== OUTLINE ===\n{outline}"
    f"\n\n=== SECTION 3 ===\n{setup}"
    f"\n\n=== SECTION 4.1 ===\n{llama_harm}"
    f"\n\n=== SECTION 4.2 ===\n{qwen}"
    f"\n\n=== SECTION 4.3 ===\n{gemma}"
    f"\n\n=== SECTION 4.4 ===\n{gptoss}"
    f"\n\n=== SECTION 5.1 ===\n{gpt5}"
    f"\n\n=== SECTION 5.2 ===\n{claude}"
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
        "max_tokens": 12288,
    },
    timeout=900,
)

if response.status_code != 200:
    print(f"HTTP {response.status_code}", file=sys.stderr)
    print(response.text[:2000], file=sys.stderr)
    raise SystemExit(1)

payload = response.json()
print((payload["choices"][0]["message"].get("content") or ""))
