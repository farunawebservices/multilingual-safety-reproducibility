#!/usr/bin/env python3

import os, sys, requests, textwrap, traceback

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    raise SystemExit("OPENROUTER_API_KEY missing")

model = os.environ.get("OPENROUTER_MODEL", "~anthropic/claude-opus-latest")

prompt = textwrap.dedent("""\
You are auditing an empirical multilingual AI safety paper.

Task: list the models you will audit and the sections you will produce.
Do not draft the paper yet.

Return:
1. A bullet list of models (Llama 3.3 70B, Qwen2.5 72B, Gemma 4 31B, GPT-5, Claude Sonnet 4, Claude Sonnet 5, GPT-OSS 20B).
2. A bullet list of audit sections you will produce.
3. A one-paragraph description of your rules (no invented numbers, preserve denominators, etc.).
""")

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
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1024,
    },
    timeout=120,
)

if resp.status_code != 200:
    print(f"HTTP {resp.status_code}", file=sys.stderr)
    print(resp.text[:2000], file=sys.stderr)
    raise SystemExit(1)

payload = resp.json()
text = (payload["choices"][0]["message"].get("content") or "")
print(text)
