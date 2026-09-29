#!/usr/bin/env python3

import os, sys, tarfile, requests, traceback, textwrap, json

def main():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("OPENROUTER_MODEL", "~anthropic/claude-opus-latest")

    packet_path = "/workspace/multilingual-multiagent-safety/opus_packet.tar.gz"
    prompt_path = "/workspace/multilingual-multiagent-safety/opus_runs/001_evidence_audit/prompt.txt"

    try:
        prompt_text = open(prompt_path, encoding="utf-8").read()
    except Exception as e:
        print(f"ERROR reading prompt: {e}", file=sys.stderr)
        sys.exit(1)

    text_files = []
    total_chars = 0
    MAX_CHARS = 1_200_000

    try:
        with tarfile.open(packet_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name.lower()
                if not (name.endswith(".md") or name.endswith(".csv") or name.endswith(".json") or name.endswith(".tex")):
                    continue
                if total_chars >= MAX_CHARS:
                    break
                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read().decode("utf-8", errors="replace")
                if total_chars + len(content) > MAX_CHARS:
                    content = content[:MAX_CHARS - total_chars] + "\n\n[TRUNCATED]"
                text_files.append((member.name, content))
                total_chars += len(content)
    except Exception as e:
        print(f"ERROR reading packet: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    context_parts = []
    for path, content in sorted(text_files):
        context_parts.append(f"\n\n===== {path} =====\n{content}")

    context = "".join(context_parts)
    user_content = f"{prompt_text}\n\n=== EVIDENCE PACKET (capped at {MAX_CHARS} chars) ==={context}"

    print(f"DEBUG: prompt_chars={len(prompt_text)}, context_chars={len(context)}, total={len(user_content)}", file=sys.stderr)

    try:
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
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0,
                "max_tokens": 32768,
            },
            timeout=900,
        )
    except Exception as e:
        print(f"ERROR calling OpenRouter: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Save raw body for debugging
    with open("/workspace/multilingual-multiagent-safety/opus_runs/001_evidence_audit/raw_response.json", "w", encoding="utf-8") as f:
        f.write(response.text)

    if response.status_code != 200:
        print(f"HTTP {response.status_code}", file=sys.stderr)
        print(response.text[:3000], file=sys.stderr)
        sys.exit(1)

    try:
        payload = response.json()
    except Exception as e:
        print(f"ERROR parsing JSON: {e}", file=sys.stderr)
        print("Raw response preview:", response.text[:2000], file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    try:
        message = payload["choices"][0]["message"]
        text = (message.get("content") or "")
        print(text)
    except Exception as e:
        print(f"ERROR extracting content: {e}", file=sys.stderr)
        print("Full payload keys:", list(payload.keys()), file=sys.stderr)
        print("choices:", payload.get("choices"), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
