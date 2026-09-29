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
llama_benign = open("/workspace/multilingual-multiagent-safety/paper_sections/04_5_llama_benign.tex", encoding="utf-8").read()
gpt5 = open("/workspace/multilingual-multiagent-safety/paper_sections/05_1_gpt5_baseline.tex", encoding="utf-8").read()
claude = open("/workspace/multilingual-multiagent-safety/paper_sections/05_2_claude_baseline.tex", encoding="utf-8").read()
prompt = open("/workspace/multilingual-multiagent-safety/opus_runs/019_additional_analyses/prompt.txt", encoding="utf-8").read()

# Load key Addition artifacts so Opus can see their contents directly
add1_audit = open("/workspace/multilingual-multiagent-safety/opus_packet/metrics/addition1_llama33_hausa_patching_v3_audit.csv", encoding="utf-8").read()
add1_table = open("/workspace/multilingual-multiagent-safety/opus_packet/tables/addition1_llama33_hausa_patching_v3_table.tex", encoding="utf-8").read()
add1_handoff = open("/workspace/multilingual-multiagent-safety/opus_packet/handoffs/addition1_llama33_hausa_patching_v3_OPUS_HANDOFF.md", encoding="utf-8").read()
add1_manifest = open("/workspace/multilingual-multiagent-safety/opus_packet/manifests/addition1_llama33_hausa_patching_v3_artifact_manifest.json", encoding="utf-8").read()

add2_status = open("/workspace/multilingual-multiagent-safety/paper_artifacts/metrics/addition_2_status.md", encoding="utf-8").read()
add23_boundary = open("/workspace/multilingual-multiagent-safety/opus_packet/evidence/additions_2_3_claim_boundary_audit.md", encoding="utf-8").read()

add4_metrics = open("/workspace/multilingual-multiagent-safety/opus_packet/metrics/addition4_judge_calibration_metrics.csv", encoding="utf-8").read()
add4_table = open("/workspace/multilingual-multiagent-safety/opus_packet/metrics/addition4_calibration_paper_table.csv", encoding="utf-8").read()
add4_tex = open("/workspace/multilingual-multiagent-safety/opus_packet/tables/addition4_calibration_agreement.tex", encoding="utf-8").read()
add4_handoff = open("/workspace/multilingual-multiagent-safety/opus_packet/handoffs/addition4_calibration_OPUS_HANDOFF.md", encoding="utf-8").read()

add5_metrics = open("/workspace/multilingual-multiagent-safety/opus_packet/metrics/addition5_adversarial_robustness_metrics.csv", encoding="utf-8").read()
add5_table = open("/workspace/multilingual-multiagent-safety/opus_packet/metrics/addition5_adversarial_robustness_table.csv", encoding="utf-8").read()
add5_tex = open("/workspace/multilingual-multiagent-safety/opus_packet/tables/addition5_adversarial_robustness.tex", encoding="utf-8").read()
add5_handoff = open("/workspace/multilingual-multiagent-safety/opus_packet/handoffs/addition5_adversarial_robustness_OPUS_HANDOFF.md", encoding="utf-8").read()

add6_metrics = open("/workspace/multilingual-multiagent-safety/opus_packet/metrics/addition6_prompt_filter_baseline_metrics.csv", encoding="utf-8").read()
add6_table = open("/workspace/multilingual-multiagent-safety/opus_packet/metrics/addition6_prompt_filter_baseline_table.csv", encoding="utf-8").read()
add6_tex = open("/workspace/multilingual-multiagent-safety/opus_packet/tables/addition6_prompt_filter_baseline.tex", encoding="utf-8").read()
add6_handoff = open("/workspace/multilingual-multiagent-safety/opus_packet/handoffs/addition6_prompt_filter_baseline_OPUS_HANDOFF.md", encoding="utf-8").read()
add6_manifest = open("/workspace/multilingual-multiagent-safety/opus_packet/manifests/addition6_prompt_filter_baseline_artifact_manifest.json", encoding="utf-8").read()

cross_phase_tex = open("/workspace/multilingual-multiagent-safety/opus_packet/tables/llama33_turbo_alpha30_cross_phase_comparison.tex", encoding="utf-8").read()

user_content = (
    f"{prompt}\n\n=== AUDIT ===\n{audit}"
    f"\n\n=== CLAIMS ===\n{claims}"
    f"\n\n=== OUTLINE ===\n{outline}"
    f"\n\n=== SECTION 3 ===\n{setup}"
    f"\n\n=== SECTION 4.1 ===\n{llama_harm}"
    f"\n\n=== SECTION 4.2 ===\n{qwen}"
    f"\n\n=== SECTION 4.3 ===\n{gemma}"
    f"\n\n=== SECTION 4.4 ===\n{gptoss}"
    f"\n\n=== SECTION 4.5 ===\n{llama_benign}"
    f"\n\n=== SECTION 5.1 ===\n{gpt5}"
    f"\n\n=== SECTION 5.2 ===\n{claude}"
    f"\n\n=== ADDITION 1 (causal patching) ===\n{add1_audit}\n\n{add1_table}\n\n{add1_handoff}\n\n{add1_manifest}"
    f"\n\n=== ADDITION 2 (matched controls) ===\n{add2_status}\n\n{add23_boundary}"
    f"\n\n=== ADDITION 4 (calibration) ===\n{add4_metrics}\n\n{add4_table}\n\n{add4_tex}\n\n{add4_handoff}"
    f"\n\n=== ADDITION 5 (adversarial) ===\n{add5_metrics}\n\n{add5_table}\n\n{add5_tex}\n\n{add5_handoff}"
    f"\n\n=== ADDITION 6 (prompt filter) ===\n{add6_metrics}\n\n{add6_table}\n\n{add6_tex}\n\n{add6_handoff}\n\n{add6_manifest}"
    f"\n\n=== CROSS-PHASE (Llama Turbo) ===\n{cross_phase_tex}"
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
        "max_tokens": 16384,
    },
    timeout=900,
)

if response.status_code != 200:
    print(f"HTTP {response.status_code}", file=sys.stderr)
    print(response.text[:2000], file=sys.stderr)
    raise SystemExit(1)

payload = response.json()
print((payload["choices"][0]["message"].get("content") or ""))
