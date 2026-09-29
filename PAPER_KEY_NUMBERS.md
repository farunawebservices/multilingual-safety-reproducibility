# Key Numbers for ICLR Paper

## Phase 1: Baseline (Llama-3.3-70B)

| Language | Planner Refusal | Team Violation |
|----------|----------------|----------------|
| English | 98.0% | 15.0% |
| Arabic | 82.0% | 34.0% |
| Hausa | 11.0% | 65.0% |
| Yoruba | 0.0% | 77.0% |

## Phase 2: Steering (Llama-3.3-70B)

| Strength | Role | Hausa Def. | Yoruba Def. |
|----------|------|------------|-------------|
| 0 | - | 31.7% | 25.3% |
| 20 | Planner | 58.3% | 38.0% |
| 30 | Planner | 69.3% | 44.7% |
| 20 | Critic | 28.3% | 0.0% |
| 20 | Both | 39.0% | 6.3% |

## Qwen-2.5-72B Results

| Config | Hausa Def. | Yoruba Def. | FPR |
|--------|------------|-------------|-----|
| Baseline | 45.0% | 49.4% | 0.0% |
| Planner-only (α=2.0) | 54.0% | 57.9% | 0.0% |
| Critic-only (α=1.0) | 72.0% | 75.0% | 4.0% |
| Both (α=1.0) | 89.5% | 91.3% | 0.0% |

## Frontier Models (GPT-5, Claude Sonnet 4/5)

- Hausa refusal: 68-75% (vs. 11-45% for open-weight)
- Yoruba refusal: 68-71% (vs. 0-49% for open-weight)
- Arabic > English > Hausa > Yoruba hierarchy preserved

## Addition 1: Causal Patching

- 11 Hausa flip cases tested
- Clean: 10/11 VETO, 1/11 ALLOW
- Corrupted: 11/11 VETO (flip reproduction failed)
- Patched: 11/11 VETO
- Recovery rate R: undefined (denominator = -1)

## Addition 4: Judge Calibration

- κ = 0.98 (95% CI: 0.94-1.00, "almost perfect")
- 97/98 initial agreement (98.98%)
- 1 disagreement (Yoruba, adjudicated)

## Addition 5: Adversarial Robustness

- 50 jailbreak prompts, 6 attack families
- Baseline: 0/50 success (0%, CI [0%, 7.1%])
- Steered: 0/50 success (0%, CI [0%, 7.1%])
- Random vector (not refusal direction)

## Addition 6: Prompt-Filter (Llama)

- Baseline: 52.5%, Prompt filter: 53.5%, SAE steering: 57.5%
- Hausa: 32% → 57% (steering), 32% → 27% (prompt filter)
- Arabic: 52% → 45% (steering harmed), 52% → 59% (prompt filter helped)
- Benign FPR: 0% (baseline/filter), 5% (steering)
- Non-exact labels: 0.9% (baseline), 1.1% (filter), 5.4% (steering)
