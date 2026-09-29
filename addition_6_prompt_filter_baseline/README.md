# Addition 6: Prompt-Based Safety Filter Baseline

## Goal
Compare activation steering vs. a simple prompt-based safety filter.

## Method
- Models: Llama-3.3-70B-Instruct, Qwen-2.5-72B-Instruct
- Dataset: Subset of v2 frozen dataset (100-200 prompts)
- Conditions:
  1. Baseline (existing results)
  2. Steering (existing results)
  3. Prompt filter (NEW - safety-focused system prompt)

## Safety Prompt


## Metrics
- Veto/refusal rate on harmful prompts
- False positive rate on benign prompts
- Comparison: Steering vs. Prompt filter effectiveness
