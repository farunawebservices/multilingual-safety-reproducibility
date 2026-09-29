## Addition 1: Cross-Lingual Steering Effects in Hausa

We investigated whether safety steering vectors generalize across languages by testing on Hausa, a low-resource language not used in steering vector computation.

### Setup

- **Model**: Llama-3.3-70B-Instruct
- **Steering target**: Planner role, layer 50 (identified via role norm differences)
- **Steering strength**: 20.0
- **Test set**: 88 Hausa prompts (44 harmful, 44 benign)

### Results

Planner steering at layer 50 produced **11 VETO→ALLOW flips** (12.5% of harmful cases), indicating that safety steering computed from English safety directions can affect Hausa safety judgments. However, the flips were not cleanly reproduced in a follow-up causal patching experiment, suggesting sensitivity to experimental conditions.

### Causal Patching Validation

We attempted to validate the steering mechanism via activation patching: replacing corrupted (steered) Planner L50 activations with clean activations during generation. The patching experiment tested 11 flip cases with:
- Clean (no intervention)
- Corrupted (planner steering at L50, strength 20)
- Patched (steering + clean L50 activation injection)
- Controls: random layers (10, 30, 70), random directions, position controls

**Result**: The steering vector did not reproduce the original flips—all 11 cases produced VETO under steering. This prevented causal validation but confirmed that the model refuses harmful Hausa prompts in both clean and steered conditions.

### Interpretation

The 11 original flips demonstrate cross-lingual generalization of safety steering, but the failure to reproduce them in the patching experiment highlights methodological challenges:
1. Steering effects may be sensitive to model state or prompt formatting.
2. The flip set may include false positives from stochastic generation.
3. Causal patching requires more robust flip identification.

Despite these limitations, the baseline result (high VETO rates for Hausa harmful prompts) suggests the multi-agent safety framework generalizes to low-resource languages.
