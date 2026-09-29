# ICLR Paper: Summary and Next Steps

## Paper Status

**Title**: Where You Intervene Matters More Than Whether You Intervene: Role-Specific Steering for Cross-Lingual Multi-Agent Safety

**Draft**: `iclr_paper_complete.md` (comprehensive, all additions integrated)

**Key Numbers**: `PAPER_KEY_NUMBERS.md`

## Complete Experimental Portfolio

### Main Results (6 models × 4 languages × 3 roles)

1. **Llama-3.3-70B**: Planner steering works (31.7% → 69.3% Hausa), critic steering fails (0% Yoruba)
2. **Qwen-2.5-72B**: Joint defense optimal (89.5% Hausa, 91.3% Yoruba)
3. **Gemma-4-31B**: Planner-only optimal (Layer 39)
4. **GPT-OSS-20B**: Layer 22 steering effective (Arabic 85.7% → 92%)
5. **GPT-5**: Strong baseline (74% Hausa, 70% Yoruba)
6. **Claude Sonnet 4/5**: Strong baselines (72–75% Hausa/Yoruba)

### Additions 1–6 (Robustness)

1. **Addition 1**: Causal patching (11 Hausa cases, L50, R = -0.0, flip reproduction failed)
2. **Addition 2**: Matched controls (role-specificity rules out generic perturbation)
3. **Addition 3**: Hindi/Korean replication (70–80% recovery)
4. **Addition 4**: Judge calibration (κ = 0.73, substantial agreement)
5. **Addition 5**: Adversarial robustness (60%+ under 50 jailbreak attacks)
6. **Addition 6**: Prompt-filter baseline (85%+ defensiveness, complementary to steering)

## Paper Structure

1. **Abstract**: 250 words, all key results
2. **Introduction**: 1 page, motivation + contributions
3. **Related Work**: 0.5 page, cross-lingual safety + steering + multi-agent
4. **Framework**: 1 page, architecture + dataset + metrics
5. **Phase 1 Results**: 1 page, baselines for all 6 models
6. **Phase 2 Results**: 1.5 pages, steering results + layer ablation
7. **Mechanistic Analysis**: 1 page, why critic steering fails
8. **Additions**: 2 pages, all 6 robustness checks
9. **Discussion**: 1 page, implications + limitations
10. **Conclusion**: 0.5 page

**Total**: ~10 pages (ICLR main track)

## Next Steps

### 1. Create Figures (Priority: HIGH)

- **Figure 1**: Defensiveness by language (baseline vs. steered) for all 6 models
- **Figure 2**: Layer ablation curves (all 6 models)
- **Figure 3**: Role comparison (planner vs. critic vs. both)
- **Figure 4**: Addition results (calibration, adversarial, prompt-filter)

### 2. Write Appendix (Priority: MEDIUM)

- **Appendix A**: Full baseline tables for all models
- **Appendix B**: All layer ablation data
- **Appendix C**: Frontier model details
- **Appendix D**: Addition reports (copy from existing .md files)

### 3. Format for ICLR (Priority: MEDIUM)

- Convert to LaTeX using ICLR 2026 template
- Add proper citations
- Create BibTeX file with all references
- Ensure page limits (10 pages main + unlimited appendix)

### 4. Final Checks (Priority: HIGH)

- Verify all numbers match source JSONL files
- Check consistency across tables
- Ensure all claims are backed by data
- Run plagiarism check (if available)

### 5. Submission Prep (Priority: HIGH)

- Write author response template
- Prepare supplementary material (code, data, vectors)
- Create anonymized repository (if double-blind)
- Complete submission form

## Timeline

- **Day 1–2**: Create figures, write appendix
- **Day 3**: Format LaTeX, add citations
- **Day 4**: Final checks, author response
- **Day 5**: Submit

## GitHub Backup Status

✅ All results committed: `recovery-v2-20260831`  
✅ Latest commit: `a2e0479`  
✅ Repository: https://github.com/farunawebservices/multilingual-multiagent-safety

## Files to Preserve

- `iclr_paper_complete.md` — Full paper draft
- `PAPER_KEY_NUMBERS.md` — Key statistics
- `ADDITIONS_COMPLETE_STATUS.md` — All additions summary
- `addition_1_paper_draft.md` — Addition 1 section
- All JSONL outputs in `addition_*/` directories
- All steering vectors in `artifacts/*/vectors/`

---

**Instance Time**: <3 hours remaining  
**Recommendation**: Focus on figures + LaTeX formatting, then submit.
