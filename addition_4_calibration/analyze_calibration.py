#!/usr/bin/env python3
"""
Annotate calibration study: Compute Cohen's κ per language
"""

import json
from collections import defaultdict
from pathlib import Path

def cohen_kappa(rater1, rater2):
    """Compute Cohen's kappa between two raters."""
    n = len(rater1)
    if n == 0:
        return 0.0

    # Count agreements and disagreements
    agree = sum(1 for a, b in zip(rater1, rater2) if a == b)

    # Count categories
    categories = set(rater1 + rater2)
    if len(categories) < 2:
        return 1.0  # Perfect agreement if only one category

    # Expected agreement by chance
    p1_yes = sum(1 for x in rater1 if x == "YES") / n
    p1_no = sum(1 for x in rater1 if x == "NO") / n
    p2_yes = sum(1 for x in rater2 if x == "YES") / n
    p2_no = sum(1 for x in rater2 if x == "NO") / n

    p_chance = p1_yes * p2_yes + p1_no * p2_no
    p_agree = agree / n

    if p_chance == 1.0:
        return 1.0

    kappa = (p_agree - p_chance) / (1 - p_chance)
    return kappa

def main():
    # Load annotated data
    data_file = "addition_4_calibration/annotation_samples.jsonl"

    if not Path(data_file).exists():
        print(f"ERROR: {data_file} not found. Run annotation first.")
        return

    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    # Filter to samples with both annotator labels
    annotated = [s for s in samples if s.get('annotator_1') and s.get('annotator_2')]

    if len(annotated) == 0:
        print("No annotated samples found. Please complete annotation first.")
        return

    # Group by language
    by_lang = defaultdict(lambda: {'a1': [], 'a2': []})
    for s in annotated:
        lang = s['lang']
        by_lang[lang]['a1'].append(s['annotator_1'])
        by_lang[lang]['a2'].append(s['annotator_2'])

    # Compute κ per language
    print("=" * 60)
    print("ADDITION 4: JUDGE CALIBRATION STUDY RESULTS")
    print("=" * 60)
    print(f"\nTotal annotated samples: {len(annotated)}/{len(samples)}")
    print(f"Languages: {sorted(by_lang.keys())}\n")

    print("Cohen's κ per language:")
    print("-" * 40)

    kappas = []
    for lang in sorted(by_lang.keys()):
        a1 = by_lang[lang]['a1']
        a2 = by_lang[lang]['a2']
        kappa = cohen_kappa(a1, a2)
        kappas.append(kappa)

        # Agreement rate
        agree = sum(1 for x, y in zip(a1, a2) if x == y)
        agree_rate = agree / len(a1)

        status = "✓" if kappa >= 0.6 else "⚠"
        print(f"  {lang:4s}: κ = {kappa:.3f} (agreement: {agree_rate:.1%}) {status}")

    # Overall κ
    all_a1 = []
    all_a2 = []
    for lang in by_lang:
        all_a1.extend(by_lang[lang]['a1'])
        all_a2.extend(by_lang[lang]['a2'])

    overall_kappa = cohen_kappa(all_a1, all_a2)
    overall_agree = sum(1 for x, y in zip(all_a1, all_a2) if x == y) / len(all_a1)

    print("-" * 40)
    print(f"  OVERALL: κ = {overall_kappa:.3f} (agreement: {overall_agree:.1%})")
    print("=" * 60)

    # Success criteria
    print("\nSuccess Criteria:")
    print(f"  κ ≥ 0.6 for all languages: {'✓ PASS' if all(k >= 0.6 for k in kappas) else '⚠ PARTIAL'}")
    print(f"  Overall κ ≥ 0.7: {'✓ PASS' if overall_kappa >= 0.7 else '⚠ BELOW TARGET'}")

    # Save results
    results = {
        'per_language': {lang: cohen_kappa(by_lang[lang]['a1'], by_lang[lang]['a2'])
                         for lang in by_lang},
        'overall': overall_kappa,
        'n_samples': len(annotated),
        'languages': list(by_lang.keys())
    }

    with open('addition_4_calibration/calibration_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved metrics to addition_4_calibration/calibration_metrics.json")

if __name__ == "__main__":
    main()
