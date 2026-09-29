#!/usr/bin/env python3
"""
Addition 4: Sample outputs for human annotation (Judge Calibration Study)
Sample 100 outputs stratified across 7 languages (~14 per language)
"""

import json
import random
from pathlib import Path

# Use the full dataset
DATASET_FILE = "data/v2/processed/frozen_dataset_v2.jsonl"
LANGUAGES = ["ar", "en", "ha", "hi", "ko", "sw", "yo"]  # 7 languages
SAMPLE_PER_LANG = 14  # ~14 per language = 98 total

random.seed(42)

# Load dataset
prompts_by_lang = {lang: [] for lang in LANGUAGES}
with open(DATASET_FILE, 'r') as f:
    for line in f:
        row = json.loads(line)
        lang = row.get('language', '')
        if lang in LANGUAGES:
            prompts_by_lang[lang].append({
                'prompt_hash': row.get('prompt_hash', ''),
                'text': row.get('translated_text', ''),
                'harm_label': row.get('harm_label', ''),
                'language': lang
            })

# Sample stratified by language and harm type
sampled = []
for lang in LANGUAGES:
    lang_prompts = prompts_by_lang[lang]
    harmful = [p for p in lang_prompts if p['harm_label'] == 'harmful']
    benign = [p for p in lang_prompts if p['harm_label'] == 'benign']

    # Sample ~7 harmful and ~7 benign per language
    n_harmful = min(7, len(harmful))
    n_benign = min(7, len(benign))

    sampled_harmful = random.sample(harmful, n_harmful)
    sampled_benign = random.sample(benign, n_benign)

    for p in sampled_harmful + sampled_benign:
        sampled.append({
            'annotation_id': f"ann_{len(sampled)+1:03d}",
            'lang': p['language'],
            'harm_label': p['harm_label'],
            'prompt_hash': p['prompt_hash'][:20],
            'text': p['text'],
            'human_label': None,  # To be filled by annotators
            'annotator_1': None,
            'annotator_2': None,
            'adjudicator': None
        })

    print(f"{lang}: {n_harmful} harmful + {n_benign} benign = {n_harmful + n_benign} samples")

# Save for annotation
output_file = "addition_4_calibration/annotation_samples.jsonl"
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    for r in sampled:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"\n✓ Saved {len(sampled)} samples to {output_file}")
print(f"  Languages: {LANGUAGES}")
print(f"  Total: {len(sampled)} (~14 per language)")
