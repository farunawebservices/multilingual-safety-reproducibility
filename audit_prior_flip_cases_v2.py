#!/usr/bin/env python3
"""
Audit script to verify Hausa VETO→ALLOW flip cases from prior Llama-3.3-70B runs.

Data structure:
- turbo_steering_results_phase2.jsonl: has is_vetoed, config, strength, language, harmfulness
- baseline_completed_backup (1).jsonl: has is_vetoed, strength=0.0

Usage:
  python audit_prior_flip_cases_v2.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

# File paths
STEERING_FILE = "artifacts/prior_llama33_phase2/results/turbo_steering_results_phase2.jsonl"
BASELINE_FILE = "artifacts/prior_llama33_phase2/results/baseline_completed_backup (1).jsonl"
OUTPUT_FILE = "artifacts/prior_llama33_phase2/transcripts/verified_flips_hausa.jsonl"

def load_data(path):
    """Load JSONL and return list of rows"""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
    return rows

def main():
    print("Loading steering results...")
    steering = load_data(STEERING_FILE)
    print(f"  Loaded {len(steering)} rows")
    
    print("Loading baseline...")
    baseline = load_data(BASELINE_FILE)
    print(f"  Loaded {len(baseline)} rows")
    
    # Index baseline by id
    baseline_by_id = {}
    for row in baseline:
        row_id = row.get('id')
        if row_id:
            baseline_by_id[row_id] = row
    
    print(f"  Baseline indexed: {len(baseline_by_id)} unique IDs")
    
    # Find Hausa harmful prompts with critic steering
    print("\nFiltering for Hausa harmful + critic steering...")
    
    # Group steering by ID
    steering_by_id = defaultdict(list)
    for row in steering:
        row_id = row.get('id')
        if row_id:
            steering_by_id[row_id].append(row)
    
    # Find flips: baseline VETO, critic-steered ALLOW
    flips = []
    non_flips = []
    
    for row_id, steered_rows in steering_by_id.items():
        # Get baseline
        base_row = baseline_by_id.get(row_id)
        if not base_row:
            continue
        
        # Check if Hausa harmful
        if base_row.get('language', '').lower() != 'hausa' or base_row.get('harmfulness') != 'harmful':
            continue
        
        base_vetoed = base_row.get('is_vetoed', False)
        
        # Find critic-only steering at strength 20
        critic_s20 = None
        for s_row in steered_rows:
            if (s_row.get('config') == 'critic_only' and 
                s_row.get('strength') == 20.0 and
                s_row.get('language', '').lower() == 'hausa'):
                critic_s20 = s_row
                break
        
        if not critic_s20:
            continue
        
        critic_vetoed = critic_s20.get('is_vetoed', False)
        
        # Check for flip
        if base_vetoed and not critic_vetoed:
            # VETO → ALLOW flip
            flips.append({
                'id': row_id,
                'baseline_vetoed': base_vetoed,
                'critic_vetoed': critic_vetoed,
                'category': base_row.get('category', ''),
                'baseline_row': base_row,
                'critic_row': critic_s20,
            })
        else:
            non_flips.append({
                'id': row_id,
                'baseline_vetoed': base_vetoed,
                'critic_vetoed': critic_vetoed,
            })
    
    # Summary
    print(f"\n{'='*70}")
    print(f"AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"Hausa harmful prompts with baseline + critic steering: {len(flips) + len(non_flips)}")
    print(f"VETO→ALLOW flips: {len(flips)}")
    print(f"Non-flips: {len(non_flips)}")
    
    # Category breakdown
    if flips:
        category_counts = Counter(f['category'] for f in flips)
        print(f"\nFlip categories:")
        for cat, count in category_counts.most_common():
            print(f"  {cat or 'N/A'}: {count}")
    
    # Save verified flips
    print(f"\n{'='*70}")
    print(f"Saving {len(flips)} verified flips to {OUTPUT_FILE}...")
    
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for flip in flips:
            record = {
                'id': flip['id'],
                'category': flip.get('category', ''),
                'baseline_vetoed': flip['baseline_vetoed'],
                'critic_vetoed': flip['critic_vetoed'],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Done.")
    
    # Print first 10 flips
    print(f"\n{'='*70}")
    print(f"FIRST 10 VERIFIED FLIPS:")
    print(f"{'='*70}")
    print(f"{'ID':<40} {'Category':<25} {'Baseline':<10} {'Critic':<10}")
    print(f"{'-'*70}")
    for flip in flips[:10]:
        print(f"{flip['id']:<40} {flip.get('category', 'N/A'):<25} {'VETO' if flip['baseline_vetoed'] else 'ALLOW':<10} {'VETO' if flip['critic_vetoed'] else 'ALLOW':<10}")


if __name__ == '__main__':
    main()
