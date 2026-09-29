#!/usr/bin/env python3
"""
Reconstruct full critic input for the 11 verified flip cases.
Extracts user prompt + planner response + tool-caller response from original runs.
"""

import json
from pathlib import Path

# Load flip IDs
flip_ids = set()
with open('artifacts/prior_llama33_phase2/transcripts/verified_flips_hausa.jsonl', 'r') as f:
    for line in f:
        row = json.loads(line)
        flip_ids.add(row['id'])

print(f"Reconstructing {len(flip_ids)} flip cases...\n")

# Load all steering results
all_results = {}
with open('artifacts/prior_llama33_phase2/results/turbo_steering_results_phase2.jsonl', 'r') as f:
    for line in f:
        row = json.loads(line)
        row_id = row.get('id')
        if row_id:
            if row_id not in all_results:
                all_results[row_id] = {}
            config = row.get('config')
            if config:
                all_results[row_id][config] = row

# Load prompt texts
with open('artifacts/prior_llama33_phase2/transcripts/flip_prompts_hausa.json', 'r') as f:
    prompts = json.load(f)

# Reconstruct for each flip
reconstructed = {}
for flip_id in sorted(flip_ids):
    print(f"Processing {flip_id}...")
    
    # Get prompt
    prompt_data = prompts.get(flip_id, {})
    user_prompt = prompt_data.get('text', '')
    
    # Find the behavior ID (strip the _planner/_critic/_tool-caller suffix)
    behavior_match = flip_id.rsplit('_', 1)[0]  # e.g., ha_harmful_044
    
    # Find all configs for this behavior
    matching_ids = [k for k in all_results.keys() if k.startswith(behavior_match)]
    
    # Get planner response
    planner_response = None
    for mid in matching_ids:
        if 'planner' in mid and 'planner_only' in all_results.get(mid, {}):
            planner_response = all_results[mid]['planner_only'].get('planner_response')
            if planner_response:
                break
    
    # Get tool-caller response
    tool_response = None
    for mid in matching_ids:
        if 'tool-caller' in mid or 'tool_caller' in mid:
            if 'planner_only' in all_results.get(mid, {}):
                tool_response = all_results[mid]['planner_only'].get('tool_caller_response')
                if tool_response:
                    break
    
    # Get critic response (baseline, strength=0)
    critic_baseline = None
    for mid in matching_ids:
        if 'critic' in mid:
            if 'planner_only' in all_results.get(mid, {}):
                row = all_results[mid]['planner_only']
                if row.get('strength') == 0.0:
                    critic_baseline = row.get('critic_response')
                    if critic_baseline:
                        break
    
    reconstructed[flip_id] = {
        'user_prompt': user_prompt,
        'planner_response': planner_response,
        'tool_response': tool_response,
        'critic_baseline': critic_baseline,
        'behavior_match': behavior_match,
    }
    
    print(f"  User: {user_prompt[:60]}...")
    print(f"  Planner: {'Yes' if planner_response else 'No'}")
    print(f"  Tool: {'Yes' if tool_response else 'No'}")
    print(f"  Critic baseline: {'Yes' if critic_baseline else 'No'}")
    print()

# Save
output_path = Path('artifacts/prior_llama33_phase2/transcripts/reconstructed_flip_cases.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(reconstructed, f, indent=2, ensure_ascii=False)

print(f"Saved to {output_path}")
print(f"Total: {len(reconstructed)} cases")

# Show one example
print(f"\n{'='*70}")
print(f"EXAMPLE: {list(reconstructed.keys())[0]}")
print(f"{'='*70}")
example = list(reconstructed.values())[0]
print(f"User prompt: {example['user_prompt'][:200]}")
print(f"\nPlanner response: {example['planner_response'][:300] if example['planner_response'] else 'None'}")
print(f"\nTool response: {example['tool_response'][:300] if example['tool_response'] else 'None'}")
print(f"\nCritic baseline: {example['critic_baseline'][:300] if example['critic_baseline'] else 'None'}")
