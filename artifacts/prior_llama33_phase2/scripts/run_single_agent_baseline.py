# run_single_agent_baseline.py
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DATASET_JSONL = "role_level_eval_dataset.jsonl"
OUTPUT_LOGS = "single_agent_baseline_log.jsonl"

# Load the model with memory saving turned on so it fits your single H100
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")

def query_llm(user_message):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message}
    ]
    
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=None, do_sample=False)
        
    input_length = inputs['input_ids'].shape[-1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response

def run_baseline(limit=None):
    if not Path(DATASET_JSONL).exists():
        print(f"Error: {DATASET_JSONL} not found.")
        return

    print(f"--- Starting Single-Agent Baseline Run ---")
    
    with open(DATASET_JSONL, "r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f]

    unique_prompt_ids = sorted(list(set("_".join(r["id"].split("_")[:-1]) for r in all_rows)))
    if limit:
        unique_prompt_ids = unique_prompt_ids[:limit]
    
    with open(OUTPUT_LOGS, "w", encoding="utf-8") as out_f:
        for base_id in unique_prompt_ids:
            matching_rows = [r for r in all_rows if r["id"].startswith(base_id)]
            if not matching_rows:
                continue
                
            sample_row = matching_rows[0]
            raw_prompt = sample_row["prompt_text"]
            lang_name = sample_row["language"]
            harm_status = sample_row["harmfulness"]
            
            print(f"\nEvaluating: {base_id} | Language: {lang_name} | Target: {harm_status}")
            
            # The single agent just gets the prompt directly
            model_response = query_llm(raw_prompt)
            
            log_entry = {
                "base_id": base_id,
                "language": lang_name,
                "harmfulness": harm_status,
                "raw_prompt": raw_prompt,
                "model_response": model_response,
            }
            
            out_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            print(f"Completed baseline response for {base_id}")

    print(f"\n--- Baseline run finished. Results saved to {OUTPUT_LOGS} ---")

if __name__ == "__main__":
    # Passing None runs the entire dataset to match your multi-agent full run
    run_baseline(limit=None)