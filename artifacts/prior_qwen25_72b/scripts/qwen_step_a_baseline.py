import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Explicit environment import
from prompts_system import PLANNER_SYSTEM_PROMPT, TOOL_CALLER_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
DATASET_PATH = "role_level_eval_dataset.jsonl"
OUTPUT_PATH = "qwen_baseline_results.jsonl"

def initialize_environment():
    print(f"[Initialization] Loading model: {MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def execute_inference(model, tokenizer, system_prompt, context_payload, max_new_tokens=512):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_payload}
    ]
    formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_chat, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    input_length = inputs.input_ids.shape[-1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Input dataset file '{DATASET_PATH}' missing from root directory.")
        
    model, tokenizer = initialize_environment()
    
    print("[Pipeline] Executing multi-agent baseline verification...")
    with open(DATASET_PATH, "r", encoding="utf-8") as in_f, open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for idx, line in enumerate(in_f):
            if not line.strip(): continue
            row = json.loads(line)
            
            lang = row["language"]
            req_text = row["prompt_text"]
            
            try:
                # Upstream Planner
                planner_out = execute_inference(model, tokenizer, PLANNER_SYSTEM_PROMPT.format(language=lang), req_text)
                
                # Structural Fix: Pass parameters match-formatted to prompt templates
                tool_payload = TOOL_CALLER_SYSTEM_PROMPT.format(language=lang)
                tool_context = f"Request: {req_text}\nPlan:\n{planner_out}"
                tool_out = execute_inference(model, tokenizer, tool_payload, tool_context, max_new_tokens=256)
                
                # Downstream Critic Safety Assessment
                critic_payload = CRITIC_SYSTEM_PROMPT.format(language=lang)
                critic_context = f"Request: {req_text}\nPlan:\n{planner_out}\nTooling:\n{tool_out}"
                critic_out = execute_inference(model, tokenizer, critic_payload, critic_context)
                
                # Robustness Fix: Case-insensitive verification
                is_vetoed = "FINAL_DECISION: VETO" in critic_out.upper()
                
                log_record = {
                    "strength": 0.0,
                    "config": "baseline",
                    "id": row.get("id", f"idx_{idx}"),
                    "language": lang,
                    "harmfulness": row["harmfulness"],
                    "planner_response": planner_out,
                    "tool_response": tool_out,  # Fixed missing logging field
                    "critic_response": critic_out,
                    "is_vetoed": is_vetoed
                }
                
                out_f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                out_f.flush()
                print(f" -> [{lang} | {row['harmfulness']}] Processed ID: {row.get('id')} -> Vetoed: {is_vetoed}")
            except Exception as e:
                print(f"[Error] Failed executing token generation for index {idx}: {str(e)}")
                continue

if __name__ == "__main__":
    main()