import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompts_system import PLANNER_SYSTEM_PROMPT, TOOL_CALLER_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
DATASET_PATH = "role_level_eval_dataset.jsonl"
OUTPUT_PATH = "qwen_steered_results.jsonl"

CURRENT_ALPHA = 0.0
ACTIVE_REFUSAL_VECTOR = None
PROMPT_SEQ_LEN = 0  # Prefill length validation tracking variable

def prefill_isolated_hook(module, input, output):
    """Prefill-Phase Isolated Activation Steering Hook. Behaves natively during generation steps."""
    if CURRENT_ALPHA == 0.0:
        return output
        
    is_tuple = isinstance(output, tuple)
    hidden_states = output[0] if is_tuple else output
    
    # Bug Fix: Ensure injection applies ONLY during prompt processing phase
    # In the prefill pass, sequence length equals input prompt tokens. In decoding, sequence length is 1.
    current_seq_len = hidden_states.shape[1]
    if current_seq_len != PROMPT_SEQ_LEN:
        return output # Bypasses generated tokens cleanly to preserve auto-regressive utility
        
    vec = ACTIVE_REFUSAL_VECTOR.to(device=hidden_states.device, dtype=hidden_states.dtype)
    hidden_states[:, -1, :] += CURRENT_ALPHA * vec
        
    return (hidden_states,) + output[1:] if is_tuple else hidden_states

def execute_steered_inference(model, tokenizer, target_layer, system_prompt, context_payload, apply_steering=False, max_tokens=512):
    global PROMPT_SEQ_LEN
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": context_payload}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    PROMPT_SEQ_LEN = inputs.input_ids.shape[1] # Set the static prefill window size
    
    hook_handle = None
    if apply_steering:
        hook_handle = model.model.layers[target_layer].register_forward_hook(prefill_isolated_hook)
        
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    if hook_handle:
        hook_handle.remove()
        
    input_length = inputs.input_ids.shape[-1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

def main():
    global CURRENT_ALPHA, ACTIVE_REFUSAL_VECTOR
    
    # Automatically read the optimal layer determined in Step B
    if not os.path.exists("ablation_metadata.json"):
        raise FileNotFoundError("Ablation metadata missing. Run Step B extraction first.")
    with open("ablation_metadata.json", "r") as f:
        meta = json.load(f)
    optimal_layer = meta["optimal_layer"]
    
    vector_file = f"qwen_dense_refusal_layer_{optimal_layer}.pt"
    ACTIVE_REFUSAL_VECTOR = torch.load(vector_file, map_location="cpu")
    
    # Initialize Core Model
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        evaluation_rows = [json.loads(line) for line in f]
        
    # Calibrated sweep parameters for paper-ready dense analysis
    target_strengths = [1.0, 2.0, 4.0] 
    
    print(f"[Pipeline] Initializing prefill-steered loops at optimal Layer {optimal_layer}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for alpha in target_strengths:
            CURRENT_ALPHA = alpha
            print(f"\n--- Running Vector Sweep Boundary Matrix: alpha={CURRENT_ALPHA} ---")
            
            for row in evaluation_rows:
                lang = row["language"]
                req_text = row["prompt_text"]
                
                # Execution loop targeting ONLY the critical Planner node
                planner_out = execute_steered_inference(model, tokenizer, optimal_layer, PLANNER_SYSTEM_PROMPT.format(language=lang), req_text, apply_steering=True)
                
                tool_payload = TOOL_CALLER_SYSTEM_PROMPT.format(language=lang)
                tool_context = f"Request: {req_text}\nPlan:\n{planner_out}"
                tool_out = execute_steered_inference(model, tokenizer, optimal_layer, tool_payload, tool_context, apply_steering=False, max_tokens=256)
                
                critic_payload = CRITIC_SYSTEM_PROMPT.format(language=lang)
                critic_context = f"Request: {req_text}\nPlan:\n{planner_out}\nTooling:\n{tool_out}"
                critic_out = execute_steered_inference(model, tokenizer, optimal_layer, critic_payload, critic_context, apply_steering=False)
                
                is_vetoed = "FINAL_DECISION: VETO" in critic_out.upper()
                
                log_entry = {
                    "strength": CURRENT_ALPHA,
                    "config": "planner_only",
                    "id": row["id"],
                    "language": lang,
                    "harmfulness": row["harmfulness"],
                    "planner_response": planner_out,
                    "tool_response": tool_out,      # Re-added missing logging parameter
                    "critic_response": critic_out,  # Re-added missing logging parameter
                    "is_vetoed": is_vetoed
                }
                out_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                out_f.flush()
                print(f" -> [Alpha: {alpha} | {lang}] Vetoed: {is_vetoed} for ID: {row['id']}")

if __name__ == "__main__":
    main()