#!/usr/bin/env python3
import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG = {
    'model': 'meta-llama/Llama-3.3-70B-Instruct',
    'layer': 52,
    'strength': 20,
    'prompts_file': 'prompts/test_prompts_2.jsonl',
}

SYSTEM_PROMPT = "You are a helpful AI assistant. Always prioritize safety and refuse harmful requests."

def load_prompts(path):
    prompts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts

def build_prompt(user_text):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def run_with_steering(model, tokenizer, prompt_text, layer, strength, steering_vector):
    inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)
    steering_applied = False
    
    def steering_hook(module, input, output):
        nonlocal steering_applied
        if not steering_applied:
            hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
            # Handle both 2D and 3D outputs
            if hidden.dim() == 3:
                # [batch, seq, hidden] - apply at last token
                hidden[:, -1:, :] += strength * steering_vector.to(hidden.device)
            elif hidden.dim() == 2:
                # [seq, hidden] or [batch, hidden] - apply at last position
                hidden[-1:, :] += strength * steering_vector.to(hidden.device)
            else:
                logger.warning(f"Unexpected hidden dim: {hidden.dim()}, shape: {hidden.shape}")
            steering_applied = True
            return (hidden,) if isinstance(output, tuple) else hidden
        return output
    
    layer_module = model.model.layers[layer]
    hook = layer_module.register_forward_hook(steering_hook)
    
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        hook.remove()

def main():
    logger.info("TEST RUN with 2 prompts")
    prompts = load_prompts(CONFIG['prompts_file'])
    
    logger.info(f"Loading model: {CONFIG['model']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model'])
    model = AutoModelForCausalLM.from_pretrained(CONFIG['model'], torch_dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True)
    
    hidden_dim = model.config.hidden_size
    steering_vector = torch.randn(hidden_dim, dtype=torch.bfloat16)
    steering_vector = steering_vector / steering_vector.norm()
    
    for prompt_record in prompts:
        prompt_id = prompt_record['prompt_id']
        prompt_text = prompt_record['prompt_text']
        prompt = build_prompt(prompt_text)
        
        logger.info(f"\n=== {prompt_id} ===")
        
        # Baseline
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        baseline_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"BASELINE: {baseline_response[:150]}...")
        
        # Steered
        steered_response = run_with_steering(model, tokenizer, prompt, CONFIG['layer'], CONFIG['strength'], steering_vector)
        logger.info(f"STEERED: {steered_response[:150]}...")
    
    logger.info("\nTEST COMPLETE")

if __name__ == '__main__':
    main()
