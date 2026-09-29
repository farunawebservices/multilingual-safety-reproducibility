#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'meta-llama/Llama-3.3-70B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')

prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

# Test 1: Simple forward pass to see output shapes
print("Testing forward pass output shapes...")
with torch.no_grad():
    outputs = model(**inputs, return_dict=True)
    print(f"Output type: {type(outputs)}")
    print(f"Output keys (if dict): {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")
    if hasattr(outputs, 'last_hidden_state'):
        print(f"last_hidden_state shape: {outputs.last_hidden_state.shape}")

# Test 2: Check layer output during generation
print("\nTesting layer hook during generation...")
steering_applied = False
test_vec = torch.randn(model.config.hidden_size).to(model.device)

def test_hook(module, input, output):
    global steering_applied
    if not steering_applied:
        print(f"Hook triggered!")
        print(f"Output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"Tuple length: {len(output)}")
            for i, o in enumerate(output):
                if hasattr(o, 'shape'):
                    print(f"  [{i}] shape: {o.shape}")
        elif hasattr(output, 'shape'):
            print(f"Output shape: {output.shape}")
        steering_applied = True
    return output

hook = model.model.layers[52].register_forward_hook(test_hook)
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
hook.remove()

print("\nDone!")
