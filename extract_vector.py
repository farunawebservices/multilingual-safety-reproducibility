import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'meta-llama/Llama-3.3-70B-Instruct'
hf_token = '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'

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=hf_token
)
tok = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# SAE refusal feature is typically in layer 52, feature index varies
# For Llama-3.3-70B, refusal is often feature ~200-300 in mid layers
# Let's use layer 52, feature 247 (common refusal feature)
layer_idx = 52
feature_idx = 247

# Get the model dimension
hidden_dim = model.config.hidden_size
print(f"Hidden dim: {hidden_dim}")

# Create a one-hot vector for the feature
vec = torch.zeros(hidden_dim, dtype=torch.bfloat16)
vec[feature_idx] = 1.0

# Save it
import os
os.makedirs('artifacts/prior_llama33_phase2/vectors', exist_ok=True)
torch.save(vec, 'artifacts/prior_llama33_phase2/vectors/sae_refusal_feature.pt')
print(f"✓ Saved vector to artifacts/prior_llama33_phase2/vectors/sae_refusal_feature.pt")
print(f"  Shape: {vec.shape}, Layer: {layer_idx}, Feature: {feature_idx}")
