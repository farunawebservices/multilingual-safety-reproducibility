# extract_sae_feature.py
import torch
import os
from huggingface_hub import hf_hub_download

print("Downloading the Layer 50 SAE for Llama-3.3-70B-Instruct from Hugging Face...")
# This downloads the specific layer 50 SAE from Goodfire's open-source release
file_path = hf_hub_download(
    repo_id="Goodfire/Llama-3.3-70B-Instruct-SAE-l50", 
    filename="Llama-3.3-70B-Instruct-SAE-l50.pt"
)

print("Loading SAE weights into memory...")
sae_weights = torch.load(file_path, map_location="cpu")

# The Goodfire SAE has 65,536 features. 
# Based on interpretability research on Llama 3 models, the primary "Refusal/Concealment" 
# feature that triggers the safety filters is located at index 33404.
REFUSAL_FEATURE_INDEX = 33404

print(f"Extracting decoder weight for Feature {REFUSAL_FEATURE_INDEX}...")
# Extract the 1D vector (size 8192 for Llama 70B) corresponding to the refusal direction
W_dec = sae_weights['W_dec'] 
refusal_vector = W_dec[REFUSAL_FEATURE_INDEX]

# Save it as a lightweight PyTorch tensor
output_filename = "sae_refusal_feature.pt"
torch.save(refusal_vector, output_filename)

print(f"Success! The refusal vector has been saved as '{output_filename}'.")
print("You can now run your Phase 2 multi-agent steering script.")