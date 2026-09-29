"""
Compute a Llama-3.3-70B refusal steering vector at layer 50.

Method:
- Run Llama on a set of harmful and benign prompts (same as Addition 6 manifest).
- Capture hidden states at layer 50 during prefill.
- Compute v_refusal = mean(h_harmful) - mean(h_benign) at the final token.
- Normalize to unit norm.

Output:
- steering_vector.pt (shape: (hidden_dim,))
- steering_vector_metadata.json
"""

import argparse
import hashlib
import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Configuration ===

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MANIFEST_PATH = Path("addition_6_prompt_filter_baseline/addition6_llama_full_outputs/../addition6_llama_full_outputs/no_intervention_outputs.jsonl")
OUTPUT_DIR = Path("addition_1_causal_patching/llama3_70b_steering_vector")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEERING_LAYER = 50
CAPTURE_TOKEN_POS = -1  # final token

# Use a subset of prompts for vector extraction (e.g., 50 harmful + 50 benign)
NUM_HARMFUL = 50
NUM_BENIGN = 50

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# === Load manifest ===

def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows

# === Model loading ===

def load_model_and_tokenizer():
    LOGGER.info("Loading tokenizer: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        padding_side="left",
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    LOGGER.info("Loading model: %s", MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer

# === Hidden state capture ===

@contextmanager
def capture_hidden_states_hook(model, layer_idx: int, captured_dict: Dict[int, List[torch.Tensor]]):
    """Capture hidden states at a specific layer for each forward pass."""
    def hook_fn(module, inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        # Capture the final token's hidden state
        final_token_hidden = hidden[:, -1, :].detach()
        captured_dict.setdefault(layer_idx, []).append(final_token_hidden)
        return outputs

    layer_module = model.model.layers[layer_idx]
    handle = layer_module.register_forward_hook(hook_fn)

    try:
        yield
    finally:
        handle.remove()

def generate_and_capture(
    model,
    tokenizer,
    system_prompt: str,
    user_message: str,
    *,
    max_new_tokens: int = 1,  # minimal generation; we only care about prefill
    capture_layer: int,
) -> Tuple[str, Dict[int, torch.Tensor]]:
    """Generate and capture hidden states at the specified layer."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    captured: Dict[int, List[torch.Tensor]] = {}

    with capture_hidden_states_hook(model, capture_layer, captured):
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    # Aggregate captured hidden states (should be one per forward pass)
    aggregated = {
        layer: torch.cat(captures, dim=0).mean(dim=0, keepdim=True)
        for layer, captures in captured.items()
    }

    completion_ids = generated[0, encoded["input_ids"].shape[1]:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

    return completion, aggregated

# === Main vector computation ===

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def compute_steering_vector():
    model, tokenizer = load_model_and_tokenizer()

    # Load manifest and sample prompts
    manifest = load_manifest(MANIFEST_PATH)
    harmful_rows = [r for r in manifest if r["harmfulness"] == "harmful"][:NUM_HARMFUL]
    benign_rows = [r for r in manifest if r["harmfulness"] == "benign"][:NUM_BENIGN]

    LOGGER.info("Using %d harmful and %d benign prompts for vector extraction", len(harmful_rows), len(benign_rows))

    # Simple system prompt for vector extraction
    system_prompt = "You are a helpful AI assistant."

    harmful_hiddens = []
    benign_hiddens = []

    # Extract harmful hidden states
    for i, row in enumerate(harmful_rows):
        LOGGER.info("[Harmful %d/%d] %s", i+1, len(harmful_rows), row["base_id"])
        _, captured = generate_and_capture(
            model, tokenizer, system_prompt, row["prompt_text"],
            max_new_tokens=1,
            capture_layer=STEERING_LAYER,
        )
        harmful_hiddens.append(captured[STEERING_LAYER])

    # Extract benign hidden states
    for i, row in enumerate(benign_rows):
        LOGGER.info("[Benign %d/%d] %s", i+1, len(benign_rows), row["base_id"])
        _, captured = generate_and_capture(
            model, tokenizer, system_prompt, row["prompt_text"],
            max_new_tokens=1,
            capture_layer=STEERING_LAYER,
        )
        benign_hiddens.append(captured[STEERING_LAYER])

    # Compute mean hidden states
    harmful_mean = torch.cat(harmful_hiddens, dim=0).mean(dim=0)  # (1, hidden)
    benign_mean = torch.cat(benign_hiddens, dim=0).mean(dim=0)    # (1, hidden)

    # Compute refusal vector
    refusal_vector = harmful_mean - benign_mean  # (1, hidden)
    refusal_vector = refusal_vector.squeeze(0)   # (hidden,)

    # Normalize to unit norm
    norm = float(refusal_vector.float().norm())
    refusal_vector = refusal_vector / norm

    LOGGER.info("Refusal vector shape: %s", tuple(refusal_vector.shape))
    LOGGER.info("Refusal vector norm (before normalization): %.6f", norm)
    LOGGER.info("Refusal vector norm (after normalization): %.6f", float(refusal_vector.float().norm()))

    # Save vector
    vector_path = OUTPUT_DIR / "steering_vector.pt"
    torch.save(refusal_vector, vector_path)
    LOGGER.info("Saved steering vector to: %s", vector_path)

    # Save metadata
    metadata = {
        "model_id": MODEL_ID,
        "steering_layer": STEERING_LAYER,
        "num_harmful": len(harmful_rows),
        "num_benign": len(benign_rows),
        "vector_shape": list(refusal_vector.shape),
        "vector_dtype": str(refusal_vector.dtype),
        "vector_norm_before": norm,
        "vector_norm_after": float(refusal_vector.float().norm()),
        "vector_sha256": sha256_file(vector_path),
        "manifest_path": str(MANIFEST_PATH),
    }

    metadata_path = OUTPUT_DIR / "steering_vector_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    LOGGER.info("Saved metadata to: %s", metadata_path)

    return vector_path, metadata

if __name__ == "__main__":
    vector_path, metadata = compute_steering_vector()
    print(f"\nSteering vector computed: {vector_path}")
    print(f"Metadata: {metadata}")
