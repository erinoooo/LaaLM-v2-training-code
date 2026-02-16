"""
Package LaaLM-v2 - Technical Files Only
Converts .pt to safetensors + generates required HF files
"""

import torch
import json
from pathlib import Path
from safetensors.torch import save_file
import shutil

from model import ModelConfig, LaaLMModel

# Config - paths match train.py output
CHECKPOINT_PATH = "checkpoints_v2/laalm_v2_final.pt"
TOKENIZER_PATH = "laalm_v2_tokenizer_v3.json"
OUTPUT_DIR = "laalm-v2"

def package():
    # Create output dir
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    print(f"Packaging model to {OUTPUT_DIR}/")

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model_state = checkpoint['model_state_dict']
    config = checkpoint['config']

    print(f"Model: {config.n_params/1e6:.1f}M parameters")

    # Clean up state dict (remove _orig_mod prefix from torch.compile)
    print("Cleaning state dict...")
    cleaned_state = {}
    for key, value in model_state.items():
        clean_key = key.replace('_orig_mod.', '')
        cleaned_state[clean_key] = value

    # Handle weight tying: lm_head.weight and token_emb.weight share the same
    # tensor. safetensors save_file doesn't allow duplicate tensors, so remove
    # the tied copy and note it in config.
    if 'lm_head.weight' in cleaned_state and 'token_emb.weight' in cleaned_state:
        if cleaned_state['lm_head.weight'].data_ptr() == cleaned_state['token_emb.weight'].data_ptr():
            del cleaned_state['lm_head.weight']

    # 1. Save as safetensors
    print("Converting to safetensors...")
    save_file(cleaned_state, str(output_dir / "model.safetensors"))

    # 2. Create config.json
    print("Creating config.json...")
    hf_config = {
        "architectures": ["LaaLMModel"],
        "model_type": "laalm",
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "intermediate_size": config.d_ff,
        "max_position_embeddings": config.max_seq_len,
        "hidden_dropout_prob": config.dropout,
        "attention_dropout_prob": config.dropout,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": True,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(hf_config, f, indent=2)

    # 3. Copy tokenizer
    print("Copying tokenizer...")
    shutil.copy(TOKENIZER_PATH, output_dir / "tokenizer.json")

    # 4. Tokenizer config
    print("Creating tokenizer_config.json...")
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": config.max_seq_len,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    with open(output_dir / "tokenizer_config.json", 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    # 5. Special tokens map
    print("Creating special_tokens_map.json...")
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    with open(output_dir / "special_tokens_map.json", 'w') as f:
        json.dump(special_tokens, f, indent=2)

    # 6. Generation config
    print("Creating generation_config.json...")
    gen_config = {
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
        "max_length": config.max_seq_len,
        "do_sample": False,
    }
    with open(output_dir / "generation_config.json", 'w') as f:
        json.dump(gen_config, f, indent=2)

    print(f"\nDone! Model packaged in {OUTPUT_DIR}/")
    print("\nFiles:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size / (1024*1024)
        print(f"  - {f.name} ({size:.1f} MB)")

if __name__ == "__main__":
    package()
