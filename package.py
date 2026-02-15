"""
Package LaaLM-v2 - Technical Files Only
Converts .pt to safetensors + generates required HF files
"""

import torch
import json
from pathlib import Path
from safetensors.torch import save_file
import shutil
from dataclasses import dataclass

# Import ModelConfig from training script
@dataclass
class ModelConfig:
    vocab_size: int = 8000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1
    
    def __post_init__(self):
        self.n_params = self.calculate_params()
    
    def calculate_params(self):
        emb = self.vocab_size * self.d_model
        pos_emb = self.max_seq_len * self.d_model
        attn_qkv = 3 * self.d_model * self.d_model
        attn_out = self.d_model * self.d_model
        ffn = 2 * self.d_model * self.d_ff
        layer_norm = 2 * 2 * self.d_model
        per_layer = attn_qkv + attn_out + ffn + layer_norm
        final_ln = 2 * self.d_model
        lm_head = self.vocab_size * self.d_model
        total = emb + pos_emb + (per_layer * self.n_layers) + final_ln + lm_head
        return total

# Config
CHECKPOINT_PATH = "checkpoints/laalm_v2_final.pt"
TOKENIZER_PATH = "laalm_v2_tokenizer.json"
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
    
    # 1. Convert to safetensors
    print("Converting to safetensors...")
    save_file(model_state, str(output_dir / "model.safetensors"))
    
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
        "model_max_length": 512,
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
        "max_length": 512,
        "do_sample": False,
    }
    with open(output_dir / "generation_config.json", 'w') as f:
        json.dump(gen_config, f, indent=2)
    
    print(f"\n✓ Done! Model packaged in {OUTPUT_DIR}/")
    print("\nFiles:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size / (1024*1024)
        print(f"  - {f.name} ({size:.1f} MB)")

if __name__ == "__main__":
    package()
