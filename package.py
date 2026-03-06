"""
Package LaaLM-v2 (LSTM edition)
Converts .pt to safetensors + generates HF-compatible config files.
"""

import os
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from model import LaaLMv2Config, LaaLMModel

BEST_PATH       = "checkpoints_v2/laalm_v2_best.pt"
FINAL_PATH      = "checkpoints_v2/laalm_v2_final.pt"
CHECKPOINT_PATH = BEST_PATH if os.path.exists(BEST_PATH) else FINAL_PATH
TOKENIZER_PATH  = "laalm_v2_tokenizer_v3.json"
OUTPUT_DIR      = "laalm-v2"


def package():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"Packaging model to {OUTPUT_DIR}/")

    checkpoint  = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]
    config      = checkpoint["config"]
    print(f"Model: {config.n_params / 1e6:.1f}M parameters")

    # Strip torch.compile prefix
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}

    # Drop tied lm_head.weight duplicate (safetensors forbids shared tensors)
    if ("lm_head.weight" in cleaned and "token_emb.weight" in cleaned and
            cleaned["lm_head.weight"].data_ptr() == cleaned["token_emb.weight"].data_ptr()):
        del cleaned["lm_head.weight"]

    # 1. safetensors
    print("Converting to safetensors...")
    save_file(cleaned, str(output_dir / "model.safetensors"))

    # 2. config.json — updated for LSTM (no n_heads / d_ff / swiglu fields)
    print("Creating config.json...")
    hf_config = {
        "architectures":        ["LaaLMModel"],
        "model_type":           "laalm-lstm",
        "vocab_size":           config.vocab_size,
        "hidden_size":          config.d_model,
        "num_hidden_layers":    config.n_layers,
        "max_position_embeddings": config.max_seq_len,
        "hidden_dropout_prob":  config.dropout,
        "rms_norm_eps":         1e-6,
        "torch_dtype":          "bfloat16",
        "tie_word_embeddings":  True,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # 3–6. Tokenizer + config files (unchanged)
    print("Copying tokenizer...")
    shutil.copy(TOKENIZER_PATH, output_dir / "tokenizer.json")

    for fname, data in [
        ("tokenizer_config.json", {
            "tokenizer_class":  "PreTrainedTokenizerFast",
            "model_max_length": config.max_seq_len,
            "bos_token": "<s>", "eos_token": "</s>",
            "unk_token": "<unk>", "pad_token": "<pad>",
        }),
        ("special_tokens_map.json", {
            "bos_token": "<s>", "eos_token": "</s>",
            "unk_token": "<unk>", "pad_token": "<pad>",
        }),
        ("generation_config.json", {
            "bos_token_id": 2, "eos_token_id": 3,
            "pad_token_id": 0, "max_length":   config.max_seq_len,
            "do_sample": False,
        }),
    ]:
        print(f"Creating {fname}...")
        with open(output_dir / fname, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\nDone! Model packaged in {OUTPUT_DIR}/")
    for p in sorted(output_dir.iterdir()):
        print(f"  - {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    package()
```

---

### `requirements.txt`
```
torch>=2.1.0
tokenizers>=0.15.0
safetensors>=0.4.0
wandb>=0.16.0
tqdm>=4.60.0
