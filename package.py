"""
Package LaaLM-v2 (LSTM edition)

Bug fixes:
  - data_ptr() comparison: after torch.load(), weight-tied tensors become
    separate copies in memory — data_ptr() never matches, so lm_head.weight
    was never removed and save_file() crashed on duplicate tensor data.
    Fix: if both keys exist, unconditionally delete lm_head.weight.
  - config.n_params: old checkpoints may not have this attribute
    (added in a later version of model.py). Use getattr() with a fallback.
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
    print(f"Packaging model → {OUTPUT_DIR}/")

    checkpoint  = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]
    config      = checkpoint["config"]

    # getattr fallback: old checkpoints saved before n_params was added
    n_params = getattr(config, "n_params", None)
    if n_params is not None:
        print(f"Parameters: {n_params / 1e6:.2f}M")

    # Strip torch.compile _orig_mod prefix
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}

    # Bug fix: after round-tripping through torch.save/load, weight-tied
    # tensors become independent copies — data_ptr() will never match.
    # The canonical weight is token_emb.weight; always drop lm_head.weight.
    if "lm_head.weight" in cleaned and "token_emb.weight" in cleaned:
        del cleaned["lm_head.weight"]
        print("Removed tied lm_head.weight (token_emb.weight is canonical)")

    # 1. safetensors
    print("Saving model.safetensors...")
    save_file(cleaned, str(output_dir / "model.safetensors"))

    # 2. config.json — LSTM fields only (no n_heads / d_ff / swiglu)
    hf_config = {
        "architectures":           ["LaaLMModel"],
        "model_type":              "laalm-lstm",
        "vocab_size":              config.vocab_size,
        "hidden_size":             config.d_model,
        "num_hidden_layers":       config.n_layers,
        "max_position_embeddings": config.max_seq_len,
        "hidden_dropout_prob":     config.dropout,
        "torch_dtype":             "bfloat16",
        "tie_word_embeddings":     True,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # 3–6. Tokenizer + companion files
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
            "do_sample":    False,
        }),
    ]:
        with open(output_dir / fname, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\nDone. Files in {OUTPUT_DIR}/:")
    for p in sorted(output_dir.iterdir()):
        print(f"  {p.name:40s} {p.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    package()
