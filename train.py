"""
LaaLM-v2 Training Script - UPDATED FOR V3 FORMAT
Uses new unambiguous delimiter format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import json
import math
import os
from pathlib import Path
from tqdm import tqdm
import wandb
from dataclasses import dataclass

from model import ModelConfig, LaaLMModel

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class TrainConfig:
    # Data - UPDATED PATHS
    data_path: str = "laalm_v2_training_data_v3.jsonl"
    tokenizer_path: str = "laalm_v2_tokenizer_v3.json"

    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 30000
    warmup_steps: int = 2000
    eval_interval: int = 500
    save_interval: int = 2000

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # System
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    compile: bool = True
    num_workers: int = 4

    # Checkpointing
    output_dir: str = "checkpoints_v2"
    wandb_project: str = "laalm-v2"
    wandb_run_name: str = "laalm-v2-final"

# ============================================================================
# DATASET - UPDATED FOR V3 FORMAT
# ============================================================================

class LaaLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = tokenizer.token_to_id("<pad>")

        print(f"Loading data from {data_path}...")
        self.conversations = []
        with open(data_path) as f:
            for line in f:
                conv = json.loads(line)
                self.conversations.append(conv['text'])

        print(f"Loaded {len(self.conversations)} conversations")

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        text = self.conversations[idx]

        # Tokenize
        encoded = self.tokenizer.encode(text)
        tokens = encoded.ids[:self.max_len]

        # Pad if needed
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        if len(tokens) < self.max_len:
            tokens = tokens + [pad_id] * (self.max_len - len(tokens))

        # Create input and target
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y

# ============================================================================
# TRAINING
# ============================================================================

def get_lr(step, config):
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step > config.max_steps:
        return config.learning_rate * 0.1
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * 0.1 + coeff * (config.learning_rate - config.learning_rate * 0.1)

def train():
    model_config = ModelConfig()
    train_config = TrainConfig()

    print(f"Model has {model_config.n_params/1e6:.1f}M parameters")

    # Build wandb config dict without non-serializable types
    wandb_config = {**model_config.__dict__, **train_config.__dict__}
    wandb_config['dtype'] = str(train_config.dtype)
    wandb.init(project=train_config.wandb_project, name=train_config.wandb_run_name, config=wandb_config)

    tokenizer = Tokenizer.from_file(train_config.tokenizer_path)
    dataset = LaaLMDataset(train_config.data_path, tokenizer, max_len=model_config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=train_config.num_workers, pin_memory=True)

    model = LaaLMModel(model_config)
    model = model.to(train_config.device).to(train_config.dtype)

    if train_config.compile:
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, betas=(train_config.beta1, train_config.beta2), weight_decay=train_config.weight_decay)

    model.train()
    step = 0
    running_loss = 0.0

    Path(train_config.output_dir).mkdir(exist_ok=True)
    dataloader_iter = iter(dataloader)

    print("Starting training...")
    pbar = tqdm(total=train_config.max_steps)

    while step < train_config.max_steps:
        try:
            x, y = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            x, y = next(dataloader_iter)

        x = x.to(train_config.device)
        y = y.to(train_config.device)

        with torch.autocast(device_type='cuda', dtype=train_config.dtype):
            logits, loss = model(x, y)
            loss = loss / train_config.gradient_accumulation_steps

        loss.backward()
        running_loss += loss.item()

        if (step + 1) % train_config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            lr = get_lr(step, train_config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()

            # running_loss already accumulated (loss / grad_accum) over grad_accum
            # steps, so it equals the true average loss — no further division needed
            avg_loss = running_loss
            wandb.log({"loss": avg_loss, "lr": lr, "step": step})
            pbar.set_description(f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            running_loss = 0.0

        if step > 0 and step % train_config.save_interval == 0:
            checkpoint_path = f"{train_config.output_dir}/checkpoint_{step}.pt"
            torch.save({'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'config': model_config}, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")

        step += 1
        pbar.update(1)

    pbar.close()
    final_path = f"{train_config.output_dir}/laalm_v2_final.pt"
    torch.save({'model_state_dict': model.state_dict(), 'config': model_config}, final_path)
    print(f"Training complete! Saved final model to {final_path}")
    wandb.finish()

if __name__ == "__main__":
    train()
