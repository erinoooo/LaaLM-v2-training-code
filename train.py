"""
LaaLM-v2 Training Script
Train a 200-300M parameter decoder-only transformer from scratch on MI300X
Optimized for ROCm/AMD GPUs
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

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 8000
    d_model: int = 768          # Hidden dimension
    n_layers: int = 12          # Number of transformer layers
    n_heads: int = 12           # Attention heads
    d_ff: int = 3072            # FFN dimension (4x d_model)
    max_seq_len: int = 512      # Maximum sequence length
    dropout: float = 0.1
    
    def __post_init__(self):
        self.n_params = self.calculate_params()
    
    def calculate_params(self):
        """Estimate parameter count"""
        # Embedding
        emb = self.vocab_size * self.d_model
        pos_emb = self.max_seq_len * self.d_model
        
        # Per layer
        attn_qkv = 3 * self.d_model * self.d_model
        attn_out = self.d_model * self.d_model
        ffn = 2 * self.d_model * self.d_ff
        layer_norm = 2 * 2 * self.d_model
        
        per_layer = attn_qkv + attn_out + ffn + layer_norm
        
        # Final
        final_ln = 2 * self.d_model
        lm_head = self.vocab_size * self.d_model
        
        total = emb + pos_emb + (per_layer * self.n_layers) + final_ln + lm_head
        return total

@dataclass
class TrainConfig:
    # Data
    data_path: str = "laalm_v2_training_data.jsonl"
    tokenizer_path: str = "laalm_v2_tokenizer.json"
    
    # Training
    batch_size: int = 128
    gradient_accumulation_steps: int = 1  # Effective batch = 64
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
    output_dir: str = "checkpoints"
    wandb_project: str = "laalm-v2"
    wandb_run_name: str = "laalm-v2-training"

# ============================================================================
# MODEL
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm (more efficient than LayerNorm)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute for max_seq_len
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x, seq_len):
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device),
        )

def apply_rotary_emb(q, k, cos, sin):
    """Apply RoPE to queries and keys"""
    # Rotate half
    q_rot = torch.cat((-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]), dim=-1)
    k_rot = torch.cat((-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]), dim=-1)
    
    # Apply rotation
    q_out = q * cos + q_rot * sin
    k_out = k * cos + k_rot * sin
    
    return q_out, k_out

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
    
    def forward(self, x):
        B, T, C = x.shape
        
        # QKV
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(x, T)
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Use PyTorch SDPA (optimized for MI300X)
        # CRITICAL: Use is_causal=True for autoregressive
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # SwiGLU activation
        return self.dropout(self.w2(F.silu(self.w1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class LaaLMModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token embeddings
        x = self.token_emb(idx)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

# ============================================================================
# DATASET
# ============================================================================

class LaaLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load all conversations
        print(f"Loading data from {data_path}...")
        self.conversations = []
        with open(data_path) as f:
            for line in f:
                conv = json.loads(line)
                self.conversations.append(conv)
        
        print(f"Loaded {len(self.conversations)} conversations")
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # Format conversation
        text = ""
        for msg in conv['messages']:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                text += f"{content}\n"
            elif role == 'user':
                text += f"$ {content}\n"
            elif role == 'assistant':
                text += f"{content}\n"
        
        # Tokenize
        encoded = self.tokenizer.encode(text)
        tokens = encoded.ids[:self.max_len]
        
        # Pad if needed
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        
        # Create input and target
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y

# ============================================================================
# TRAINING
# ============================================================================

def get_lr(step, config):
    """Cosine learning rate schedule with warmup"""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    
    if step > config.max_steps:
        return config.learning_rate * 0.1
    
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * 0.1 + coeff * (config.learning_rate - config.learning_rate * 0.1)

def train():
    # Configs
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    print(f"Model has {model_config.n_params/1e6:.1f}M parameters")
    
    # Initialize wandb
    wandb.init(
        project=train_config.wandb_project,
        name=train_config.wandb_run_name,
        config={
            **model_config.__dict__,
            **train_config.__dict__
        }
    )
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(train_config.tokenizer_path)
    
    # Create dataset
    dataset = LaaLMDataset(
        train_config.data_path,
        tokenizer,
        max_len=model_config.max_seq_len
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = LaaLMModel(model_config)
    model = model.to(train_config.device)
    model = model.to(train_config.dtype)
    
    # Compile model for better performance
    if train_config.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay
    )
    
    # Training loop
    model.train()
    step = 0
    running_loss = 0.0
    
    # Create output directory
    Path(train_config.output_dir).mkdir(exist_ok=True)
    
    dataloader_iter = iter(dataloader)
    
    print("Starting training...")
    pbar = tqdm(total=train_config.max_steps)
    
    while step < train_config.max_steps:
        # Get batch
        try:
            x, y = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            x, y = next(dataloader_iter)
        
        x = x.to(train_config.device)
        y = y.to(train_config.device)
        
        # Forward pass
        with torch.autocast(device_type='cuda', dtype=train_config.dtype):
            logits, loss = model(x, y)
            loss = loss / train_config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        running_loss += loss.item()
        
        # Update weights
        if (step + 1) % train_config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            
            # Learning rate schedule
            lr = get_lr(step, train_config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            avg_loss = running_loss / train_config.gradient_accumulation_steps
            wandb.log({
                "loss": avg_loss,
                "lr": lr,
                "step": step
            })
            
            pbar.set_description(f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            running_loss = 0.0
        
        # Save checkpoint
        if step > 0 and step % train_config.save_interval == 0:
            checkpoint_path = f"{train_config.output_dir}/checkpoint_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
        
        step += 1
        pbar.update(1)
    
    # Save final model
    final_path = f"{train_config.output_dir}/laalm_v2_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
    }, final_path)
    print(f"Training complete! Saved final model to {final_path}")
    
    wandb.finish()

if __name__ == "__main__":
    train()
