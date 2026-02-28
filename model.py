"""
LaaLM Model Architecture
Migrated from TPU/bfloat16 to NVIDIA L40S with FP8 via NVIDIA Transformer Engine.

Changes from original:
  - RMSNorm              → te.RMSNorm (TE-optimized, FP8-aware)
  - nn.Linear (attn/FFN) → te.Linear  (participates in FP8 Tensor Core ops)
  - token_emb + lm_head  → kept as nn.Embedding / nn.Linear with weight tying
                           (first/last layers recommended to stay in BF16)
  - No changes to RoPE, SwiGLU logic, or model structure
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

import transformer_engine.pytorch as te  # NVIDIA Transformer Engine


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class LaaLMConfig:
    """Base configuration for LaaLM models."""
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    dropout: float = 0.1
    use_swiglu: bool = False
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        self.head_dim = self.d_model // self.n_heads
        self.n_params = self.calculate_params()

    def calculate_params(self) -> int:
        emb = self.vocab_size * self.d_model
        attn_qkv = 3 * self.d_model * self.d_model
        attn_out = self.d_model * self.d_model
        if self.use_swiglu:
            ffn = 3 * self.d_model * self.d_ff
        else:
            ffn = 2 * self.d_model * self.d_ff
        layer_norm = 2 * self.d_model
        per_layer = attn_qkv + attn_out + ffn + layer_norm
        final_ln = self.d_model
        return emb + (per_layer * self.n_layers) + final_ln


@dataclass
class LaaLMv2Config(LaaLMConfig):
    """LaaLM v2 configuration.

    All inner dimensions must be divisible by 16 for FP8 Tensor Core ops:
      d_model=128  → 128/16=8   ✓
      d_ff=352     → 352/16=22  ✓
      3*d_model=384 → 384/16=24 ✓
    """
    vocab_size: int = 8000
    d_model: int = 128
    n_layers: int = 12
    n_heads: int = 4
    d_ff: int = 352
    max_seq_len: int = 1024
    dropout: float = 0.05
    use_swiglu: bool = True
    use_gradient_checkpointing: bool = False


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# RMSNorm is now te.RMSNorm — used directly below instead of a custom class.
# te.RMSNorm(hidden_size, eps) is a drop-in replacement that integrates with
# the TE FP8 graph.

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE). Unchanged from original."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
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
    """Apply rotary embeddings to queries and keys. Unchanged from original."""
    q_rot = torch.cat((-q[..., q.shape[-1] // 2:], q[..., :q.shape[-1] // 2]), dim=-1)
    k_rot = torch.cat((-k[..., k.shape[-1] // 2:], k[..., :k.shape[-1] // 2]), dim=-1)
    return q * cos + q_rot * sin, k * cos + k_rot * sin


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE.

    FP8 change: qkv and out_proj are now te.Linear so they participate in
    FP8 Tensor Core matrix multiplies when wrapped in fp8_autocast in train.py.
    The attention computation (sdpa) stays in BF16 — TE does not FP8-ify that.
    """

    def __init__(self, config: LaaLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        # te.Linear replaces nn.Linear — activates FP8 inside fp8_autocast.
        # bias=False matches the original. Both in/out dims must be div by 16.
        self.qkv = te.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = te.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x, T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Feed-forward network with optional SwiGLU.

    FP8 change: all weight matrices are now te.Linear.
    """

    def __init__(self, config: LaaLMConfig):
        super().__init__()
        self.use_swiglu = config.use_swiglu

        self.w1 = te.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = te.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if self.use_swiglu:
            self.w3 = te.Linear(config.d_model, config.d_ff, bias=False)

    def forward(self, x):
        if self.use_swiglu:
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        else:
            return self.dropout(self.w2(F.silu(self.w1(x))))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture.

    FP8 change: RMSNorm is now te.RMSNorm.
    """

    def __init__(self, config: LaaLMConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        # te.RMSNorm replaces the custom RMSNorm — same interface, same math,
        # but integrates with TE's FP8 graph planning.
        self.norm1 = te.RMSNorm(config.d_model)
        self.norm2 = te.RMSNorm(config.d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LaaLMModel(nn.Module):
    """LaaLM Language Model.

    FP8 notes:
      - All inner te.Linear layers activate FP8 when forward() is called
        inside a te.fp8_autocast() context (done in train.py).
      - token_emb (nn.Embedding) and lm_head (nn.Linear) are intentionally
        kept in BF16. The first/last layers are recommended to stay at higher
        precision for numerical stability, and weight tying between them is
        simpler with plain nn.Linear.
    """

    def __init__(self, config: LaaLMConfig):
        super().__init__()
        self.config = config

        # Kept as nn.Embedding — embeddings don't benefit from FP8 matmuls
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final norm: te.RMSNorm
        self.norm_f = te.RMSNorm(config.d_model)

        # lm_head kept as nn.Linear for weight tying with token_emb.
        # Running the vocab projection in BF16 is standard practice.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

        self.apply(self._init_weights)

        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape

        x = self.token_emb(idx)

        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = gradient_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss

    def generate(
        self,
        idx,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        stop_token: Optional[int] = None,
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
                if stop_token is not None and idx_next.item() == stop_token:
                    break
                continue

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if stop_token is not None and idx_next.item() == stop_token:
                break

        return idx
