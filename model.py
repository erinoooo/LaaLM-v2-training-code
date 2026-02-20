"""
LaaLM Model Architecture
Shared module with base configuration and version-specific configs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class LaaLMConfig:
    """Base configuration for LaaLM models.

    This is the generic configuration class that can be used for any version
    of LaaLM or customized for new experiments.
    """
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    dropout: float = 0.1
    use_swiglu: bool = False
    # Recompute layer activations during backward instead of storing all
    # n_layers worth simultaneously.  Cuts activation memory from
    # O(n_layers) to O(1) at the cost of one extra forward pass per layer.
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        """Validate configuration and compute derived values."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        self.head_dim = self.d_model // self.n_heads
        self.n_params = self.calculate_params()

    def calculate_params(self) -> int:
        """Calculate total model parameters (excluding RoPE which is computed)."""
        # Token embeddings (tied with lm_head, so only count once)
        emb = self.vocab_size * self.d_model

        # Per-layer parameters
        attn_qkv = 3 * self.d_model * self.d_model  # Q, K, V projections
        attn_out = self.d_model * self.d_model       # Output projection
        if self.use_swiglu:
            ffn = 3 * self.d_model * self.d_ff  # w1 (gate) + w3 (up) + w2 (down)
        else:
            ffn = 2 * self.d_model * self.d_ff  # w1 (up) + w2 (down)
        layer_norm = 2 * self.d_model  # 2 RMSNorms per layer
        per_layer = attn_qkv + attn_out + ffn + layer_norm

        # Final layer norm
        final_ln = self.d_model

        # Total (lm_head is tied with token_emb, so don't double-count)
        total = emb + (per_layer * self.n_layers) + final_ln
        return total


@dataclass
class LaaLMv2Config(LaaLMConfig):
    """LaaLM v2 configuration with default hyperparameters.

    v2 uses:
    - 8K vocab with BPE tokenization
    - 1024 hidden dim, 20 layers, 16 heads (~265M parameters)
    - SwiGLU FFN with d_ff=2816 (8/3 * d_model, rounded to multiple of 256)
    - 8192 max sequence length for extended conversation context
    - Trained on unambiguous delimiter format with reasoning traces
    """
    vocab_size: int = 8000
    d_model: int = 1024
    n_layers: int = 20
    n_heads: int = 16
    d_ff: int = 2816
    max_seq_len: int = 1024
    dropout: float = 0.05
    use_swiglu: bool = True
    use_gradient_checkpointing: bool = True  # required to fit 20 layers on 30 GB TPU


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute for efficiency
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
    """Apply rotary embeddings to queries and keys."""
    q_rot = torch.cat((-q[..., q.shape[-1] // 2 :], q[..., : q.shape[-1] // 2]), dim=-1)
    k_rot = torch.cat((-k[..., k.shape[-1] // 2 :], k[..., : k.shape[-1] // 2]), dim=-1)
    return q * cos + q_rot * sin, k * cos + k_rot * sin


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, config: LaaLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)

    def forward(self, x):
        B, T, C = x.shape

        # QKV projection and split
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rope(x, T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Feed-forward network.

    Supports two modes:
    - Standard: w2(silu(w1(x)))
    - SwiGLU:   w2(silu(w1(x)) * w3(x))

    SwiGLU is empirically better at the same parameter count.
    With SwiGLU, use d_ff = 2/3 * original_d_ff to keep param count equal.
    """

    def __init__(self, config: LaaLMConfig):
        super().__init__()
        self.use_swiglu = config.use_swiglu

        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if self.use_swiglu:
            self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)

    def forward(self, x):
        if self.use_swiglu:
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        else:
            return self.dropout(self.w2(F.silu(self.w1(x))))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config: LaaLMConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LaaLMModel(nn.Module):
    """LaaLM Language Model.

    A decoder-only transformer model with:
    - Token embeddings with weight tying
    - Rotary position embeddings (RoPE)
    - Pre-norm transformer blocks with RMSNorm
    - SwiGLU or SiLU feed-forward networks
    - Scaled residual initialization for training stability
    """

    def __init__(self, config: LaaLMConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm
        self.norm_f = RMSNorm(config.d_model)

        # Output projection (tied with token embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

        # Scale residual projections (out_proj, w2) by 1/sqrt(2*n_layers)
        # for training stability in deep networks
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            idx: Input token indices [B, T]
            targets: Target token indices [B, T] (optional, for training)

        Returns:
            logits: Output logits [B, T, vocab_size]
            loss: Cross-entropy loss (if targets provided)
        """
        B, T = idx.shape

        # Token embeddings
        x = self.token_emb(idx)

        # Transformer blocks
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                # use_reentrant=False is the recommended API for XLA/TPU:
                # it doesn't rely on Python re-entrancy and traces correctly
                # under XLA's lazy evaluation model.
                x = gradient_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Final norm and projection
        x = self.norm_f(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # Ignore pad token in loss
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
        """Generate tokens autoregressively.

        Args:
            idx: Starting token indices [B, T]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = unchanged, <1 = sharper)
            top_k: If set, only sample from top k tokens
            stop_token: If set, stop when this token is generated

        Returns:
            Generated token indices [B, T + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature
            else:
                # Greedy
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
                if stop_token is not None and idx_next.item() == stop_token:
                    break
                continue

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

            # Check stop condition
            if stop_token is not None and idx_next.item() == stop_token:
                break

        return idx
