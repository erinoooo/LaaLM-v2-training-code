"""
LaaLM Model Architecture — LSTM Version

Migrated from Transformer + NVIDIA TE/FP8 to a stacked LSTM LM.

Changes:
  - Removed: MultiHeadAttention, FeedForward, TransformerBlock, RotaryEmbedding
  - Removed: transformer_engine dependency entirely
  - Added:   nn.LSTM (stacked, batch_first, inter-layer dropout)
  - Config:  removed n_heads, d_ff, use_swiglu (not used by LSTM)
  - Init:    orthogonal init for recurrent weights; forget-gate bias=1
  - forward(): signature unchanged → (logits, loss), train.py needs no edits
  - generate(): uses incremental hidden-state caching → O(1) per new token
                instead of re-encoding the full context each step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class LaaLMConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    max_seq_len: int
    dropout: float = 0.1
    # Kept for checkpoint-loading compat; unused by LSTM
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        self.n_params = self._calculate_params()

    def _calculate_params(self) -> int:
        emb = self.vocab_size * self.d_model
        # 4 gates × (W_ih + W_hh + 2 biases) per layer
        lstm = self.n_layers * 4 * (
            self.d_model * self.d_model   # W_ih
            + self.d_model * self.d_model  # W_hh
            + 2 * self.d_model             # biases
        )
        norm = 2 * self.d_model            # LayerNorm weight + bias
        head = self.d_model * self.vocab_size  # lm_head (tied, but count once)
        return emb + lstm + norm + head


@dataclass
class LaaLMv2Config(LaaLMConfig):
    """
    LaaLM v2 — LSTM edition.

    d_model bumped to 256: LSTMs need a wider hidden dim to compensate for
    the lack of attention's global mixing, while staying in the same ~10M
    parameter budget.
    """
    vocab_size: int = 8000
    d_model: int   = 256
    n_layers: int  = 4
    max_seq_len: int = 1024
    dropout: float = 0.1
    use_gradient_checkpointing: bool = False


# ============================================================================
# MODEL
# ============================================================================

class LaaLMModel(nn.Module):
    """
    Stacked-LSTM Language Model with weight-tied input/output embeddings.

    Training forward:  (idx, targets) → (logits, loss)   [train.py unchanged]
    Inference forward: generate(idx, ...) uses _forward_with_hidden() to cache
                       the LSTM hidden state across steps instead of recomputing.
    """

    def __init__(self, config: LaaLMConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_drop  = nn.Dropout(config.dropout)

        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            # dropout only applied *between* layers; must be 0 for single layer
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            batch_first=True,
        )

        self.norm        = nn.LayerNorm(config.d_model)
        self.output_drop = nn.Dropout(config.dropout)

        # lm_head in standard nn.Linear for weight tying
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    # Xavier for input→hidden weights
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    # Orthogonal for hidden→hidden weights — standard LSTM best practice
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Set forget-gate bias to 1.0 — empirically reduces vanishing gradients
                    # LSTM bias layout: [input | forget | cell | output] gate, each size d_model
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    # ------------------------------------------------------------------
    # Forward (training / full-sequence)
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx     : (B, T) long — input token ids
            targets : (B, T) long — target token ids, or None

        Returns:
            logits : (B, T, vocab_size)
            loss   : scalar or None
        """
        x = self.emb_drop(self.token_emb(idx))
        # Hidden state is reset to zero each batch — correct for fixed-context LM training
        x, _ = self.lstm(x)
        x = self.output_drop(self.norm(x))
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Incremental forward (inference only) — threads hidden state through
    # ------------------------------------------------------------------

    def _forward_with_hidden(
        self,
        idx: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One step forward that passes hidden state in and out.
        Used only by generate() — not called during training."""
        x = self.emb_drop(self.token_emb(idx))
        x, hidden = self.lstm(x, hidden)
        x = self.output_drop(self.norm(x))
        logits = self.lm_head(x)
        return logits, hidden

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        stop_token: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Efficiency: encodes the full prompt in a single LSTM pass to prime
        the hidden state, then generates one token at a time — O(prompt + new)
        vs O(prompt * new) for re-encoding each step.
        """
        self.eval()

        # Truncate prompt to fit context window
        if idx.size(1) > self.config.max_seq_len:
            idx = idx[:, -self.config.max_seq_len:]

        # ---- Prime hidden state from prompt ----
        hidden = None
        if idx.size(1) > 1:
            prompt_emb = self.emb_drop(self.token_emb(idx[:, :-1]))
            _, hidden = self.lstm(prompt_emb, hidden)

        # ---- Autoregressively generate ----
        current = idx[:, -1:]   # (B, 1) — last token of prompt

        for _ in range(max_new_tokens):
            logits, hidden = self._forward_with_hidden(current, hidden)
            logits = logits[:, -1, :]   # (B, vocab)

            if temperature <= 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx     = torch.cat((idx, idx_next), dim=1)
            current = idx_next

            if stop_token is not None and idx_next.item() == stop_token:
                break

        return idx
