"""
LaaLM Model Architecture — LSTM, production-ready

Bug fixes over previous version:
  - _calculate_params: lm_head is tied to token_emb — was double-counted,
    showing 6.2M instead of the actual 4.15M unique parameters
  - generate(): self.eval() had no try/finally — an exception mid-generation
    permanently left the model in eval mode during training
  - padding_idx=0 added to Embedding + zero-out after init; pad token was
    accumulating gradient into the embedding table
  - @torch.inference_mode() is the correct decorator for generate (stronger
    than @no_grad: also disables version tracking for autograd)
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
    d_model:    int
    n_layers:   int
    max_seq_len: int
    dropout:    float = 0.1
    use_gradient_checkpointing: bool = False  # kept for compat

    def __post_init__(self):
        self.n_params = self._calculate_params()

    def _calculate_params(self) -> int:
        # Embedding (= lm_head via weight tying — counted once)
        emb = self.vocab_size * self.d_model
        # LSTM: 4 gates × (W_ih + W_hh + bias_ih + bias_hh) per layer
        # input_size == hidden_size == d_model for every layer
        lstm = self.n_layers * (
            4 * self.d_model * self.d_model   # W_ih
            + 4 * self.d_model * self.d_model  # W_hh
            + 4 * self.d_model                 # bias_ih
            + 4 * self.d_model                 # bias_hh
        )
        norm = 2 * self.d_model  # LayerNorm: weight + bias
        return emb + lstm + norm  # lm_head tied with emb → NOT added again


@dataclass
class LaaLMv2Config(LaaLMConfig):
    vocab_size:  int   = 8000
    d_model:     int   = 256
    n_layers:    int   = 4
    max_seq_len: int   = 512
    dropout:     float = 0.1
    use_gradient_checkpointing: bool = False


# ============================================================================
# MODEL
# ============================================================================

class LaaLMModel(nn.Module):
    """
    Stacked-LSTM Language Model with weight-tied input/output embeddings.

    forward():               (B,T) → (logits, loss)   — training path
    _forward_with_hidden():  threads LSTM state in/out — inference path
    generate():              autoregressive decoding with hidden-state caching
    """

    def __init__(self, config: LaaLMConfig):
        super().__init__()
        self.config = config

        # padding_idx=0: prevents pad-token embedding from receiving gradient
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.emb_drop  = nn.Dropout(config.dropout)

        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            # inter-layer dropout only when n_layers > 1; nn.LSTM requires 0.0 otherwise
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            batch_first=True,
        )

        self.norm        = nn.LayerNorm(config.d_model)
        self.output_drop = nn.Dropout(config.dropout)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Init all weights first, then tie (tying after init avoids lm_head
        # reinitialising the already-initialised embedding)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    # Orthogonal init for recurrent weights: standard LSTM best practice
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Forget-gate bias = 1.0: reduces vanishing gradients at init
                    # cuDNN layout: [input | forget | cell | output] gates, each d_model wide
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    # ------------------------------------------------------------------
    # Full-sequence forward — training
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.emb_drop(self.token_emb(idx))
        x, _ = self.lstm(x)                   # hidden resets to zero each batch
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
    # Stateful forward — inference
    # ------------------------------------------------------------------

    def _forward_with_hidden(
        self,
        idx: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single step (or short sequence) forward that threads hidden state."""
        x = self.emb_drop(self.token_emb(idx))
        x, hidden = self.lstm(x, hidden)
        x = self.output_drop(self.norm(x))
        logits = self.lm_head(x)
        return logits, hidden

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        stop_token: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encodes the prompt in one pass to prime the hidden state, then
        generates one token at a time — O(prompt + new_tokens) total work.

        Bug fix: saves and restores self.training so generate() is safe to call
        mid-training (e.g. for sample logging) without permanently switching the
        model to eval mode.
        """
        was_training = self.training
        self.eval()

        try:
            if idx.size(1) > self.config.max_seq_len:
                idx = idx[:, -self.config.max_seq_len:]

            # Prime hidden state with the full prompt (excluding last token)
            hidden = None
            if idx.size(1) > 1:
                _, hidden = self._forward_with_hidden(idx[:, :-1], hidden)

            current = idx[:, -1:]  # (B, 1) — last prompt token

            for _ in range(max_new_tokens):
                logits, hidden = self._forward_with_hidden(current, hidden)
                logits = logits[:, -1, :]  # (B, vocab)

                if temperature <= 0.0:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float("-inf")
                    probs    = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                idx     = torch.cat((idx, idx_next), dim=1)
                current = idx_next

                if stop_token is not None and idx_next.item() == stop_token:
                    break

        finally:
            # Always restore prior training state — even if an exception occurs
            if was_training:
                self.train()

        return idx
