"""
LaaLM-v2 Training Script — Optimized for Maximum Model Quality (TPU)

Key improvements over naive training:
  1. Pre-tokenized concatenated data — zero padding waste, uses 99%+ of all
     tokens instead of truncating 80%+ of each conversation
  2. Validation evaluation loop — tracks val loss, saves best model checkpoint
  3. Optimizer-step counting — max_steps/warmup/save/eval all count actual
     optimizer updates, not micro-batches
  4. Proper weight decay groups — 2D params (matrices) get decay, 1D params
     (norms, biases) do not
  5. SwiGLU architecture — same param count, better quality
  6. Scaled residual initialization — stable training for deep networks
  7. Cosine schedule with lower min_lr — more refined final training
  8. Gradient norm logging — detect instability early

TPU backend: Uses PyTorch/XLA for Google Cloud TPU acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import json
import math
import time
from pathlib import Path
from tqdm import tqdm
import wandb
from dataclasses import dataclass

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
except ImportError as e:
    _msg = str(e)
    if "undefined symbol" in _msg or "_XLAC" in _msg:
        raise SystemExit(
            "\n[ERROR] torch_xla version mismatch with torch.\n"
            "torch and torch_xla must be exactly the same version.\n\n"
            "Fix — run these commands in your venv:\n"
            "  pip show torch               # note the version (e.g. 2.5.1)\n"
            "  pip uninstall torch_xla -y\n"
            "  pip install torch_xla==<same-version-as-torch>\n\n"
            "Or reinstall both together:\n"
            "  pip install torch==2.5.1 torch_xla==2.5.0\n\n"
            "On a Google Cloud TPU VM, use the pre-installed environment\n"
            "or follow: https://pytorch.org/xla/release/r2.5/index.html\n\n"
            f"Original error: {_msg}"
        ) from None
    raise

from model import LaaLMv2Config, LaaLMModel

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class TrainConfig:
    # Data
    data_path: str = "laalm_v2_training_data_v3.jsonl"
    val_data_path: str = "splits/laalm_v2_val.jsonl"
    tokenizer_path: str = "laalm_v2_tokenizer_v3.json"
    val_split_ratio: float = 0.05  # Fallback if val file doesn't exist

    # Training — all step counts are OPTIMIZER steps (not micro-batches)
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    max_steps: int = 50000   # 50K optimizer steps
    warmup_steps: int = 3000
    eval_interval: int = 500
    save_interval: int = 2000

    # Optimizer
    learning_rate: float = 2e-4
    min_lr_ratio: float = 0.01  # Minimum LR = learning_rate * 0.01
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # System — TPU via PyTorch/XLA
    # device is set dynamically via xm.xla_device(); this field is kept for
    # compatibility but overridden in train()
    device: str = "xla"
    dtype: torch.dtype = torch.bfloat16  # TPU natively supports bfloat16
    compile: bool = False  # torch.compile with XLA backend; disabled by default

    # Checkpointing
    output_dir: str = "checkpoints_v2"
    wandb_project: str = "laalm-v2"
    wandb_run_name: str = "laalm-v2-quality"

# ============================================================================
# DATASET — PRE-TOKENIZED & CONCATENATED
# ============================================================================

class LaaLMDataset(Dataset):
    """Pre-tokenized, concatenated dataset for maximum data utilization.

    Instead of truncating each conversation to max_len (throwing away 80%+
    of long conversations) and padding short ones (wasting compute on pad
    tokens), this:
      1. Tokenizes all conversations
      2. Concatenates them with EOS separators
      3. Splits the stream into fixed-size chunks

    Result: zero padding, zero waste, 99%+ token utilization.
    """

    def __init__(self, data_path, tokenizer, max_len=8192):
        self.max_len = max_len
        eos_id = tokenizer.token_to_id("</s>")
        if eos_id is None:
            eos_id = 3

        print(f"Loading and tokenizing {data_path}...")
        all_tokens = []
        n_convs = 0
        with open(data_path) as f:
            for line in f:
                conv = json.loads(line)
                tokens = tokenizer.encode(conv['text']).ids
                all_tokens.extend(tokens)
                all_tokens.append(eos_id)
                n_convs += 1

        total_tokens = len(all_tokens)

        # Each chunk needs max_len+1 tokens (for the input/target shift)
        n_chunks = total_tokens // (max_len + 1)
        usable = n_chunks * (max_len + 1)
        all_tokens = all_tokens[:usable]

        self.data = torch.tensor(all_tokens, dtype=torch.long).view(n_chunks, max_len + 1)
        del all_tokens  # free the large Python list immediately; tensor owns the data now

        print(f"  Conversations: {n_convs:,}")
        print(f"  Total tokens:  {total_tokens:,}")
        print(f"  Chunks ({max_len}): {n_chunks:,}")
        print(f"  Utilization:   {usable / total_tokens * 100:.1f}%")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]  # (input, target)

# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================

def get_lr(optimizer_step, config):
    """Cosine annealing with linear warmup."""
    min_lr = config.learning_rate * config.min_lr_ratio

    if optimizer_step < config.warmup_steps:
        return config.learning_rate * optimizer_step / config.warmup_steps
    if optimizer_step > config.max_steps:
        return min_lr

    progress = (optimizer_step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (config.learning_rate - min_lr)

# ============================================================================
# OPTIMIZER WITH PROPER WEIGHT DECAY
# ============================================================================

def configure_optimizer(model, config):
    """Create AdamW with proper weight decay groups.

    2D parameters (weight matrices) get weight decay.
    1D parameters (norms, biases, embeddings) do not.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"  Weight decay: {n_decay:,} params | No decay: {n_no_decay:,} params")

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )

# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def evaluate(model, val_loader, config):
    """Evaluate model on validation set. Returns average loss.

    Accumulates loss as an XLA scalar tensor across all val batches so that
    only ONE device sync (.item()) happens at the end, instead of one per
    batch.  MpDeviceLoader already places tensors on the device so we skip
    the redundant .to() calls.
    """
    model.eval()
    total_loss_xla = torch.zeros((), device=config.device, dtype=torch.float32)
    n_batches = 0

    for x, y in val_loader:
        _, loss = model(x, y)
        total_loss_xla = total_loss_xla + loss.detach().float()
        n_batches += 1

    # Single graph flush for the whole validation pass
    xm.mark_step()
    result = (total_loss_xla / max(n_batches, 1)).item()

    model.train()
    return result

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train():
    model_config = LaaLMv2Config()
    train_config = TrainConfig()

    # ---- TPU device setup ----
    device = xm.xla_device()
    train_config.device = device
    print(f"Using TPU device: {device}")

    print("=" * 60)
    print("LaaLM-v2 Training — Quality Optimized (TPU)")
    print("=" * 60)
    print(f"\nModel: {model_config.n_params/1e6:.1f}M parameters")
    print(f"  d_model={model_config.d_model}, n_layers={model_config.n_layers}, "
          f"n_heads={model_config.n_heads}, d_ff={model_config.d_ff}")
    print(f"  max_seq_len={model_config.max_seq_len}, "
          f"swiglu={'yes' if model_config.use_swiglu else 'no'}, "
          f"dropout={model_config.dropout}")
    eff_batch = train_config.batch_size * train_config.gradient_accumulation_steps
    tokens_per_step = eff_batch * model_config.max_seq_len
    print(f"\nTraining:")
    print(f"  Effective batch: {eff_batch} ({train_config.batch_size} x {train_config.gradient_accumulation_steps})")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Optimizer steps: {train_config.max_steps:,}")
    print(f"  Warmup: {train_config.warmup_steps:,} steps")
    print(f"  LR: {train_config.learning_rate} -> {train_config.learning_rate * train_config.min_lr_ratio}")
    print()

    # ---- Wandb ----
    wandb_config = {**model_config.__dict__, **train_config.__dict__}
    wandb_config['dtype'] = str(train_config.dtype)
    wandb_config['effective_batch_size'] = eff_batch
    wandb_config['tokens_per_step'] = tokens_per_step
    wandb.init(
        project=train_config.wandb_project,
        name=train_config.wandb_run_name,
        config=wandb_config,
    )

    # ---- Data ----
    tokenizer = Tokenizer.from_file(train_config.tokenizer_path)

    if Path(train_config.val_data_path).exists():
        print("Using existing validation split")
        train_dataset = LaaLMDataset(train_config.data_path, tokenizer, max_len=model_config.max_seq_len)
        val_dataset = LaaLMDataset(train_config.val_data_path, tokenizer, max_len=model_config.max_seq_len)
    else:
        print(f"No val split found — holding out {train_config.val_split_ratio*100:.0f}%")
        full_dataset = LaaLMDataset(train_config.data_path, tokenizer, max_len=model_config.max_seq_len)
        n_val = max(int(len(full_dataset) * train_config.val_split_ratio), 1)
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    # num_workers > 0 lets the host prefetch data on background threads so
    # the TPU never idles waiting for the next batch.
    # persistent_workers keeps the worker processes alive across epochs,
    # avoiding re-spawn overhead at every StopIteration.
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # Wrap DataLoaders with MpDeviceLoader for efficient host-to-TPU transfer
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    val_device_loader = pl.MpDeviceLoader(val_loader, device)

    print(f"\n  Train chunks: {len(train_dataset):,}")
    print(f"  Val chunks:   {len(val_dataset):,}")

    # ---- Model ----
    model = LaaLMModel(model_config)
    model = model.to(train_config.dtype).to(device)

    if train_config.compile:
        print("\nCompiling model with torch.compile (openxla backend)...")
        model = torch.compile(model, backend='openxla')

    # ---- Optimizer ----
    optimizer = configure_optimizer(model, train_config)

    # ---- Training state ----
    model.train()
    optimizer_step = 0
    micro_step = 0
    running_loss = 0.0
    best_val_loss = float('inf')
    tokens_processed = 0
    # XLA scalar for deferred loss accumulation — stays on device until after
    # xm.optimizer_step() to avoid splitting the compiled graph prematurely
    step_loss_xla = torch.zeros((), device=device, dtype=torch.float32)

    Path(train_config.output_dir).mkdir(exist_ok=True)
    train_iter = iter(train_device_loader)

    # ---- Initial validation ----
    val_loss = evaluate(model, val_device_loader, train_config)
    val_ppl = math.exp(min(val_loss, 20))
    print(f"\nInitial val loss: {val_loss:.4f} (ppl: {val_ppl:.2f})")
    wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "step": 0})

    # ---- Train ----
    print(f"\nStarting training for {train_config.max_steps:,} optimizer steps...\n")
    pbar = tqdm(total=train_config.max_steps, desc="Training")
    t0 = time.time()

    while optimizer_step < train_config.max_steps:
        # Get batch (cycle through epochs)
        # MpDeviceLoader handles host-to-device transfer automatically
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_device_loader)
            x, y = next(train_iter)

        # Forward + backward (micro-step)
        # Model is already in bfloat16 on TPU — no autocast needed
        _, loss = model(x, y)
        scaled_loss = loss / train_config.gradient_accumulation_steps

        # Accumulate loss on the XLA device — do NOT call .item() here.
        # Calling .item() before xm.optimizer_step() forces an early graph
        # flush, splitting the forward+backward graph from the optimizer graph.
        # That doubles peak TPU RAM usage and causes two graph compilations
        # per step instead of one.
        step_loss_xla = step_loss_xla + scaled_loss.detach().float()
        scaled_loss.backward()
        micro_step += 1
        tokens_processed += x.numel()

        # ---- Optimizer step (every gradient_accumulation_steps micro-steps) ----
        if micro_step % train_config.gradient_accumulation_steps == 0:
            # Clip gradients.
            # xm.reduce_gradients() is a distributed allreduce — omit on
            # single-device training to avoid unnecessary communication overhead.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )

            # LR schedule
            lr = get_lr(optimizer_step, train_config)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # xm.optimizer_step flushes the entire accumulated XLA graph
            # (forward + backward + optimizer updates) in one compiled kernel.
            xm.optimizer_step(optimizer)
            optimizer.zero_grad(set_to_none=True)

            # ---- Logging ----
            # Read loss AFTER the graph flush — step_loss_xla is now a
            # materialized scalar; no live activations remain in TPU RAM.
            avg_loss = step_loss_xla.item()
            step_loss_xla = torch.zeros((), device=device, dtype=torch.float32)
            running_loss = avg_loss
            dt = time.time() - t0
            tps = tokens_processed / dt if dt > 0 else 0

            wandb.log({
                "train/loss": avg_loss,
                "train/perplexity": math.exp(min(avg_loss, 20)),
                "train/lr": lr,
                "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "train/tokens_per_sec": tps,
                "step": optimizer_step,
            })
            pbar.set_description(
                f"loss={avg_loss:.4f} lr={lr:.1e} tps={tps:.0f}"
            )

            # ---- Validation ----
            if optimizer_step > 0 and optimizer_step % train_config.eval_interval == 0:
                val_loss = evaluate(model, val_device_loader, train_config)
                val_ppl = math.exp(min(val_loss, 20))
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                    "step": optimizer_step,
                })

                improved = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = f"{train_config.output_dir}/laalm_v2_best.pt"
                    # Move state dict to CPU for portable checkpoints
                    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    xm.save({
                        'optimizer_step': optimizer_step,
                        'model_state_dict': cpu_state,
                        'config': model_config,
                        'val_loss': val_loss,
                    }, best_path)
                    improved = " ** NEW BEST **"

                tqdm.write(
                    f"  [step {optimizer_step}] val_loss={val_loss:.4f} "
                    f"ppl={val_ppl:.2f}{improved}"
                )

            # ---- Checkpoint ----
            if optimizer_step > 0 and optimizer_step % train_config.save_interval == 0:
                ckpt_path = f"{train_config.output_dir}/checkpoint_{optimizer_step}.pt"
                cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                cpu_opt_state = {
                    k: {sk: sv.cpu() if torch.is_tensor(sv) else sv
                        for sk, sv in v.items()} if isinstance(v, dict) else v
                    for k, v in optimizer.state_dict().items()
                }
                xm.save({
                    'optimizer_step': optimizer_step,
                    'model_state_dict': cpu_state,
                    'optimizer_state_dict': cpu_opt_state,
                    'config': model_config,
                    'best_val_loss': best_val_loss,
                }, ckpt_path)
                tqdm.write(f"  Saved checkpoint -> {ckpt_path}")

            optimizer_step += 1
            pbar.update(1)

    pbar.close()

    # ---- Final save & eval ----
    final_path = f"{train_config.output_dir}/laalm_v2_final.pt"
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    xm.save({
        'model_state_dict': cpu_state,
        'config': model_config,
        'best_val_loss': best_val_loss,
    }, final_path)

    final_val = evaluate(model, val_device_loader, train_config)
    final_ppl = math.exp(min(final_val, 20))

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"  Final val loss:  {final_val:.4f} (ppl: {final_ppl:.2f})")
    print(f"  Best val loss:   {best_val_loss:.4f} (ppl: {math.exp(min(best_val_loss, 20)):.2f})")
    print(f"  Total time:      {time.time() - t0:.0f}s")
    print(f"  Best model:      {train_config.output_dir}/laalm_v2_best.pt")
    print(f"  Final model:     {final_path}")
    print("=" * 60)

    wandb.finish()


if __name__ == "__main__":
    train()
