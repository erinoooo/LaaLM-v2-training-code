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
  9. Multi-chip data parallelism — all TPU chips train in parallel via
     xmp.spawn + DistributedSampler + xm.reduce_gradients

TPU backend: Uses PyTorch/XLA for Google Cloud TPU acceleration.
Multi-chip:  xmp.spawn launches one worker process per chip.  Each worker
             owns one chip, sees a unique data shard (DistributedSampler),
             and gradients are averaged across chips via xm.reduce_gradients
             before every optimizer update.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
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
    # batch=32 × accum=4 × world_size chips = effective global batch.
    # Per-chip attention score tensor stays at [32,4,1024,1024] = 256 MB (bf16)
    # regardless of world_size — memory footprint per chip is unchanged.
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
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

    def __init__(self, data_path, tokenizer, max_len=8192, verbose=True):
        self.max_len = max_len
        eos_id = tokenizer.token_to_id("</s>")
        if eos_id is None:
            eos_id = 3

        if verbose:
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

        if verbose:
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

def configure_optimizer(model, config, is_master=True):
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
    if is_master:
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

    In multi-chip mode, each chip evaluates its own data shard and the
    results are aggregated via xm.mesh_reduce (a collective called by all
    processes simultaneously) before returning the global average.
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
    local_loss = total_loss_xla.item()

    # Aggregate loss and batch count across all chips.
    # mesh_reduce is a collective — every process calls it simultaneously.
    # On a single chip it is a no-op (sum of one value).
    global_loss = xm.mesh_reduce('val_loss', local_loss, sum)
    global_n = xm.mesh_reduce('val_n_batches', float(n_batches), sum)
    result = global_loss / max(global_n, 1)

    model.train()
    return result

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(index):
    """Training worker — one instance runs on each TPU chip via xmp.spawn.

    Args:
        index: Chip ordinal (0..world_size-1), injected automatically by
               xmp.spawn.
    """
    model_config = LaaLMv2Config()
    train_config = TrainConfig()

    # ---- TPU device & distributed setup ----
    device = xm.xla_device()
    train_config.device = device
    world_size = xr.world_size()
    rank = xr.global_ordinal()
    is_master = (rank == 0)

    if is_master:
        print(f"Distributed setup: {world_size} TPU chip(s) — rank {rank} on {device}")

    if is_master:
        print("=" * 60)
        print("LaaLM-v2 Training — Quality Optimized (TPU)")
        print("=" * 60)
        print(f"\nModel: {model_config.n_params/1e6:.1f}M parameters")
        print(f"  d_model={model_config.d_model}, n_layers={model_config.n_layers}, "
              f"n_heads={model_config.n_heads}, d_ff={model_config.d_ff}")
        print(f"  max_seq_len={model_config.max_seq_len}, "
              f"swiglu={'yes' if model_config.use_swiglu else 'no'}, "
              f"dropout={model_config.dropout}")

    # Effective batch scales linearly with the number of chips.
    eff_batch = train_config.batch_size * train_config.gradient_accumulation_steps * world_size
    tokens_per_step = eff_batch * model_config.max_seq_len

    if is_master:
        print(f"\nTraining:")
        print(f"  Chips:           {world_size}")
        print(f"  Per-chip batch:  {train_config.batch_size} x {train_config.gradient_accumulation_steps} accum")
        print(f"  Effective batch: {eff_batch} (per-chip × {world_size} chips)")
        print(f"  Tokens/step:     {tokens_per_step:,}")
        print(f"  Optimizer steps: {train_config.max_steps:,}")
        print(f"  Warmup:          {train_config.warmup_steps:,} steps")
        print(f"  LR:              {train_config.learning_rate} -> "
              f"{train_config.learning_rate * train_config.min_lr_ratio}")
        print()

    # ---- Wandb (master only) ----
    if is_master:
        wandb_config = {**model_config.__dict__, **train_config.__dict__}
        wandb_config['dtype'] = str(train_config.dtype)
        wandb_config['effective_batch_size'] = eff_batch
        wandb_config['tokens_per_step'] = tokens_per_step
        wandb_config['world_size'] = world_size
        # Default to offline mode so wandb never blocks on network I/O.
        # Cloud TPU VMs often can't reach wandb servers.
        # To sync later:  wandb sync ./wandb/run-*/
        # To use online:  WANDB_MODE=online python train.py
        wandb.init(
            project=train_config.wandb_project,
            name=train_config.wandb_run_name,
            config=wandb_config,
            mode=os.environ.get("WANDB_MODE", "offline"),
        )

    # ---- Data ----
    tokenizer = Tokenizer.from_file(train_config.tokenizer_path)

    # Each chip loads its own copy of the dataset (independent I/O).
    # Only the master chip prints progress to avoid 4× duplicate output.
    if Path(train_config.val_data_path).exists():
        if is_master:
            print("Using existing validation split")
        train_dataset = LaaLMDataset(
            train_config.data_path, tokenizer,
            max_len=model_config.max_seq_len, verbose=is_master,
        )
        val_dataset = LaaLMDataset(
            train_config.val_data_path, tokenizer,
            max_len=model_config.max_seq_len, verbose=False,
        )
    else:
        if is_master:
            print(f"No val split found — holding out {train_config.val_split_ratio*100:.0f}%")
        full_dataset = LaaLMDataset(
            train_config.data_path, tokenizer,
            max_len=model_config.max_seq_len, verbose=is_master,
        )
        n_val = max(int(len(full_dataset) * train_config.val_split_ratio), 1)
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    # DistributedSampler gives each chip a non-overlapping shard of the data.
    # set_epoch() is called at every epoch boundary so the shuffle seed changes,
    # preventing chips from seeing the same ordering across epochs.
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=42,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False,
    )

    # num_workers=0: the dataset is a pre-tokenized in-memory tensor so
    # __getitem__ is O(1) tensor slicing — background workers add no benefit
    # and can deadlock when forked inside xmp.spawn child processes.
    # MpDeviceLoader handles host→TPU transfer asynchronously regardless.
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        sampler=val_sampler,
        num_workers=0,
    )

    # Wrap DataLoaders with MpDeviceLoader for efficient host-to-TPU transfer
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    val_device_loader = pl.MpDeviceLoader(val_loader, device)

    if is_master:
        print(f"\n  Train chunks: {len(train_dataset):,}")
        print(f"  Val chunks:   {len(val_dataset):,}")

    # ---- Model ----
    model = LaaLMModel(model_config)
    model = model.to(train_config.dtype).to(device)

    if train_config.compile:
        if is_master:
            print("\nCompiling model with torch.compile (openxla backend)...")
        model = torch.compile(model, backend='openxla')

    # ---- Optimizer ----
    optimizer = configure_optimizer(model, train_config, is_master=is_master)

    # ---- Training state ----
    model.train()
    optimizer_step = 0
    micro_step = 0
    best_val_loss = float('inf')
    tokens_processed = 0
    epoch = 0
    # XLA scalar for deferred loss accumulation — stays on device until after
    # xm.optimizer_step() to avoid splitting the compiled graph prematurely
    step_loss_xla = torch.zeros((), device=device, dtype=torch.float32)

    if is_master:
        Path(train_config.output_dir).mkdir(exist_ok=True)
    train_iter = iter(train_device_loader)

    # ---- Train ----
    # Note: the first optimizer step will trigger XLA graph compilation, which
    # can take several minutes.  This is a one-time cost per process restart.
    if is_master:
        print(f"\nStarting training for {train_config.max_steps:,} optimizer steps...")
        print("(XLA graph compiles on the first step — expect a delay before the bar moves)\n")
    pbar = tqdm(total=train_config.max_steps, desc="Training") if is_master else None
    t0 = time.time()

    while optimizer_step < train_config.max_steps:
        # Get batch (cycle through epochs)
        # MpDeviceLoader handles host-to-device transfer automatically.
        # set_epoch() updates DistributedSampler's shuffle seed each epoch so
        # chips see different orderings and never repeat the same shard pairing.
        try:
            x, y = next(train_iter)
        except StopIteration:
            epoch += 1
            train_sampler.set_epoch(epoch)
            train_iter = iter(train_device_loader)
            x, y = next(train_iter)

        # Forward + backward (micro-step)
        # Model is already in bfloat16 on TPU — no autocast needed
        _, loss = model(x, y)
        scaled_loss = loss / train_config.gradient_accumulation_steps

        # Accumulate loss on the XLA device — do NOT call .item() here.
        step_loss_xla = step_loss_xla + scaled_loss.detach().float()
        scaled_loss.backward()
        micro_step += 1
        tokens_processed += x.numel()

        # Between gradient accumulation micro-steps (but NOT before the
        # optimizer step), flush the XLA graph so activations from this
        # micro-step are freed before the next one is traced.  Without this,
        # XLA lazily accumulates ALL micro-step graphs into one giant graph
        # (gradient_accumulation_steps × the per-step graph size), which
        # multiplies the compilation memory requirement by the accumulation
        # factor.  Gradients are tensors on the device and survive mark_step.
        is_last_micro_step = (micro_step % train_config.gradient_accumulation_steps == 0)
        if not is_last_micro_step:
            xm.mark_step()

        # ---- Optimizer step (every gradient_accumulation_steps micro-steps) ----
        if micro_step % train_config.gradient_accumulation_steps == 0:
            # All-reduce gradients across chips before clipping.
            # xm.reduce_gradients performs REDUCE_SUM / world_size so every
            # chip gets the identical mean gradient.  This is a no-op when
            # world_size == 1.  Must happen BEFORE clip_grad_norm_ so that
            # clipping is applied to the globally averaged gradient.
            xm.reduce_gradients(optimizer)

            # Clip the averaged gradients.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )

            # LR schedule
            lr = get_lr(optimizer_step, train_config)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # xm.optimizer_step flushes the entire accumulated XLA graph
            # (forward + backward + allreduce + clip + optimizer updates)
            # in one compiled kernel.
            xm.optimizer_step(optimizer)
            optimizer.zero_grad(set_to_none=True)

            # ---- Logging ----
            # Read loss AFTER the graph flush — step_loss_xla is now a
            # materialized scalar; no live activations remain in TPU RAM.
            local_avg_loss = step_loss_xla.item()
            # Average training loss across chips for accurate global reporting
            avg_loss = xm.mesh_reduce(
                'train_loss', local_avg_loss,
                lambda vals: sum(vals) / len(vals),
            )
            step_loss_xla = torch.zeros((), device=device, dtype=torch.float32)

            dt = time.time() - t0
            # Global token throughput: each chip processed tokens_processed
            # tokens; multiply by world_size for total across all chips.
            tps = tokens_processed * world_size / dt if dt > 0 else 0

            if is_master:
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
                # evaluate() is a collective — all chips must call it together
                val_loss = evaluate(model, val_device_loader, train_config)
                val_ppl = math.exp(min(val_loss, 20))
                if is_master:
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
            if is_master and optimizer_step > 0 and optimizer_step % train_config.save_interval == 0:
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
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # ---- Final save & eval ----
    # evaluate() is collective — all chips participate; only master saves.
    final_val = evaluate(model, val_device_loader, train_config)
    final_ppl = math.exp(min(final_val, 20))

    if is_master:
        final_path = f"{train_config.output_dir}/laalm_v2_final.pt"
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        xm.save({
            'model_state_dict': cpu_state,
            'config': model_config,
            'best_val_loss': best_val_loss,
        }, final_path)

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
    # xmp.spawn launches one worker process per TPU chip and injects the chip
    # ordinal as the first argument to train().  nprocs=None lets torch_xla
    # auto-detect the number of available TPU cores (4 on a v2-4/v3-4, 8 on
    # v3-8, etc.).  If auto-detection fails, set nprocs explicitly, e.g.:
    #   xmp.spawn(train, args=(), nprocs=4)
    xmp.spawn(train, args=(), nprocs=None)
