"""
LaaLM-v2 Training Script — Migrated from TPU/XLA to NVIDIA L40S with FP8

Backend changes from original (TPU/XLA → CUDA/L40S):
  - torch_xla / xmp.spawn      → torch.distributed (NCCL) + torchrun
  - xm.xla_device()            → torch.device("cuda:<rank>")
  - xm.optimizer_step()        → optimizer.step() (standard)
  - xm.reduce_gradients()      → DDP handles allreduce automatically
  - xm.mesh_reduce()           → dist.all_reduce (explicit collectives)
  - pl.MpDeviceLoader          → standard DataLoader (tensors go to CUDA)
  - xm.save()                  → torch.save()
  - xm.mark_step()             → removed (XLA graph flushing not needed)
  - model.no_sync()            → used during gradient accumulation micro-steps
                                  to suppress redundant DDP allreduce

FP8 changes (new):
  - te.fp8_autocast()          → wraps the forward pass; activates FP8 Tensor
                                  Core ops in all te.Linear / te.RMSNorm layers
  - DelayedScaling recipe      → per-tensor scale factors updated every step
                                  using a history of amax values (standard for
                                  FP8 training stability)
  - dtype stays bfloat16       → model weights + optimizer states in BF16;
                                  only the matmul inputs/outputs go through FP8

Usage:
  Single GPU:
    python train.py

  Multi-GPU (e.g. 4× L40S):
    torchrun --nproc_per_node=4 train.py
"""

import os
import math
import time
import json
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
import wandb
from tokenizers import Tokenizer

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

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
    val_split_ratio: float = 0.05

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_steps: int = 50000
    warmup_steps: int = 3000
    eval_interval: int = 500
    save_interval: int = 2000

    # Optimizer
    learning_rate: float = 2e-4
    min_lr_ratio: float = 0.01
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # System
    dtype: torch.dtype = torch.bfloat16  # L40S has good BF16 support; FP8 is applied on top
    num_workers: int = 4                 # CPU workers for DataLoader prefetch (unlike TPU, safe here)
    compile: bool = False                # torch.compile — optional, can help on CUDA

    # FP8 recipe (DelayedScaling)
    fp8_amax_history_len: int = 16       # how many steps to track tensor amax history
    fp8_amax_compute_algo: str = "max"   # "max" or "most_recent"

    # Checkpointing
    output_dir: str = "checkpoints_v2"
    wandb_project: str = "laalm-v2"
    wandb_run_name: str = "laalm-v2-l40s-fp8"


# ============================================================================
# DATASET — unchanged from original
# ============================================================================

class LaaLMDataset(Dataset):
    """Pre-tokenized, concatenated dataset. Unchanged from original."""

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
        n_chunks = total_tokens // (max_len + 1)
        usable = n_chunks * (max_len + 1)
        all_tokens = all_tokens[:usable]

        self.data = torch.tensor(all_tokens, dtype=torch.long).view(n_chunks, max_len + 1)
        del all_tokens

        if verbose:
            print(f"  Conversations: {n_convs:,}")
            print(f"  Total tokens:  {total_tokens:,}")
            print(f"  Chunks ({max_len}): {n_chunks:,}")
            print(f"  Utilization:   {usable / total_tokens * 100:.1f}%")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


# ============================================================================
# LEARNING RATE SCHEDULE — unchanged
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
# OPTIMIZER — unchanged logic, but no XLA-specific handling needed
# ============================================================================

def configure_optimizer(model, config, is_master=True):
    """AdamW with weight decay on 2D params only."""
    decay_params = []
    no_decay_params = []

    # If DDP-wrapped, access the underlying module for named_parameters
    raw_model = model.module if isinstance(model, DDP) else model

    for name, param in raw_model.named_parameters():
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
# DISTRIBUTED HELPERS
# ============================================================================

def all_reduce_mean(tensor: torch.Tensor) -> float:
    """Average a scalar tensor across all ranks. Returns Python float."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def evaluate(model, val_loader, device, fp8_recipe):
    """Evaluate on validation set. Returns global average loss across all ranks.

    FP8 note: we also wrap validation in fp8_autocast so the same FP8 weights
    are used — this gives accurate validation numbers that match train behavior.
    """
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    n_batches = 0

    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            _, loss = raw_model(x, y)
        total_loss += loss.detach().float()
        n_batches += 1

    # Reduce across ranks to get global average loss
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    n_batches_tensor = torch.tensor(n_batches, device=device, dtype=torch.float32)
    dist.all_reduce(n_batches_tensor, op=dist.ReduceOp.SUM)

    raw_model.train()
    return (total_loss / n_batches_tensor.clamp(min=1)).item()


# ============================================================================
# TRAINING
# ============================================================================

def train():
    # ---- Distributed setup ----
    # torchrun sets LOCAL_RANK, RANK, WORLD_SIZE automatically.
    # For single-GPU, these default to 0 / 0 / 1 via the fallback below.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
    else:
        # Single-GPU: create a trivial process group so dist.all_reduce still works
        dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29500",
                                world_size=1, rank=0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_master = (rank == 0)

    model_config = LaaLMv2Config()
    train_config = TrainConfig()

    # ---- FP8 recipe ----
    # DelayedScaling: reuses per-tensor scale factors from the previous step,
    # recomputes them every `interval` steps using a rolling amax history.
    # HYBRID format: E4M3 in forward pass (more precision), E5M2 in backward
    # (more dynamic range for gradients).
    fp8_recipe = DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=train_config.fp8_amax_history_len,
        amax_compute_algo=train_config.fp8_amax_compute_algo,
    )

    eff_batch = train_config.batch_size * train_config.gradient_accumulation_steps * world_size
    tokens_per_step = eff_batch * model_config.max_seq_len

    if is_master:
        print("=" * 60)
        print("LaaLM-v2 Training — NVIDIA L40S + FP8")
        print("=" * 60)
        print(f"\nModel: {model_config.n_params/1e6:.1f}M parameters")
        print(f"  d_model={model_config.d_model}, n_layers={model_config.n_layers}, "
              f"n_heads={model_config.n_heads}, d_ff={model_config.d_ff}")
        print(f"  max_seq_len={model_config.max_seq_len}, swiglu={'yes' if model_config.use_swiglu else 'no'}")
        print(f"\nDistributed: {world_size} GPU(s), rank {rank}")
        print(f"FP8 recipe: HYBRID, amax_history={train_config.fp8_amax_history_len}")
        print(f"\nTraining:")
        print(f"  Per-GPU batch:   {train_config.batch_size} x {train_config.gradient_accumulation_steps} accum")
        print(f"  Effective batch: {eff_batch}")
        print(f"  Tokens/step:     {tokens_per_step:,}")
        print(f"  Optimizer steps: {train_config.max_steps:,}")
        print(f"  LR:              {train_config.learning_rate} -> "
              f"{train_config.learning_rate * train_config.min_lr_ratio}")
        print()

    # ---- Wandb ----
    if is_master:
        wandb.init(
            project=train_config.wandb_project,
            name=train_config.wandb_run_name,
            config={
                **model_config.__dict__,
                **{k: str(v) if isinstance(v, torch.dtype) else v
                   for k, v in train_config.__dict__.items()},
                'effective_batch_size': eff_batch,
                'tokens_per_step': tokens_per_step,
                'world_size': world_size,
                'fp8': True,
            },
            mode="disabled",
        )

    # ---- Data ----
    tokenizer = Tokenizer.from_file(train_config.tokenizer_path)

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

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=42,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False,
    )

    # pin_memory=True: pre-pins host tensors so CUDA DMA transfer is faster.
    # num_workers > 0 is safe here (unlike inside xmp.spawn child processes).
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        sampler=val_sampler,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )

    if is_master:
        print(f"\n  Train chunks: {len(train_dataset):,}")
        print(f"  Val chunks:   {len(val_dataset):,}")

    # ---- Model ----
    model = LaaLMModel(model_config)
    model = model.to(train_config.dtype).to(device)

    if train_config.compile:
        if is_master:
            print("\nCompiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap with DDP after moving to device.
    # DDP automatically averages gradients across ranks on backward().
    # We use model.no_sync() during gradient accumulation micro-steps to
    # suppress the allreduce until the final micro-step.
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    if is_master:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel on device. Trainable params: {n_params/1e6:.2f}M")

    # ---- Optimizer ----
    optimizer = configure_optimizer(model, train_config, is_master=is_master)

    # ---- Training state ----
    model.train()
    optimizer_step = 0
    micro_step = 0
    best_val_loss = float('inf')
    tokens_processed = 0
    epoch = 0
    step_loss_accum = 0.0  # CPU float accumulator (unlike XLA, .item() is cheap on CUDA)

    Path(train_config.output_dir).mkdir(exist_ok=True, parents=True)
    train_iter = iter(train_loader)

    if is_master:
        print(f"\nStarting training for {train_config.max_steps:,} optimizer steps...")
    pbar = tqdm(total=train_config.max_steps, desc="Training") if is_master else None
    t0 = time.time()

    while optimizer_step < train_config.max_steps:
        # Get next batch, cycling through epochs
        try:
            x, y = next(train_iter)
        except StopIteration:
            epoch += 1
            train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        is_last_micro_step = ((micro_step + 1) % train_config.gradient_accumulation_steps == 0)

        # ---- Forward + backward (micro-step) ----
        # Use model.no_sync() on all but the last micro-step to prevent DDP
        # from doing an allreduce after each backward — we only want one
        # allreduce per optimizer step, not per micro-step.
        ctx = model.no_sync() if (world_size > 1 and not is_last_micro_step) else contextlib_nullcontext()

        with ctx:
            # te.fp8_autocast activates FP8 Tensor Core ops in all te.Linear
            # and te.RMSNorm layers. Outside this context they run in BF16.
            # The context must wrap only the forward pass — backward runs
            # outside it (TE handles the backward scaling factors internally).
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                _, loss = model(x, y)

            scaled_loss = loss / train_config.gradient_accumulation_steps
            scaled_loss.backward()

        step_loss_accum += scaled_loss.detach().float().item()
        micro_step += 1
        tokens_processed += x.numel()

        # ---- Optimizer step ----
        if is_last_micro_step:
            # Gradient clipping — applied after DDP has allreduced gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )

            # LR schedule
            lr = get_lr(optimizer_step, train_config)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Average training loss across all ranks
            loss_tensor = torch.tensor(step_loss_accum, device=device, dtype=torch.float32)
            avg_loss = all_reduce_mean(loss_tensor)
            step_loss_accum = 0.0

            dt = time.time() - t0
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
                # evaluate() calls dist.all_reduce — all ranks must enter together
                val_loss = evaluate(model, val_loader, device, fp8_recipe)
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
                        raw_model = model.module if isinstance(model, DDP) else model
                        torch.save({
                            'optimizer_step': optimizer_step,
                            'model_state_dict': raw_model.state_dict(),
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
                raw_model = model.module if isinstance(model, DDP) else model
                torch.save({
                    'optimizer_step': optimizer_step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': model_config,
                    'best_val_loss': best_val_loss,
                }, ckpt_path)
                tqdm.write(f"  Saved checkpoint -> {ckpt_path}")

            optimizer_step += 1
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # ---- Final eval & save ----
    final_val = evaluate(model, val_loader, device, fp8_recipe)
    final_ppl = math.exp(min(final_val, 20))

    if is_master:
        raw_model = model.module if isinstance(model, DDP) else model
        final_path = f"{train_config.output_dir}/laalm_v2_final.pt"
        torch.save({
            'model_state_dict': raw_model.state_dict(),
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

    if world_size > 1:
        dist.destroy_process_group()


# Small helper — contextlib.nullcontext equivalent inline
class contextlib_nullcontext:
    def __enter__(self): return self
    def __exit__(self, *args): pass


if __name__ == "__main__":
    train()
