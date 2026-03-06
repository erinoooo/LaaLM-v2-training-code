"""
LaaLM-v2 Training Script — LSTM on CUDA (single or multi-GPU via torchrun)

Removed from original:
  - transformer_engine / FP8 (te.fp8_autocast, DelayedScaling, te.Linear, etc.)
  - torch_xla / TPU backend

Bug fixes:
  - Replaced hand-rolled contextlib_nullcontext with contextlib.nullcontext
  - Single-GPU distributed init now uses env:// properly (avoids hard-coded port)
  - evaluate() no longer calls non-existent fp8_recipe argument
  - model.no_sync() guard now correctly falls through to nullcontext on 1 GPU

Usage:
  Single GPU:   python train.py
  Multi-GPU:    torchrun --nproc_per_node=4 train.py
"""

import os
import math
import time
import json
from contextlib import nullcontext
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

from model import LaaLMv2Config, LaaLMModel


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class TrainConfig:
    # Data
    data_path:       str   = "laalm_v2_training_data_v3.jsonl"
    val_data_path:   str   = "splits/laalm_v2_val.jsonl"
    tokenizer_path:  str   = "laalm_v2_tokenizer_v3.json"
    val_split_ratio: float = 0.05

    # Training
    batch_size:                  int   = 32
    gradient_accumulation_steps: int   = 4
    max_steps:                   int   = 50000
    warmup_steps:                int   = 3000
    eval_interval:               int   = 500
    save_interval:               int   = 2000

    # Optimizer
    learning_rate: float = 2e-4
    min_lr_ratio:  float = 0.01
    weight_decay:  float = 0.1
    beta1:         float = 0.9
    beta2:         float = 0.95
    grad_clip:     float = 1.0

    # System
    dtype:       torch.dtype = torch.bfloat16
    num_workers: int         = 4
    compile:     bool        = False

    # Checkpointing
    output_dir:    str = "checkpoints_v2"
    wandb_project: str = "laalm-v2"
    wandb_run_name: str = "laalm-v2-cuda-lstm"


# ============================================================================
# DATASET
# ============================================================================

class LaaLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=1024, verbose=True):
        self.max_len = max_len
        eos_id = tokenizer.token_to_id("</s>") or 3

        if verbose:
            print(f"Loading and tokenizing {data_path}...")
        all_tokens = []
        n_convs = 0
        with open(data_path) as f:
            for line in f:
                conv = json.loads(line)
                tokens = tokenizer.encode(conv["text"]).ids
                all_tokens.extend(tokens)
                all_tokens.append(eos_id)
                n_convs += 1

        total_tokens = len(all_tokens)
        n_chunks     = total_tokens // (max_len + 1)
        all_tokens   = all_tokens[: n_chunks * (max_len + 1)]

        self.data = torch.tensor(all_tokens, dtype=torch.long).view(n_chunks, max_len + 1)
        del all_tokens

        if verbose:
            print(f"  Conversations: {n_convs:,}")
            print(f"  Total tokens:  {total_tokens:,}")
            print(f"  Chunks ({max_len}): {n_chunks:,}")
            print(f"  Utilization:   {n_chunks * (max_len + 1) / total_tokens * 100:.1f}%")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


# ============================================================================
# LR SCHEDULE
# ============================================================================

def get_lr(step: int, config: TrainConfig) -> float:
    min_lr = config.learning_rate * config.min_lr_ratio
    if step < config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    if step >= config.max_steps:
        return min_lr
    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (config.learning_rate - min_lr)


# ============================================================================
# OPTIMIZER
# ============================================================================

def configure_optimizer(model: nn.Module, config: TrainConfig, is_master: bool = True):
    raw = model.module if isinstance(model, DDP) else model
    decay, no_decay = [], []
    for name, param in raw.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    if is_master:
        n_decay    = sum(p.numel() for p in decay)
        n_no_decay = sum(p.numel() for p in no_decay)
        print(f"  Weight decay: {n_decay:,} | No decay: {n_no_decay:,}")

    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


# ============================================================================
# DISTRIBUTED HELPERS
# ============================================================================

def all_reduce_mean(tensor: torch.Tensor) -> float:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Evaluate on val set. Returns global-average loss across all ranks."""
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    n_batches  = 0

    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        _, loss = raw(x, y)
        total_loss += loss.detach().float()
        n_batches  += 1

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    n_batches_t = torch.tensor(n_batches, device=device, dtype=torch.float32)
    dist.all_reduce(n_batches_t, op=dist.ReduceOp.SUM)

    raw.train()
    return (total_loss / n_batches_t.clamp(min=1)).item()


# ============================================================================
# TRAINING
# ============================================================================

def train():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE",  1))
    is_master  = rank == 0

    # ---- Distributed init ----
    # torchrun sets MASTER_ADDR / MASTER_PORT and uses the "env://" init method
    # by default — no need to hard-code a port. Works for both 1 and N GPUs.
    dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model_config = LaaLMv2Config()
    train_config = TrainConfig()

    eff_batch       = train_config.batch_size * train_config.gradient_accumulation_steps * world_size
    tokens_per_step = eff_batch * model_config.max_seq_len

    if is_master:
        print("=" * 60)
        print("LaaLM-v2 Training — CUDA LSTM")
        print("=" * 60)
        print(f"\nModel: {model_config.n_params / 1e6:.1f}M parameters")
        print(f"  d_model={model_config.d_model}, n_layers={model_config.n_layers}, "
              f"max_seq_len={model_config.max_seq_len}")
        print(f"\nDistributed: {world_size} GPU(s), rank {rank}")
        print(f"\nTraining:")
        print(f"  Per-GPU batch:   {train_config.batch_size} × {train_config.gradient_accumulation_steps} accum")
        print(f"  Effective batch: {eff_batch}")
        print(f"  Tokens/step:     {tokens_per_step:,}")
        print(f"  Optimizer steps: {train_config.max_steps:,}")
        print(f"  LR:              {train_config.learning_rate} → "
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
                "effective_batch_size": eff_batch,
                "tokens_per_step":      tokens_per_step,
                "world_size":           world_size,
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
            print(f"No val split found — holding out {train_config.val_split_ratio * 100:.0f}%")
        full_dataset = LaaLMDataset(
            train_config.data_path, tokenizer,
            max_len=model_config.max_seq_len, verbose=is_master,
        )
        n_val   = max(int(len(full_dataset) * train_config.val_split_ratio), 1)
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False,
    )

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

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    if is_master:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel on device. Trainable params: {n / 1e6:.2f}M")

    # ---- Optimizer ----
    optimizer = configure_optimizer(model, train_config, is_master=is_master)

    # ---- Training loop ----
    model.train()
    optimizer_step   = 0
    micro_step       = 0
    best_val_loss    = float("inf")
    tokens_processed = 0
    epoch            = 0
    step_loss_accum  = 0.0

    Path(train_config.output_dir).mkdir(exist_ok=True, parents=True)
    train_iter = iter(train_loader)

    if is_master:
        print(f"\nStarting training for {train_config.max_steps:,} optimizer steps...")
    pbar = tqdm(total=train_config.max_steps, desc="Training") if is_master else None
    t0   = time.time()

    while optimizer_step < train_config.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            epoch += 1
            train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        is_last_micro = (micro_step + 1) % train_config.gradient_accumulation_steps == 0

        # Suppress DDP allreduce on non-final micro-steps
        sync_ctx = (
            nullcontext()
            if (world_size == 1 or is_last_micro)
            else model.no_sync()
        )

        with sync_ctx:
            _, loss = model(x, y)
            scaled  = loss / train_config.gradient_accumulation_steps
            scaled.backward()

        step_loss_accum  += scaled.detach().float().item()
        micro_step       += 1
        tokens_processed += x.numel()

        if is_last_micro:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )

            lr = get_lr(optimizer_step, train_config)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_t   = torch.tensor(step_loss_accum, device=device, dtype=torch.float32)
            avg_loss = all_reduce_mean(loss_t)
            step_loss_accum = 0.0

            dt  = time.time() - t0
            tps = tokens_processed * world_size / dt if dt > 0 else 0

            if is_master:
                wandb.log({
                    "train/loss":           avg_loss,
                    "train/perplexity":     math.exp(min(avg_loss, 20)),
                    "train/lr":             lr,
                    "train/grad_norm":      grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/tokens_per_sec": tps,
                    "step":                 optimizer_step,
                })
                pbar.set_description(f"loss={avg_loss:.4f} lr={lr:.1e} tps={tps:.0f}")

            # ---- Validation ----
            if optimizer_step > 0 and optimizer_step % train_config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                val_ppl  = math.exp(min(val_loss, 20))

                if is_master:
                    wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "step": optimizer_step})
                    improved = ""
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        raw = model.module if isinstance(model, DDP) else model
                        torch.save({
                            "optimizer_step":    optimizer_step,
                            "model_state_dict":  raw.state_dict(),
                            "config":            model_config,
                            "val_loss":          val_loss,
                        }, f"{train_config.output_dir}/laalm_v2_best.pt")
                        improved = " ** NEW BEST **"
                    tqdm.write(
                        f"  [step {optimizer_step}] val_loss={val_loss:.4f} ppl={val_ppl:.2f}{improved}"
                    )

            # ---- Checkpoint ----
            if is_master and optimizer_step > 0 and optimizer_step % train_config.save_interval == 0:
                raw = model.module if isinstance(model, DDP) else model
                ckpt = f"{train_config.output_dir}/checkpoint_{optimizer_step}.pt"
                torch.save({
                    "optimizer_step":       optimizer_step,
                    "model_state_dict":     raw.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config":               model_config,
                    "best_val_loss":        best_val_loss,
                }, ckpt)
                tqdm.write(f"  Saved checkpoint → {ckpt}")

            optimizer_step += 1
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # ---- Final eval + save ----
    final_val = evaluate(model, val_loader, device)
    final_ppl = math.exp(min(final_val, 20))

    if is_master:
        raw = model.module if isinstance(model, DDP) else model
        torch.save({
            "model_state_dict": raw.state_dict(),
            "config":           model_config,
            "best_val_loss":    best_val_loss,
        }, f"{train_config.output_dir}/laalm_v2_final.pt")

        print()
        print("=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"  Final val loss: {final_val:.4f} (ppl: {final_ppl:.2f})")
        print(f"  Best val loss:  {best_val_loss:.4f} (ppl: {math.exp(min(best_val_loss, 20)):.2f})")
        print(f"  Total time:     {time.time() - t0:.0f}s")
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
