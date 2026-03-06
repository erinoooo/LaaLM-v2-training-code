"""
LaaLM-v2 Training — CUDA LSTM, production-ready

Bug fixes over previous version:
  - eos_id: `token_to_id() or 3` evaluates to 3 when the real ID is 0
    (0 is falsy in Python). Fixed with explicit None check.
  - autocast was entirely missing: model weights are bf16 but every forward
    pass ran in fp32. Added torch.amp.autocast wrapping every forward.
  - evaluate() called raw.train() unconditionally — now restores prior state.
  - best.pt lacked optimizer_state_dict — resume from best was silently broken.
  - NaN/inf loss was silently accumulating into weights; now detected,
    synchronised across ranks, and the bad step is skipped.
  - TPS was a global average (hid slowdowns); now a windowed average.
  - Utilisation % was computed after truncation (always ~100%); fixed.

Performance additions:
  - torch.set_float32_matmul_precision('high') — TF32 on Ampere / Ada GPUs
  - torch.backends.cudnn.benchmark = True
  - torch.amp.autocast (also a correctness fix — see above)
  - Fused AdamW (fused=True) — single CUDA kernel, ~2x faster update
  - torch.compile(mode='reduce-overhead') — fuses embedding+norm+lm_head;
    reduces Python dispatch overhead around the cuDNN LSTM kernel
  - DataLoader: persistent_workers=True, prefetch_factor=4
  - Dataset: pin_memory() on the token tensor for async DMA transfers

New:
  - Checkpoint resume:
      RESUME=checkpoints_v2/checkpoint_10000.pt python train.py
      RESUME=checkpoints_v2/laalm_v2_best.pt    python train.py

Usage:
  Single GPU:   python train.py
  Multi-GPU:    torchrun --nproc_per_node=4 train.py
"""

import os
import math
import time
import json
import collections
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast
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
    batch_size:                  int   = 384
    gradient_accumulation_steps: int   = 1
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
    num_workers: int         = 64
    compile:     bool        = True   # torch.compile mode="reduce-overhead"

    # Checkpointing / logging
    output_dir:     str = "checkpoints_v2"
    wandb_project:  str = "laalm-v2"
    wandb_run_name: str = "laalm-v2-cuda-lstm"
    tps_window:     int = 50   # steps over which to compute windowed TPS


# ============================================================================
# DATASET
# ============================================================================

class LaaLMDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 512, verbose: bool = True):
        self.max_len = max_len

        # Bug fix: token_to_id() returns None on miss — `None or 3` works,
        # but `0 or 3` would silently return 3 if the EOS token happens to be ID 0.
        eos_id = tokenizer.token_to_id("</s>")
        if eos_id is None:
            eos_id = 3

        if verbose:
            print(f"Loading and tokenizing {data_path}...")

        all_tokens: list = []
        n_convs = 0
        with open(data_path) as f:
            for line in f:
                conv   = json.loads(line)
                tokens = tokenizer.encode(conv["text"]).ids
                all_tokens.extend(tokens)
                all_tokens.append(eos_id)
                n_convs += 1

        total_tokens = len(all_tokens)          # measure BEFORE truncation
        n_chunks     = total_tokens // (max_len + 1)
        usable       = n_chunks * (max_len + 1)
        all_tokens   = all_tokens[:usable]

        self.data = (
            torch.tensor(all_tokens, dtype=torch.long)
            .view(n_chunks, max_len + 1)
            .pin_memory()   # pre-pin for async host→device DMA
        )
        del all_tokens

        if verbose:
            print(f"  Conversations: {n_convs:,}")
            print(f"  Total tokens:  {total_tokens:,}")
            print(f"  Chunks ({max_len}): {n_chunks:,}")
            # Bug fix: compute utilisation from pre-truncation total
            print(f"  Utilisation:   {usable / total_tokens * 100:.1f}%")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


# ============================================================================
# LR SCHEDULE
# ============================================================================

def get_lr(step: int, cfg: TrainConfig) -> float:
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(cfg.warmup_steps, 1)
    if step >= cfg.max_steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.learning_rate - min_lr)


# ============================================================================
# OPTIMIZER
# ============================================================================

def configure_optimizer(model: nn.Module, cfg: TrainConfig, is_master: bool = True):
    raw    = model.module if isinstance(model, DDP) else model
    decay, no_decay = [], []
    for name, param in raw.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    if is_master:
        print(f"  Weight decay: {sum(p.numel() for p in decay):,} "
              f"| No decay: {sum(p.numel() for p in no_decay):,}")

    # fused=True: fused CUDA kernel for the AdamW update step — ~2x faster
    use_fused = torch.cuda.is_available()
    if is_master and use_fused:
        print("  Using fused AdamW")

    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        fused=use_fused,
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
    """All ranks must call this together (contains dist.all_reduce)."""
    raw = model.module if isinstance(model, DDP) else model
    # Bug fix: save and restore training state rather than unconditionally
    # calling raw.train() at the end (which broke callers that passed an
    # already-eval model)
    was_training = raw.training
    raw.eval()

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    n_batches  = 0

    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = raw(x, y)
        if not (loss.isnan() or loss.isinf()):
            total_loss += loss.detach().float()
            n_batches  += 1

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    n_batches_t = torch.tensor(n_batches, device=device, dtype=torch.float32)
    dist.all_reduce(n_batches_t, op=dist.ReduceOp.SUM)

    if was_training:
        raw.train()

    return (total_loss / n_batches_t.clamp(min=1)).item()


# ============================================================================
# TRAINING
# ============================================================================

def train():
    # ---- Distributed / device setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE",  1))
    is_master  = rank == 0

    if world_size > 1:
        # torchrun sets MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE
        dist.init_process_group(backend="nccl")
    else:
        # Plain `python train.py` — set env vars manually then init
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Global performance knobs
    torch.set_float32_matmul_precision("high")  # TF32 on Ampere/Ada
    torch.backends.cudnn.benchmark = True        # auto-tune cuDNN kernels

    model_config = LaaLMv2Config()
    train_config = TrainConfig()

    eff_batch       = train_config.batch_size * train_config.gradient_accumulation_steps * world_size
    tokens_per_step = eff_batch * model_config.max_seq_len

    if is_master:
        print("=" * 60)
        print("LaaLM-v2 Training — CUDA LSTM")
        print("=" * 60)
        print(f"\nModel: {model_config.n_params / 1e6:.2f}M parameters")
        print(f"  d_model={model_config.d_model}, n_layers={model_config.n_layers}, "
              f"max_seq_len={model_config.max_seq_len}")
        print(f"\nDistributed: {world_size} GPU(s), rank {rank}")
        print(f"Compile:     {train_config.compile}")
        print(f"\nTraining:")
        print(f"  Per-GPU batch:   {train_config.batch_size} × {train_config.gradient_accumulation_steps} accum")
        print(f"  Effective batch: {eff_batch}")
        print(f"  Tokens/step:     {tokens_per_step:,}")
        print(f"  Optimizer steps: {train_config.max_steps:,}")
        print(f"  LR range:        {train_config.learning_rate:.2e} → "
              f"{train_config.learning_rate * train_config.min_lr_ratio:.2e}")
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
            print(f"No val split — holding out {train_config.val_split_ratio * 100:.0f}%")
        full = LaaLMDataset(
            train_config.data_path, tokenizer,
            max_len=model_config.max_seq_len, verbose=is_master,
        )
        n_val   = max(int(len(full) * train_config.val_split_ratio), 1)
        n_train = len(full) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            full, [n_train, n_val], generator=torch.Generator().manual_seed(42),
        )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False,
    )

    # persistent_workers: keep worker processes alive between epochs
    # prefetch_factor:    pre-load N batches per worker ahead of consumption
    _w = train_config.num_workers
    loader_kw = dict(
        num_workers=_w,
        pin_memory=True,
        persistent_workers=(_w > 0),
        prefetch_factor=(4 if _w > 0 else None),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=train_sampler,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        sampler=val_sampler,
        **loader_kw,
    )

    if is_master:
        print(f"\n  Train chunks: {len(train_dataset):,}")
        print(f"  Val chunks:   {len(val_dataset):,}")

    # ---- Model ----
    model = LaaLMModel(model_config)
    model = model.to(train_config.dtype).to(device)

    if train_config.compile:
        if is_master:
            print("\nCompiling model (reduce-overhead)...")
        # reduce-overhead: minimises Python dispatch cost around the cuDNN LSTM
        # kernel and fuses embedding + LayerNorm + lm_head
        model = torch.compile(model, mode="reduce-overhead")

    # Wrap with DDP after compile — this is the correct order
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    if is_master:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model on device. Trainable params: {n / 1e6:.2f}M\n")

    # ---- Optimizer ----
    optimizer = configure_optimizer(model, train_config, is_master=is_master)

    # ---- Resume ----
    start_step    = 0
    best_val_loss = float("inf")
    resume_path   = os.environ.get("RESUME", "")

    if resume_path and os.path.exists(resume_path):
        if is_master:
            print(f"\nResuming from {resume_path}...")
        # Load to the correct GPU rank, not always cuda:0
        map_loc = {"cuda:0": f"cuda:{local_rank}"}
        ckpt    = torch.load(resume_path, map_location=map_loc, weights_only=False)
        raw     = model.module if isinstance(model, DDP) else model
        # torch.compile adds _orig_mod. prefix — strip it for compatibility
        state   = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        raw.load_state_dict(state, strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step    = ckpt.get("optimizer_step", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if is_master:
            print(f"  Step: {start_step}  best_val_loss: {best_val_loss:.4f}")

    # ---- Training state ----
    model.train()
    optimizer_step  = start_step
    micro_step      = 0
    step_loss_accum = 0.0
    nan_steps       = 0
    epoch           = 0
    tokens_total    = 0

    # Windowed TPS: circular buffer of (time, cumulative_tokens) snapshots
    tps_buf: collections.deque = collections.deque(maxlen=train_config.tps_window + 1)
    tps_buf.append((time.perf_counter(), 0))

    Path(train_config.output_dir).mkdir(exist_ok=True, parents=True)
    train_iter = iter(train_loader)
    t0 = time.perf_counter()

    if is_master:
        print(f"Starting training for {train_config.max_steps:,} steps "
              f"(from step {start_step})...")

    pbar = tqdm(total=train_config.max_steps, initial=start_step, desc="Training") if is_master else None

    while optimizer_step < train_config.max_steps:
        # ---- Fetch batch ----
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

        # ---- Forward + backward ----
        with sync_ctx:
            # autocast: ensures bf16 LSTM/matmul kernels are used — without this
            # the forward was running entirely in fp32 despite the bf16 weights
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            scaled = loss / train_config.gradient_accumulation_steps
            scaled.backward()

        # ---- NaN/inf detection — synchronised across ranks ----
        # Must happen outside sync_ctx so dist.all_reduce is not suppressed.
        # Without rank synchronisation, one rank could skip an optimizer step
        # while others don't, causing a distributed deadlock.
        is_bad_local = torch.tensor(
            float(loss.isnan() or loss.isinf()), device=device
        )
        if world_size > 1:
            dist.all_reduce(is_bad_local, op=dist.ReduceOp.MAX)

        if is_bad_local.item() > 0:
            nan_steps += 1
            optimizer.zero_grad(set_to_none=True)
            micro_step      = 0
            step_loss_accum = 0.0
            if is_master:
                tqdm.write(
                    f"  [step {optimizer_step}] WARNING: NaN/Inf loss "
                    f"(cumulative bad steps: {nan_steps}) — skipping"
                )
            continue

        step_loss_accum += scaled.detach().float().item()
        micro_step      += 1
        tokens_total    += x.numel()

        # ---- Optimizer step (every gradient_accumulation_steps micro-steps) ----
        if is_last_micro:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )

            lr = get_lr(optimizer_step, train_config)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Average loss across ranks
            loss_t   = torch.tensor(step_loss_accum, device=device, dtype=torch.float32)
            avg_loss = all_reduce_mean(loss_t)
            step_loss_accum = 0.0

            # Windowed TPS (last tps_window steps) — not total average
            now = time.perf_counter()
            tps_buf.append((now, tokens_total * world_size))
            if len(tps_buf) >= 2:
                dt  = tps_buf[-1][0] - tps_buf[0][0]
                tok = tps_buf[-1][1] - tps_buf[0][1]
                tps = tok / dt if dt > 0 else 0.0
            else:
                tps = 0.0

            if is_master:
                wandb.log({
                    "train/loss":           avg_loss,
                    "train/perplexity":     math.exp(min(avg_loss, 20)),
                    "train/lr":             lr,
                    "train/grad_norm":      grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/tokens_per_sec": tps,
                    "train/nan_steps":      nan_steps,
                    "step":                 optimizer_step,
                })
                pbar.set_description(
                    f"loss={avg_loss:.4f} lr={lr:.1e} tps={tps / 1000:.1f}k"
                )

            # ---- Validation ----
            if optimizer_step > 0 and optimizer_step % train_config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                val_ppl  = math.exp(min(val_loss, 20))

                if is_master:
                    wandb.log({
                        "val/loss":      val_loss,
                        "val/perplexity": val_ppl,
                        "step":           optimizer_step,
                    })
                    improved = ""
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        raw = model.module if isinstance(model, DDP) else model
                        # Bug fix: include optimizer_state_dict so best checkpoint
                        # can be used for resume (was missing in previous version)
                        torch.save({
                            "optimizer_step":       optimizer_step,
                            "model_state_dict":     raw.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config":               model_config,
                            "val_loss":             val_loss,
                            "best_val_loss":        best_val_loss,
                        }, f"{train_config.output_dir}/laalm_v2_best.pt")
                        improved = " ★ NEW BEST"
                    tqdm.write(
                        f"  [step {optimizer_step}] val_loss={val_loss:.4f} "
                        f"ppl={val_ppl:.2f}{improved}"
                    )

            # ---- Periodic checkpoint ----
            if is_master and optimizer_step > 0 and optimizer_step % train_config.save_interval == 0:
                raw  = model.module if isinstance(model, DDP) else model
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

        elapsed = time.perf_counter() - t0
        print()
        print("=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"  Final val loss: {final_val:.4f}  ppl={final_ppl:.2f}")
        print(f"  Best  val loss: {best_val_loss:.4f}  ppl={math.exp(min(best_val_loss, 20)):.2f}")
        print(f"  Total time:     {elapsed:.0f}s  ({elapsed / 3600:.1f}h)")
        print(f"  NaN steps:      {nan_steps}")
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
