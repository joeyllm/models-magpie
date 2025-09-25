# main_fsdp.py
import os
import glob
import random
from typing import List, Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from transformers import get_scheduler, DataCollatorForLanguageModeling

from model.mistral_7b_au import load_mistral_model
from train.trainer import Trainer

# FSDP bits
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from functools import partial

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------- utils ----------------
class Logger:
    def __init__(self, run, is_main: bool = True):
        self.run = run
        self.is_main = is_main

    def print(self, *args, **kwargs):
        if self.is_main:
            print(*args, **kwargs)

    def wb(self, _tag, metrics: Dict = None, step: int = None, **kwargs):
        if self.run is not None and self.is_main and metrics:
            wandb.log(metrics)


def set_seeds(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_parquet_files(base_dir: str):
    pattern = os.path.join(base_dir, "**", "*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {base_dir}")
    return files


def tokenize_batch(examples, tok, max_len: int):
    # Only produce tokenized fields; collator will pad and create labels
    text = examples.get("text") or examples.get("content") or examples.get("body")
    if text is None:
        raise KeyError("Dataset must include 'text' (or 'content'/'body').")
    return tok(
        text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_attention_mask=True,
    )


# --------------- main ------------------
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # --- Distributed init ---
    # Expect torchrun to set these env vars
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    set_seeds(int(cfg["trainconfig"]["seed"]))

    # --- W&B on main process only ---
    wb_cfg = cfg["wandbconfig"]
    run = None
    if wb_cfg.get("mode", "online") != "disabled" and rank == 0:
        run = wandb.init(
            project=wb_cfg["project"],
            entity=wb_cfg.get("entity"),
            name=wb_cfg.get("name"),
            config=cfg,
            mode=wb_cfg.get("mode", "online"),
        )
    logger = Logger(run=run, is_main=(rank == 0))

    if rank == 0:
        logger.print("🔎 Collecting parquet shards...")
    parquet_files = collect_parquet_files(cfg["dataconfig"]["data_path"])
    if rank == 0:
        logger.print(f"Found {len(parquet_files)} files.")

    if rank == 0:
        logger.print("📦 Loading dataset...")
    raw_ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train")

    if rank == 0:
        logger.print("🧠 Loading model + tokenizer (LoRA, bf16, FSDP-ready)...")
    # IMPORTANT: device_map must be None when using FSDP so FSDP owns placement
    model, tokenizer = load_mistral_model(
        model_cfg=OmegaConf.create(cfg["modelconfig"]),
        train_cfg=OmegaConf.create(cfg["trainconfig"]),
        device_map=None,
        use_fsdp=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Pad token:", tokenizer.pad_token)
    print("Pad token ID:", tokenizer.pad_token_id)

    # FSDP wrap policy for Transformer blocks
    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MistralDecoderLayer},   # wrap each decoder block
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16 if cfg["trainconfig"]["bf16"] else torch.float16,
        reduce_dtype=torch.float32,     # <— change to fp32 for stability
        buffer_dtype=torch.bfloat16 if cfg["trainconfig"]["bf16"] else torch.float16,
    )

    # Move base to the correct CUDA before wrapping (helps with some PEFT/bnb combinations)
    model.to(device)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=device,
    )

    if rank == 0:
        # Count trainable (post-PEFT)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.print(f"🔧 Trainable parameters: {trainable:,}")

    if rank == 0:
        logger.print("✍️ Tokenizing...")
    ds_tok = raw_ds.map(
        lambda ex: tokenize_batch(ex, tokenizer, cfg["dataconfig"]["max_seq_len"]),
        batched=True,
        remove_columns=raw_ds.column_names,  # keep only tokenized fields
        desc="Tokenizing" if rank == 0 else None,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    if rank == 0:
        logger.print("🧰 Building DataLoader + DistributedSampler...")
    sampler = DistributedSampler(
        ds_tok,
        num_replicas=world_size,
        rank=rank,
        shuffle=bool(cfg["dataconfig"]["shuffle"]),
        drop_last=bool(cfg["dataconfig"]["drop_last"]),
    )

    loader = DataLoader(
        ds_tok,
        sampler=sampler,
        batch_size=int(cfg["dataconfig"]["batch_size"]),
        num_workers=int(cfg["dataconfig"]["num_workers"]),
        pin_memory=bool(cfg["dataconfig"]["pin_memory"]),
        collate_fn=data_collator,
    )

    if rank == 0:
        logger.print("📈 Setting up optimizer & scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizerconfig"]["lr"]),
        betas=tuple(cfg["optimizerconfig"]["betas"]),
        weight_decay=float(cfg["optimizerconfig"]["weight_decay"]),
    )

    # Compute steps
    # steps_per_epoch = ceil(N_batches/world_size/accum) but DataLoader len already sampler-adjusted
    steps_per_epoch = (len(loader) + int(cfg["trainconfig"]["accumulation_steps"]) - 1) // int(
        cfg["trainconfig"]["accumulation_steps"]
    )
    max_steps = cfg["trainconfig"]["total_steps"] or (steps_per_epoch * int(cfg["trainconfig"]["epochs"]))
    warmup_steps = int(float(cfg["schedulerconfig"]["warmup_ratio"]) * max_steps)

    scheduler = get_scheduler(
        name=cfg["schedulerconfig"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    if rank == 0:
        logger.wb("log", metrics={"params/trainable": trainable})

    if rank == 0:
        logger.print("🚀 Launching training (FSDP)…")
    trainer = Trainer(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        logger=logger,
        rank=rank,
        total_steps=max_steps,
        scheduler=scheduler,
        accumulation_steps=int(cfg["trainconfig"]["accumulation_steps"]),
        save_model_path=cfg["trainconfig"]["output_dir"],
        log_freq=int(cfg["trainconfig"]["logging_steps"]),
        save_steps=int(cfg["trainconfig"]["save_steps"]),
        max_grad_norm=float(cfg["schedulerconfig"]["max_grad_norm"]),
        use_bf16=bool(cfg["trainconfig"]["bf16"]),
        device=device,
        save_optimizer_state=False,  # FSDP opt state can be huge; enable if you really need it
    )

    trainer.train(int(cfg["trainconfig"]["epochs"]))

    # Clean finalize
    if rank == 0 and run is not None:
        run.finish()

    # Ensure all ranks complete before teardown
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()




# # main_fsdp.py
# import os
# import glob
# import random
# from typing import List, Dict, Optional

# import torch
# import torch.distributed as dist
# from torch.utils.data import DataLoader, DistributedSampler

# import hydra
# import wandb
# from omegaconf import DictConfig, OmegaConf
# from datasets import load_dataset
# from transformers import get_scheduler, DataCollatorForLanguageModeling

# from model.mistral_7b_au import load_mistral_model
# from train.trainer import Trainer

# from functools import partial

# # FSDP bits
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import MixedPrecision
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# from torch.distributed.fsdp import ShardingStrategy
# from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
# from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


# # ---------------- utils ----------------
# class Logger:
#     def __init__(self, run, is_main: bool = True):
#         self.run = run
#         self.is_main = is_main

#     def print(self, *args, **kwargs):
#         if self.is_main:
#             print(*args, **kwargs)

#     def wb(self, _tag, metrics: Dict = None, step: int = None, **kwargs):
#         if self.run is not None and self.is_main and metrics:
#             wandb.log(metrics)


# def set_seeds(seed: int = 42):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def collect_parquet_files(base_dir: str):
#     pattern = os.path.join(base_dir, "**", "*.parquet")
#     files = sorted(glob.glob(pattern, recursive=True))
#     if not files:
#         raise FileNotFoundError(f"No parquet files found under: {base_dir}")
#     return files


# def tokenize_batch(examples, tok, max_len: int):
#     # Only produce tokenized fields; collator will pad and create labels
#     text = examples.get("text") or examples.get("content") or examples.get("body")
#     if text is None:
#         raise KeyError("Dataset must include 'text' (or 'content'/'body').")
#     return tok(
#         text,
#         truncation=True,
#         max_length=max_len,
#         padding=False,
#         return_attention_mask=True,
#     )


# # --------------- main ------------------
# @hydra.main(version_base=None, config_path="configs", config_name="config")
# def main(cfg: DictConfig):
#     cfg = OmegaConf.to_container(cfg, resolve=True)

#     # --- Distributed init ---
#     # Expect torchrun to set these env vars
#     rank = int(os.environ.get("RANK", "0"))
#     local_rank = int(os.environ.get("LOCAL_RANK", "0"))
#     world_size = int(os.environ.get("WORLD_SIZE", "1"))

#     dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
#     torch.cuda.set_device(local_rank)
#     device = torch.device(f"cuda:{local_rank}")

#     set_seeds(int(cfg["trainconfig"]["seed"]))

#     # --- W&B on main process only ---
#     wb_cfg = cfg["wandbconfig"]
#     run = None
#     if wb_cfg.get("mode", "online") != "disabled" and rank == 0:
#         run = wandb.init(
#             project=wb_cfg["project"],
#             entity=wb_cfg.get("entity"),
#             name=wb_cfg.get("name"),
#             config=cfg,
#             mode=wb_cfg.get("mode", "online"),
#         )
#     logger = Logger(run=run, is_main=(rank == 0))

#     if rank == 0:
#         logger.print("🔎 Collecting parquet shards...")
#     parquet_files = collect_parquet_files(cfg["dataconfig"]["data_path"])
#     if rank == 0:
#         logger.print(f"Found {len(parquet_files)} files.")

#     if rank == 0:
#         logger.print("📦 Loading dataset...")
#     raw_ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train")

#     if rank == 0:
#         logger.print("🧠 Loading model + tokenizer (LoRA, bf16, FSDP-ready)...")
#     # IMPORTANT: device_map must be None when using FSDP so FSDP owns placement
#     model, tokenizer = load_mistral_model(
#         model_cfg=OmegaConf.create(cfg["modelconfig"]),
#         train_cfg=OmegaConf.create(cfg["trainconfig"]),
#         device_map=None,
#         use_fsdp=True,
#     )
#     tokenizer.padding_side = "right"
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # FSDP wrap policy for Transformer blocks
#     auto_wrap = partial(
#             transformer_auto_wrap_policy,
#             transformer_layer_cls={MistralDecoderLayer},   # wrap each decoder block
#         )

#     mp_policy = MixedPrecision(
#                 param_dtype=torch.bfloat16 if cfg["trainconfig"]["bf16"] else torch.float16,
#                 reduce_dtype=torch.float32,     # <— change to fp32 for stability
#                 buffer_dtype=torch.bfloat16 if cfg["trainconfig"]["bf16"] else torch.float16,
#             )

#     # Move base to the correct CUDA before wrapping (helps with some PEFT/bnb combinations)
#     model.to(device)

#     model = FSDP(
#         model,
#         auto_wrap_policy=auto_wrap,
#         mixed_precision=mp_policy,
#         sharding_strategy=ShardingStrategy.FULL_SHARD,
#         use_orig_params=True,
#         device_id=device,
#     )

#     if rank == 0:
#         # Count trainable (post-PEFT)
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         logger.print(f"🔧 Trainable parameters: {trainable:,}")

#     if rank == 0:
#         logger.print("✍️ Tokenizing...")
#     ds_tok = raw_ds.map(
#         lambda ex: tokenize_batch(ex, tokenizer, cfg["dataconfig"]["max_seq_len"]),
#         batched=True,
#         remove_columns=raw_ds.column_names,  # keep only tokenized fields
#         desc="Tokenizing" if rank == 0 else None,
#     )

#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False,
#         pad_to_multiple_of=8,
#     )

#     if rank == 0:
#         logger.print("🧰 Building DataLoader + DistributedSampler...")
#     sampler = DistributedSampler(
#         ds_tok,
#         num_replicas=world_size,
#         rank=rank,
#         shuffle=bool(cfg["dataconfig"]["shuffle"]),
#         drop_last=bool(cfg["dataconfig"]["drop_last"]),
#     )

#     loader = DataLoader(
#         ds_tok,
#         sampler=sampler,
#         batch_size=int(cfg["dataconfig"]["batch_size"]),
#         num_workers=int(cfg["dataconfig"]["num_workers"]),
#         pin_memory=bool(cfg["dataconfig"]["pin_memory"]),
#         collate_fn=data_collator,
#     )

#     if rank == 0:
#         logger.print("📈 Setting up optimizer & scheduler...")
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=float(cfg["optimizerconfig"]["lr"]),
#         betas=tuple(cfg["optimizerconfig"]["betas"]),
#         weight_decay=float(cfg["optimizerconfig"]["weight_decay"]),
#     )

#     # Compute steps
#     # steps_per_epoch = ceil(N_batches/world_size/accum) but DataLoader len already sampler-adjusted
#     steps_per_epoch = (len(loader) + int(cfg["trainconfig"]["accumulation_steps"]) - 1) // int(
#         cfg["trainconfig"]["accumulation_steps"]
#     )
#     max_steps = cfg["trainconfig"]["total_steps"] or (steps_per_epoch * int(cfg["trainconfig"]["epochs"]))
#     warmup_steps = int(float(cfg["schedulerconfig"]["warmup_ratio"]) * max_steps)

#     scheduler = get_scheduler(
#         name=cfg["schedulerconfig"]["lr_scheduler_type"],
#         optimizer=optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=max_steps,
#     )

#     if rank == 0:
#         logger.wb("log", metrics={"params/trainable": trainable})

#     if rank == 0:
#         logger.print("🚀 Launching training (FSDP)…")
#     trainer = Trainer(
#         model=model,
#         dataloader=loader,
#         optimizer=optimizer,
#         logger=logger,
#         rank=rank,
#         total_steps=max_steps,
#         scheduler=scheduler,
#         accumulation_steps=int(cfg["trainconfig"]["accumulation_steps"]),
#         save_model_path=cfg["trainconfig"]["output_dir"],
#         log_freq=int(cfg["trainconfig"]["logging_steps"]),
#         save_steps=int(cfg["trainconfig"]["save_steps"]),
#         use_bf16=bool(cfg["trainconfig"]["bf16"]),
#         device=device,
#         save_optimizer_state=False,  # FSDP opt state can be huge; enable if you really need it
#     )

#     trainer.train(int(cfg["trainconfig"]["epochs"]))

#     # Clean finalize
#     if rank == 0 and run is not None:
#         run.finish()

#     # Ensure all ranks complete before teardown
#     dist.barrier()
#     dist.destroy_process_group()


# if __name__ == "__main__":
#     main()
