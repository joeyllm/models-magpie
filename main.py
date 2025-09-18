# main.py
import os
import glob
import random
from typing import List, Dict

import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_scheduler
from transformers import DataCollatorForLanguageModeling

from model.mistral_7b_au import load_mistral_model
from train.trainer import Trainer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------------- utils --------------

class Logger:
    def __init__(self, run, is_main: bool = True):
        self.run = run
        self.is_main = is_main

    def print(self, *args, **kwargs):
        if self.is_main:
            print(*args, **kwargs)

    def wb(self, _tag, metrics: Dict = None, step: int = None, **kwargs):
        if self.run is not None and self.is_main and metrics:
            wandb.log(metrics, step=step)


def set_seeds(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_parquet_files(base_dir: str) -> List[str]:
    pattern = os.path.join(base_dir, "**", "*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {base_dir}")
    return files


def tokenize_batch(examples, tok, max_len: int):
    text = examples.get("text") or examples.get("content") or examples.get("body")
    if text is None:
        raise KeyError("Dataset rows must include a 'text' column (or 'content'/'body').")
    return tok(
        text,
        truncation=True,
        max_length=max_len,
        padding=False,              # no padding here; collator will pad per-batch
        return_attention_mask=True, # ok if missing; collator can also create it
    )



# -------------- main --------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    assert torch.cuda.is_available(), "CUDA GPU required."
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    set_seeds(cfg["trainconfig"]["seed"])

    # --- Weights & Biases ---
    wb_cfg = cfg["wandbconfig"]
    os.environ["WANDB_PROJECT"] = wb_cfg["project"]
    if "entity" in wb_cfg:
        os.environ["WANDB_ENTITY"] = wb_cfg["entity"]
    if wb_cfg.get("mode", "online") == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        run = None
    else:
        run = wandb.init(
            project=wb_cfg["project"],
            entity=wb_cfg.get("entity"),
            name=wb_cfg.get("name"),
            config=cfg,
            mode=wb_cfg.get("mode", "online"),
        )
    logger = Logger(run=run, is_main=True)

    logger.print("🔎 Collecting parquet shards...")
    parquet_files = collect_parquet_files(cfg["dataconfig"]["data_path"])
    logger.print(f"Found {len(parquet_files)} files.")

    logger.print("📦 Loading dataset...")
    ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train")

    logger.print("🧠 Loading model + tokenizer (QLoRA, bf16)...")
    model, tokenizer = load_mistral_model(
        model_cfg=OmegaConf.create(cfg["modelconfig"]),
        train_cfg=OmegaConf.create(cfg["trainconfig"]),
        device_map="auto",
        use_fsdp=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.print("✍️ Tokenizing...")
    ds_tok = ds.map(
        lambda ex: tokenize_batch(ex, tokenizer, cfg["dataconfig"]["max_seq_len"]),
        batched=True,
        remove_columns=ds.column_names,  # <— drop all original cols, including 'text'
        desc="Tokenizing",
    )
    data_collator = DataCollatorForLanguageModeling(
                        tokenizer=tokenizer,
                        mlm=False,
                        pad_to_multiple_of=8,  
                    )

    logger.print("🧰 Building DataLoader...")
    loader = DataLoader(
                ds_tok,
                batch_size=cfg["dataconfig"]["batch_size"],
                shuffle=bool(cfg["dataconfig"]["shuffle"]),
                num_workers=cfg["dataconfig"]["num_workers"],
                drop_last=bool(cfg["dataconfig"]["drop_last"]),
                pin_memory=bool(cfg["dataconfig"]["pin_memory"]),
                collate_fn=data_collator,        # <— this pads to the longest in batch
            )

    logger.print("📈 Setting up optimizer & scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizerconfig"]["lr"]),
        betas=tuple(cfg["optimizerconfig"]["betas"]),
        weight_decay=float(cfg["optimizerconfig"]["weight_decay"]),
    )

    steps_per_epoch = (len(loader) + cfg["trainconfig"]["accumulation_steps"] - 1) // cfg["trainconfig"]["accumulation_steps"]
    max_steps = cfg["trainconfig"]["total_steps"] or (steps_per_epoch * int(cfg["trainconfig"]["epochs"]))
    warmup_steps = int(float(cfg["schedulerconfig"]["warmup_ratio"]) * max_steps)

    scheduler = get_scheduler(
        name=cfg["schedulerconfig"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    logger.wb("log", metrics={"params/trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    logger.print("🚀 Launching training (single GPU, no FSDP)…")
    trainer = Trainer(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        logger=logger,
        rank=0,
        total_steps=max_steps,
        scheduler=scheduler,
        accumulation_steps=int(cfg["trainconfig"]["accumulation_steps"]),
        save_model_path=cfg["trainconfig"]["output_dir"],
        log_freq=int(cfg["trainconfig"]["logging_steps"]),
        save_steps=int(cfg["trainconfig"]["save_steps"]),
        max_grad_norm=float(cfg["schedulerconfig"]["max_grad_norm"]),
        use_bf16=bool(cfg["trainconfig"]["bf16"]),
        device=device,
        save_optimizer_state=False,
    )

    trainer.train(int(cfg["trainconfig"]["epochs"]))
    logger.print("✅ Done.")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
