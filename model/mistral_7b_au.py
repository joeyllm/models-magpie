from __future__ import annotations
from typing import Optional

import torch
from omegaconf import DictConfig
from transformers import BitsAndBytesConfig

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _to_dtype(bf16: bool) -> torch.dtype:
    return torch.bfloat16 if bf16 else torch.float16


def _build_bnb_config(model_cfg: DictConfig, bf16: bool) -> Optional[BitsAndBytesConfig]:
    """
    Build BitsAndBytes 4-bit config if load_in_4bit=True; otherwise return None.
    """
    use_4bit = bool(getattr(model_cfg, "load_in_4bit", False))
    if not use_4bit:
        return None

    # Allow overrides if present in config; otherwise safe defaults.
    compute = getattr(model_cfg, "bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = torch.bfloat16 if str(compute).lower() in ("bfloat16", "bf16") else (
        torch.float16 if str(compute).lower() in ("float16", "fp16", "half") else torch.float32
    )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=bool(getattr(model_cfg, "bnb_4bit_use_double_quant", True)),
        bnb_4bit_quant_type=str(getattr(model_cfg, "bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _print_trainable_params(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(1, total)
    print(f"🔧 Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")
