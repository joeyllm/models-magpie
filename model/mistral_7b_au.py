from __future__ import annotations
from typing import Optional

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

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


def _apply_lora(model, model_cfg: DictConfig):
    r = int(getattr(model_cfg, "lora_r", 0))
    if r <= 0:
        return model

    lora_config = LoraConfig(
        r=r,
        lora_alpha=int(getattr(model_cfg, "lora_alpha", 32)),
        lora_dropout=float(getattr(model_cfg, "lora_dropout", 0.05)),
        target_modules=list(getattr(model_cfg, "target_modules", DEFAULT_TARGET_MODULES)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def load_mistral_model(
    model_cfg: DictConfig,
    train_cfg: Optional[DictConfig] = None,
    device_map: Optional[Union[str, Dict[str, int]]] = "auto",
    use_fsdp: bool = False,
):
    """
    Load Mistral‑7B in the required format for fine‑tuning.

    Args:
        model_cfg: DictConfig with fields:
            - name: HF repo id (e.g., "mistralai/Mistral-7B-v0.3")
            - load_in_4bit: bool
            - [optional bnb_* keys]
            - lora_r, lora_alpha, lora_dropout, target_modules
            - trust_remote_code: bool (optional, default True)
        train_cfg: DictConfig with fields:
            - bf16: bool
            - gradient_checkpointing: bool
        device_map: "auto" | None | dict; when using FSDP set to None.
        use_fsdp: If True, force device_map=None (HF Trainer/FSDP manages placement).

    Returns:
        model, tokenizer
    """
    name = getattr(model_cfg, "name")
    trust_remote_code = bool(getattr(model_cfg, "trust_remote_code", True))
    bf16 = bool(getattr(train_cfg, "bf16", True)) if train_cfg is not None else True
    grad_ckpt = bool(getattr(train_cfg, "gradient_checkpointing", True)) if train_cfg is not None else True

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization (optional)
    quantization_config = _build_bnb_config(model_cfg, bf16)

    # FSDP note: when using FSDP via HF Trainer, device_map must be None.
    dm = None if use_fsdp else device_map

    model = AutoModelForCausalLM.from_pretrained(
        name,
        trust_remote_code=trust_remote_code,
        torch_dtype=_to_dtype(bf16),
        device_map=dm,
        quantization_config=quantization_config,
    )

    # Apply LoRA (QLoRA)
    model = _apply_lora(model, model_cfg)

    # Enable grad checkpointing and disable cache for training
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False  # important for checkpointing

    _print_trainable_params(model)
    return model, tokenizer