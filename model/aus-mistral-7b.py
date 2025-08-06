from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from omegaconf import DictConfig
import torch

def load_mistral_model(model_cfg: DictConfig, device_map=None):
    """
    Load Mistral model in 4-bit with LoRA.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # BitsAndBytes 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg.load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load pre-trained model with 4-bit weights
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map or {"": 0}
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        target_modules=model_cfg.target_modules,
        lora_dropout=model_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🔧 Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model
