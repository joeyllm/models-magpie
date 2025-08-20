from typing import List, Literal, Tuple
from pydantic import BaseModel, Field


# ---------- Sections ----------

class WandBConfig(BaseModel):
    project: str
    name: str
    mode: Literal["online", "offline", "disabled"]


class DataConfig(BaseModel):
    data_path: str
    batch_size: int = Field(..., gt=0)
    num_workers: int = Field(0, ge=0)
    shuffle: bool
    drop_last: bool
    pin_memory: bool
    max_seq_len: int = Field(..., gt=0)


class ModelConfig(BaseModel):
    name: str
    load_in_4bit: bool = False
    lora_r: int = Field(..., gt=0)
    lora_alpha: int = Field(..., gt=0)
    lora_dropout: float = Field(..., ge=0.0, le=1.0)
    target_modules: List[str]


class OptimizerConfig(BaseModel):
    lr: float = Field(..., gt=0)
    betas: Tuple[float, float]
    weight_decay: float = Field(..., ge=0.0)


class TrainConfig(BaseModel):
    epochs: int = Field(..., gt=0)
    total_steps: int = Field(0, ge=0)
    accumulation_steps: int = Field(..., gt=0)
    output_dir: str
    logging_steps: int = Field(..., gt=0)
    save_steps: int = Field(..., gt=0)
    save_total_limit: int = Field(..., ge=0)
    gradient_checkpointing: bool
    bf16: bool
    seed: int
    packing: bool


class SchedulerConfig(BaseModel):
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup"
    ]
    warmup_ratio: float = Field(..., ge=0.0, le=1.0)
    max_grad_norm: float = Field(..., ge=0.0)


# ---------- Root Config ----------

class Config(BaseModel):
    wandbconfig: WandBConfig
    dataconfig: DataConfig
    modelconfig: ModelConfig
    optimizerconfig: OptimizerConfig
    trainconfig: TrainConfig
    schedulerconfig: SchedulerConfig