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



