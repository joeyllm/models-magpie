from __future__ import annotations
from typing import Any, Optional, Dict

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.amp import autocast, GradScaler

try:
    # FSDP-aware no_sync for grad-accumulation
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:
    FSDP = None