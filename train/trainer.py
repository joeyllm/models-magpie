from typing import Any
import torch
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler


class Trainer:
    def __init__(
        self,
        model: Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        logger: Any,
        rank: int,
        total_steps: int,
        scheduler_cfg,
        accumulation_steps: int,
        save_model_path: str,
        log_freq: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.rank = rank
        self.device = device or torch.device(f"cuda:{rank}")
        self.model = model  # Already wrapped in FSDP
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.logger = logger
        self.loss_fn = CrossEntropyLoss()
        self.global_step = 0
        self.accumulation_steps = accumulation_steps
        self.save_model_path = save_model_path
        self.log_freq = log_freq
        self.scaler = GradScaler()

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=scheduler_cfg.max_lr,
            total_steps=total_steps,
            pct_start=scheduler_cfg.pct_start,
            anneal_strategy=scheduler_cfg.anneal_strategy,
            div_factor=scheduler_cfg.div_factor,
            final_div_factor=scheduler_cfg.final_div_factor,
            cycle_momentum=scheduler_cfg.cycle_momentum,
            base_momentum=scheduler_cfg.base_momentum,
            max_momentum=scheduler_cfg.max_momentum,
            three_phase=scheduler_cfg.three_phase,
        )

        self.logger.print(f"Trainer initialized on rank {rank}")

    def save_model(self, path=None):
        path = path or self.save_model_path
        try:
            state_dict = self.model.state_dict()
            torch.save(state_dict, path)
            if self.logger.is_main:
                self.logger.print(f"Model saved at {path}")
        except Exception as e:
            self.logger.print(f"Save error: {e}")