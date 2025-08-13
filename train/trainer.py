from typing import Any
import torch
from torch.utils.data import DataLoader
from torch.nn import Module


class Trainer:
    def __init__(
        self,
        model: Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        rank: int,
        total_steps: int,
        scheduler_cfg,
        accumulation_steps: int,
        save_model_path: str,
        log_freq: int,
        logger: Any,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.rank = rank
        self.total_steps = total_steps
        self.scheduler_cfg = scheduler_cfg
        self.accumulation_steps = accumulation_steps
        self.save_model_path = save_model_path
        self.log_freq = log_freq
        self.device = device
        self.logger = logger

    def save_model(self, path=None):
        path = path or self.save_model_path
        try:
            state_dict = self.model.state_dict()
            torch.save(state_dict, path)
            if self.logger.is_main:
                self.logger.print(f"💾 Model saved at {path}")
        except Exception as e:
            self.logger.print(f" Save error: {e}")