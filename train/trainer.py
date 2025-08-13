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
