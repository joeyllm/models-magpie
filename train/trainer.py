# train/trainer.py
from __future__ import annotations
from typing import Any, Optional, Dict
import os
import math
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.amp import autocast, GradScaler
import json

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
except Exception:
    FSDP = None
from peft.utils.save_and_load import get_peft_model_state_dict


class Trainer:

    def __init__(
        self,
        model: Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        logger: Any,
        rank: int,
        total_steps: int,
        scheduler,                          # torch scheduler (or None)
        accumulation_steps: int,
        save_model_path: str,               # base output_dir
        log_freq: int,
        save_steps: int,                    # checkpoint every N steps
        max_grad_norm: float,
        use_bf16: bool = True,
        device: Optional[torch.device] = None,
        save_optimizer_state: bool = False, # toggle to also save opt/sched state_dict
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.logger = logger
        self.rank = rank
        self.device = device or torch.device(f"cuda:{rank}")
        self.total_steps = total_steps
        self.scheduler = scheduler
        self.accumulation_steps = max(1, accumulation_steps)
        self.output_dir = save_model_path
        self.log_freq = log_freq
        self.save_steps = max(1, save_steps)
        self.save_optimizer_state = save_optimizer_state
        self.max_grad_norm = float(max_grad_norm)
        self.global_step = 0
        self.best_loss = math.inf

        # Mixed precision policy
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.scaler = None if use_bf16 else GradScaler()

        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        self.logger.print(
            f"Trainer init (rank={rank}, dtype={self.dtype}, accum={self.accumulation_steps}, "
            f"save_steps={self.save_steps})"
        )

    # ---------------------
    # Utilities
    # ---------------------
    def _unwrap(self, m: Module) -> Module:
        return getattr(m, "module", m)

    def _save_checkpoint(self, tag: str, metrics: Optional[Dict[str, float]] = None):
        # if self.rank != 0:
        #     return

        ckpt_dir = os.path.join(self.output_dir, f"ckpt-{tag}")
        os.makedirs(ckpt_dir, exist_ok=True)

        try:
            model_for_sd = self.model  # could be FSDP-wrapped
            unwrapped = getattr(model_for_sd, "module", model_for_sd)

            # 1) Build a CPU, rank0-only full state_dict if FSDP is present and the model is wrapped
            use_fsdp_ctx = FSDP is not None and isinstance(model_for_sd, FSDP)
            if use_fsdp_ctx:
                full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model_for_sd, StateDictType.FULL_STATE_DICT, full_cfg):
                    base_sd = model_for_sd.state_dict()
            else:
                # Not wrapped (or FSDP not available) → normal CPU state_dict
                base_sd = {k: v.cpu() for k, v in model_for_sd.state_dict().items()}

            # 2) Filter to adapter-only weights (LoRA)
            peft_sd = get_peft_model_state_dict(model_for_sd, state_dict=base_sd)

            # 3) Let PEFT write everything (adapter_model.safetensors + adapter_config.json)
            unwrapped.save_pretrained(
                ckpt_dir,
                state_dict=peft_sd,
                safe_serialization=True,
            )

            # 4) Small training meta
            torch.save(
                {"global_step": self.global_step, "best_loss": self.best_loss, "metrics": metrics or {}},
                os.path.join(ckpt_dir, "training_state.pt"),
            )

            self.logger.print(f"✅ Adapter checkpoint saved at: {ckpt_dir}")

        except Exception as e:
            self.logger.print(f"❌ Adapter checkpoint failed: {e}")


    def checkpoint_step(self, loss_val: float):
        if (self.global_step % self.save_steps) == 0 and self.global_step > 0:
            self._save_checkpoint(tag=f"step-{self.global_step}", metrics={"loss": loss_val})

    def _maybe_update_best(self, loss_val: float):
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self._save_checkpoint(tag="best", metrics={"loss": loss_val})

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)

        labels = batch.get("labels")
        if labels is None and "target_ids" in batch:
            labels = batch["target_ids"]
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)

        with autocast(device_type="cuda", dtype=self.dtype):
            if labels is not None:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
            else:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
        return loss

    # ---------------------
    # Train Loop
    # ---------------------
    def epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        running_count = 0

        for step, batch in enumerate(self.dataloader):
            is_accum_start = (step % self.accumulation_steps) == 0
            is_accum_end = ((step + 1) % self.accumulation_steps) == 0

            if is_accum_start:
                self.optimizer.zero_grad(set_to_none=True)

            maybe_no_sync = (
                self.model.no_sync
                if hasattr(self.model, "no_sync") and not is_accum_end
                else None
            )

            def _backward_scaled(loss):
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            context = maybe_no_sync() if maybe_no_sync is not None else torch.enable_grad()
            with context:
                loss = self._compute_loss(batch)
                running_loss += float(loss.detach().item())
                running_count += 1

                if is_accum_start and self.logger.is_main:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f} LR: {lr:.6f}")
                    self.logger.wb("log", metrics={"train/loss": loss.item(), "train/lr": lr}, step=self.global_step)

                if not torch.isfinite(loss):
                    if self.logger.is_main:
                        self.logger.print(f"⚠️ Non-finite loss at step {self.global_step}: {loss.item()} — skipping.")
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_step += 1
                    continue

                loss = loss / self.accumulation_steps
                _backward_scaled(loss)

            if is_accum_end:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]
                if self.logger.is_main and (step % self.accumulation_steps) == 0:
                    self.logger.wb("log", metrics={"train/lr": lr}, step=self.global_step)
                

            self.global_step += 1

            # Step-based checkpointing
            avg_loss = running_loss / max(1, running_count)
            self.checkpoint_step(loss_val=avg_loss)

            # Periodic lightweight save for visibility
            if (self.global_step % self.log_freq == 0) and self.logger.is_main:
                self.logger.print("Periodic save (log_freq).")
                self._save_checkpoint(tag=f"logfreq-{self.global_step}", metrics={"loss": avg_loss})

        # End-of-epoch checkpoint + best tracker
        epoch_avg = running_loss / max(1, running_count)
        if self.logger.is_main:
            self.logger.print(f"Epoch {epoch} average loss: {epoch_avg:.4f}")
        self._save_checkpoint(tag=f"epoch-{epoch}", metrics={"epoch_loss": epoch_avg})
        self._maybe_update_best(epoch_avg)

        # Finalize leftover grads if the epoch ended mid-accumulation
        remainder = len(self.dataloader) % self.accumulation_steps
        if remainder != 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.epoch(epoch)
        # Always save a "last" checkpoint at the end
        if self.logger.is_main:
            self._save_checkpoint(tag="last")
            self.logger.print("🏁 Training complete.")
