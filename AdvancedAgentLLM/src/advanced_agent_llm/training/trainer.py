from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from advanced_agent_llm.config import ModelConfig, TrainingConfig
from advanced_agent_llm.training.checkpointing import load_checkpoint, save_checkpoint
from advanced_agent_llm.training.logging import MetricsLogger
from advanced_agent_llm.training.losses import next_token_cross_entropy
from advanced_agent_llm.training.mixed_precision import (
    autocast_context,
    create_gradient_scaler,
)
from advanced_agent_llm.training.schedulers import cosine_learning_rate


class Pretrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        validation_loader: DataLoader | None,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        output_directory: str | Path,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.model_config = model_config
        self.config = training_config
        self.output_directory = Path(output_directory)
        self.device = device
        self.scaler = create_gradient_scaler(device, training_config.precision)
        self.logger = MetricsLogger(self.output_directory / "metrics.jsonl")
        self.step = 0
        self.tokens_seen = 0
        self._train_iterator: Iterator | None = None

    def resume(self, checkpoint_path: str | Path) -> None:
        state = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scaler,
            self.device,
        )
        self.step = int(state["step"])
        self.tokens_seen = int(state.get("tokens_seen", 0))

    def _next_training_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_loader)
        try:
            return next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_loader)
            try:
                return next(self._train_iterator)
            except StopIteration as error:
                raise RuntimeError("the training dataset produced no packed sequences") from error

    @torch.no_grad()
    def validate(self) -> float:
        if self.validation_loader is None:
            raise RuntimeError("validation was not configured")

        self.model.eval()
        losses = []
        validation_iterator = iter(self.validation_loader)
        for _ in range(self.config.validation_batches):
            try:
                inputs, targets = next(validation_iterator)
            except StopIteration:
                break
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with autocast_context(self.device, self.config.precision):
                logits = self.model(inputs)
                loss = next_token_cross_entropy(logits, targets)
            losses.append(loss.detach().float())

        self.model.train()
        if not losses:
            raise RuntimeError(
                "the validation split produced no batches; increase the corpus or "
                "validation_fraction, or set validation_batches to 0"
            )
        return torch.stack(losses).mean().item()

    def _save_checkpoint(self) -> Path:
        checkpoint_path = self.output_directory / "checkpoints" / f"step_{self.step:08d}.pt"
        return save_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scaler,
            self.step,
            self.tokens_seen,
            self.model_config,
            self.config,
        )

    def train(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        while self.step < self.config.max_steps:
            step_start = perf_counter()
            learning_rate = cosine_learning_rate(
                step=self.step,
                maximum_steps=self.config.max_steps,
                warmup_steps=self.config.warmup_steps,
                maximum_learning_rate=self.config.learning_rate,
                minimum_learning_rate_ratio=self.config.minimum_learning_rate_ratio,
            )
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            accumulated_loss = 0.0
            tokens_this_step = 0
            for _ in range(self.config.gradient_accumulation_steps):
                inputs, targets = self._next_training_batch()
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                tokens_this_step += targets.numel()

                with autocast_context(self.device, self.config.precision):
                    logits = self.model(inputs)
                    loss = next_token_cross_entropy(logits, targets)
                    scaled_loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(scaled_loss).backward()
                accumulated_loss += loss.detach().float().item()

            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1
            self.tokens_seen += tokens_this_step
            elapsed = perf_counter() - step_start

            if self.step % self.config.log_every == 0 or self.step == 1:
                self.logger.log(
                    {
                        "step": self.step,
                        "training_loss": accumulated_loss / self.config.gradient_accumulation_steps,
                        "gradient_norm": float(gradient_norm),
                        "learning_rate": learning_rate,
                        "tokens_seen": self.tokens_seen,
                        "tokens_per_second": tokens_this_step / elapsed,
                    }
                )

            should_validate = (
                self.validation_loader is not None
                and self.config.validation_batches > 0
                and self.step % self.config.validate_every == 0
            )
            if should_validate:
                self.logger.log(
                    {"step": self.step, "validation_loss": self.validate()}
                )

            if self.step % self.config.checkpoint_every == 0:
                checkpoint_path = self._save_checkpoint()
                print(f"saved checkpoint: {checkpoint_path}", flush=True)

        final_checkpoint = self._save_checkpoint()
        print(f"training complete; final checkpoint: {final_checkpoint}", flush=True)
