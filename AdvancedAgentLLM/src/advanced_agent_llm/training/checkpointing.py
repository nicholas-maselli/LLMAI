from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from advanced_agent_llm.config import ModelConfig, TrainingConfig


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    tokens_seen: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, temporary_path)
    temporary_path.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state.get("scaler", {}))
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return state
