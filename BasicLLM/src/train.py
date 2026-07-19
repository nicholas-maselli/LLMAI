import argparse
import csv
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from config import ModelConfig
from data import create_batch, create_tensor, tokenize_dataset
from model.language_model import TinyLanguageModel


OUTPUTS_DIRECTORY = Path(__file__).parent.parent / "outputs"


def parse_config() -> ModelConfig:
    defaults = ModelConfig()
    parser = argparse.ArgumentParser(description="Train the tiny language model.")
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--context-length", type=int, default=defaults.context_length)
    parser.add_argument("--embedding-size", type=int, default=defaults.embedding_size)
    parser.add_argument("--heads", type=int, default=defaults.number_of_heads)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--steps", type=int, default=defaults.training_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--log-every", type=int, default=defaults.logging_interval)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=defaults.checkpoint_interval,
    )
    arguments = parser.parse_args()

    return ModelConfig(
        model_name=arguments.model_name,
        context_length=arguments.context_length,
        embedding_size=arguments.embedding_size,
        number_of_heads=arguments.heads,
        batch_size=arguments.batch_size,
        training_steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        generation_length=defaults.generation_length,
        logging_interval=arguments.log_every,
        checkpoint_interval=arguments.checkpoint_every,
    )


def train_model(
    model: TinyLanguageModel,
    data: torch.Tensor,
    config: ModelConfig,
    log_path: Path,
    checkpoint_directory: Path,
    characters: list[str],
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    model.train()

    start_time = perf_counter()

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=["step", "loss", "gradient_norm", "learning_rate", "elapsed_seconds"],
        )
        log_writer.writeheader()

        for step in range(1, config.training_steps + 1):
            inputs, targets = create_batch(
                data,
                config.batch_size,
                config.context_length,
            )

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()

            gradient_norm = math.sqrt(
                sum(
                    parameter.grad.detach().norm(2).item() ** 2
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )

            optimizer.step()

            should_log = (
                step % config.logging_interval == 0
                or step == config.training_steps
            )
            if should_log:
                metrics = {
                    "step": step,
                    "loss": loss.item(),
                    "gradient_norm": gradient_norm,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": perf_counter() - start_time,
                }
                log_writer.writerow(metrics)
                log_file.flush()
                print(
                    f"step {step} | loss {metrics['loss']:.3f} | "
                    f"gradient norm {gradient_norm:.3f} | "
                    f"lr {metrics['learning_rate']:.6f}"
                )

            should_save_checkpoint = (
                step % config.checkpoint_interval == 0
                or step == config.training_steps
            )
            if should_save_checkpoint:
                checkpoint_path = checkpoint_directory / f"step_{step:06d}.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": asdict(config),
                        "characters": characters,
                        "step": step,
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint to {checkpoint_path}")


def main() -> None:
    config = parse_config()
    tokenizer, token_ids = tokenize_dataset()
    data = create_tensor(token_ids)

    model = TinyLanguageModel(
        vocabulary_size=len(tokenizer.characters),
        context_length=config.context_length,
        embedding_size=config.embedding_size,
        number_of_heads=config.number_of_heads,
    )

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_output_directory = OUTPUTS_DIRECTORY / "train" / config.model_name
    checkpoint_directory = model_output_directory / "checkpoints" / run_name
    log_directory = model_output_directory / "logs" / run_name
    checkpoint_directory.mkdir(parents=True)
    log_directory.mkdir(parents=True)

    train_model(
        model=model,
        data=data,
        config=config,
        log_path=log_directory / "training_metrics.csv",
        checkpoint_directory=checkpoint_directory,
        characters=tokenizer.characters,
    )


if __name__ == "__main__":
    main()
