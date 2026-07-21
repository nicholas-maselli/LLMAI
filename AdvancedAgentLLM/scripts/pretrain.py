import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from advanced_agent_llm.config import DataConfig, ModelConfig, TrainingConfig
from advanced_agent_llm.data import PackedPretrainingDataset, Tokenizer
from advanced_agent_llm.model import LanguageModel
from advanced_agent_llm.training.optimizers import create_adamw_optimizer
from advanced_agent_llm.training.trainer import Pretrainer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the decoder-only language model.")

    data = parser.add_argument_group("data")
    data.add_argument("--tokenizer", type=Path, required=True)
    data.add_argument("--local-text", type=Path)
    data.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    data.add_argument("--dataset-subset", default="sample-10BT")
    data.add_argument("--validation-fraction", type=float, default=0.005)
    data.add_argument("--shuffle-buffer-size", type=int, default=10_000)

    model = parser.add_argument_group("model")
    model.add_argument("--context-length", type=int, default=2_048)
    model.add_argument("--hidden-size", type=int, default=768)
    model.add_argument("--layers", type=int, default=12)
    model.add_argument("--attention-heads", type=int, default=12)
    model.add_argument("--kv-heads", type=int, default=4)
    model.add_argument("--feed-forward-multiplier", type=float, default=8 / 3)
    model.add_argument("--rope-base", type=float, default=10_000.0)

    training = parser.add_argument_group("training")
    training.add_argument("--run-name", default="advanced-agent-llm")
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--gradient-accumulation", type=int, default=4)
    training.add_argument("--steps", type=int, default=10_000)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    training.add_argument("--warmup-steps", type=int, default=500)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--gradient-clip", type=float, default=1.0)
    training.add_argument("--log-every", type=int, default=10)
    training.add_argument("--validate-every", type=int, default=500)
    training.add_argument("--validation-batches", type=int, default=20)
    training.add_argument("--checkpoint-every", type=int, default=500)
    training.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    training.add_argument("--output-directory", type=Path)
    training.add_argument("--resume", type=Path)
    return parser.parse_args()


def select_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_run_configuration(
    output_directory: Path,
    model_config: ModelConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    configuration = {
        "model": asdict(model_config),
        "data": asdict(data_config),
        "training": asdict(training_config),
    }
    with (output_directory / "config.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2, default=str)


def main() -> None:
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)

    tokenizer = Tokenizer(arguments.tokenizer)
    model_config = ModelConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        context_length=arguments.context_length,
        hidden_size=arguments.hidden_size,
        number_of_layers=arguments.layers,
        number_of_attention_heads=arguments.attention_heads,
        number_of_key_value_heads=arguments.kv_heads,
        feed_forward_multiplier=arguments.feed_forward_multiplier,
        rope_base=arguments.rope_base,
    )
    data_config = DataConfig(
        dataset_name=arguments.dataset,
        dataset_subset=arguments.dataset_subset,
        shuffle_buffer_size=arguments.shuffle_buffer_size,
        seed=arguments.seed,
        tokenizer_path=arguments.tokenizer,
        local_text_path=arguments.local_text,
        validation_fraction=arguments.validation_fraction,
    )
    training_config = TrainingConfig(
        run_name=arguments.run_name,
        batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation,
        max_steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        minimum_learning_rate_ratio=arguments.minimum_lr_ratio,
        warmup_steps=arguments.warmup_steps,
        weight_decay=arguments.weight_decay,
        gradient_clip_norm=arguments.gradient_clip,
        log_every=arguments.log_every,
        validate_every=arguments.validate_every,
        validation_batches=arguments.validation_batches,
        checkpoint_every=arguments.checkpoint_every,
        seed=arguments.seed,
        precision=arguments.precision,
        number_of_workers=arguments.workers,
    )

    if arguments.resume is not None and arguments.output_directory is None:
        output_directory = arguments.resume.resolve().parent.parent
    elif arguments.output_directory is not None:
        output_directory = arguments.output_directory
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_directory = Path("outputs/pretrain") / f"{arguments.run_name}_{timestamp}"

    write_run_configuration(output_directory, model_config, data_config, training_config)

    train_dataset = PackedPretrainingDataset(
        data_config,
        tokenizer,
        model_config.context_length,
        split="train",
    )
    validation_dataset = PackedPretrainingDataset(
        data_config,
        tokenizer,
        model_config.context_length,
        split="validation",
    )
    loader_options = {
        "batch_size": training_config.batch_size,
        "num_workers": training_config.number_of_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if training_config.number_of_workers > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, **loader_options)
    validation_loader = (
        DataLoader(validation_dataset, **loader_options)
        if training_config.validation_batches > 0
        else None
    )

    device = select_device(arguments.device)
    model = LanguageModel(model_config).to(device)
    optimizer = create_adamw_optimizer(
        model,
        training_config.learning_rate,
        training_config.weight_decay,
        fused=device.type == "cuda",
    )
    print(
        f"device: {device} | parameters: {model.parameter_count():,} | "
        f"vocabulary: {tokenizer.vocabulary_size:,}",
        flush=True,
    )

    trainer = Pretrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        model_config=model_config,
        training_config=training_config,
        output_directory=output_directory,
        device=device,
    )
    if arguments.resume is not None:
        trainer.resume(arguments.resume)
        print(f"resumed from step {trainer.step}: {arguments.resume}", flush=True)
    trainer.train()


if __name__ == "__main__":
    main()
