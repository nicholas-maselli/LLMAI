import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model.language_model import TinyLanguageModel
from tokenizer import CharacterVocabulary


OUTPUTS_DIRECTORY = Path(__file__).parent.parent / "outputs"


@torch.no_grad()
def generate_text(
    model: TinyLanguageModel,
    tokenizer: CharacterVocabulary,
    prompt: str,
    context_length: int,
    generation_length: int,
) -> str:
    token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    model.eval()

    for _ in range(generation_length):
        recent_token_ids = token_ids[:, -context_length:]
        logits = model(recent_token_ids)
        next_token_logits = logits[:, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        token_ids = torch.cat((token_ids, next_token_id), dim=1)

    return tokenizer.decode(token_ids[0].tolist())


def find_latest_checkpoint() -> Path:
    checkpoints = list(
        (OUTPUTS_DIRECTORY / "train").glob("*/checkpoints/*/step_*.pt")
    )
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found. Run train.py first.")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt", default="def ")
    parser.add_argument("--tokens", type=int)
    arguments = parser.parse_args()

    checkpoint_path = arguments.checkpoint or find_latest_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    tokenizer = CharacterVocabulary("".join(checkpoint["characters"]))

    model = TinyLanguageModel(
        vocabulary_size=len(tokenizer.characters),
        context_length=config.context_length,
        embedding_size=config.embedding_size,
        number_of_heads=config.number_of_heads,
    )
    model.load_state_dict(checkpoint["model_state"])

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=arguments.prompt,
        context_length=config.context_length,
        generation_length=arguments.tokens or config.generation_length,
    )
    print(generated_text)


if __name__ == "__main__":
    main()
