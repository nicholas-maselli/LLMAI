import argparse
from itertools import islice
from pathlib import Path

from advanced_agent_llm.config import DataConfig
from advanced_agent_llm.data.pretrain_dataset import iterate_documents
from advanced_agent_llm.data.tokenizer import train_bpe_tokenizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pretraining BPE tokenizer.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--vocabulary-size", type=int, default=50_304)
    parser.add_argument("--minimum-frequency", type=int, default=2)
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--local-text", type=Path)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-subset", default="sample-10BT")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    data_config = DataConfig(
        dataset_name=arguments.dataset,
        dataset_subset=arguments.dataset_subset,
        local_text_path=arguments.local_text,
    )
    documents = islice(
        iterate_documents(data_config, shuffle=True),
        arguments.documents,
    )
    output_path = train_bpe_tokenizer(
        documents=documents,
        output_path=arguments.output,
        vocabulary_size=arguments.vocabulary_size,
        minimum_frequency=arguments.minimum_frequency,
    )
    print(f"saved tokenizer: {output_path}")


if __name__ == "__main__":
    main()
