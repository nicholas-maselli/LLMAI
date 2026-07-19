# Tiny Transformer — One Step at a Time

See [TRAINING.md](TRAINING.md) for training commands and [GENERATION.md](GENERATION.md) for text-generation commands.

We are building a tiny language model incrementally. The code intentionally contains only the concepts discussed so far.

Current step:

1. Load the text in `dataset.txt`.
2. Discover the unique characters that make up its vocabulary.
3. Encode the dataset and create training batches.
4. Convert token IDs into learnable embedding vectors.

The project is organized by concept:

- `src/tokenizer.py` converts between characters and token IDs.
- `src/data.py` loads the dataset and creates training examples and batches.
- `src/model/` separates the model into focused components.
  - `embeddings.py` contains token, position, and input embeddings.
  - `transformer_block/` contains attention and the other block components.
  - `language_model.py` assembles the complete four-block language model.
- `src/config.py` contains model and training settings.
- `src/train.py` contains the training loop.
- `src/generate.py` contains next-token generation.
- `src/main.py` connects the complete pipeline.
- `datasets/dataset.txt` contains the raw training text.

Run the complete model from the `BasicLLM` directory:

```bash
uv run src/train.py
uv run src/generate.py
```

Training and generation are independent. Checkpoints are written to `outputs/train/<model_name>/checkpoints/` at the interval configured by `checkpoint_interval`. `generate.py` loads the most recent checkpoint by default. Use `--checkpoint`, `--prompt`, and `--tokens` to override its inputs.
